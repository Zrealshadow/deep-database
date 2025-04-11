
import torch
import math
import torch_frame
import argparse
import copy
from tqdm import tqdm

from torch_frame.nn.models import MLP, ResNet, FTTransformer
from torch.nn import L1Loss, BCEWithLogitsLoss
from sklearn.metrics import mean_absolute_error, roc_auc_score
from relbench.base import TaskType

import sys
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from model.base import construct_stype_encoder_dict, default_stype_encoder_cls_kwargs
from utils.resource import get_text_embedder_cfg
from utils.data import TableData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description="Model configuration parser")

parser.add_argument("--data_dir", type=str, required=True,
                    help="Path to the data directory.")
parser.add_argument("--channels", type=int, default=128,
                    help="Number of input channels.")
parser.add_argument("--out_channels", type=int, default=1,
                    help="Number of output channels.")
parser.add_argument("--num_layers", type=int, default=2,
                    help="Number of layers in the model.")
parser.add_argument("--dropout_prob", type=float,
                    default=0.2, help="Dropout probability.")
parser.add_argument("--norm", type=str, choices=[
                    "layer_norm", "batch_norm", "none"], default="layer_norm", help="Normalization type.")
parser.add_argument("--model", type=str,
                    choices=["MLP", "FTTrans", "ResNet"], default="MLP", help="Model architecture type.")

# --- training parameters

parser.add_argument("--batch_size", type=int, default=256,
                    help="Batch size for training.")
parser.add_argument("--lr", type=float, default=0.001,
                    help="Learning rate.")
parser.add_argument("--num_epochs", type=int, default=200,
                    help="Number of training epochs.")
parser.add_argument("--early_stop_threshold", type=int, default=10,
                    help="Number of epochs to wait for improvement before early stopping.")
parser.add_argument("--max_round_epoch", type=int,
                    default=20, help="Maximum number of epochs per round.")


args = parser.parse_args()


table_data = TableData.load_from_dir(args.data_dir)\
    
if not table_data.is_materialize:
    text_cfg = get_text_embedder_cfg(
        device = "cpu"
    )
    table_data.materilize(
        col_to_text_embedder_cfg=text_cfg,
    )

stype_encoder_dict = construct_stype_encoder_dict(
    default_stype_encoder_cls_kwargs,
)

model_args = {
    "channels": args.channels,
    "out_channels": args.out_channels,
    "num_layers": args.num_layers,
    "dropout_prob": args.dropout_prob,
    "normalization": args.norm,
    "col_names_dict": table_data.col_names_dict,
    "stype_encoder_dict": stype_encoder_dict,
    "col_stats": table_data.col_stats,
}

if args.model == "MLP":
    net = MLP(**model_args)
elif args.model == "ResNet":
    net = ResNet(**model_args)
elif args.model == "FTTrans":
    model_args.pop("normalization")
    model_args.pop("dropout_prob")
    net = FTTransformer(**model_args)


if table_data.task_type == TaskType.REGRESSION:
    loss_fn = L1Loss()
    evaluate_matric_func = mean_absolute_error
    higher_is_better = False
    is_regression = True
elif table_data.task_type == TaskType.BINARY_CLASSIFICATION:
    loss_fn = BCEWithLogitsLoss()
    evaluate_matric_func = roc_auc_score
    higher_is_better = True
    is_regression = False

batch_size = args.batch_size
lr = args.lr
num_epochs = args.num_epochs
early_stop_threshold = args.early_stop_threshold
max_round_epoch = args.max_round_epoch

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, net.parameters()), lr=lr)


data_loaders = {
    idx: torch_frame.data.DataLoader(
        getattr(table_data, f"{idx}_tf"),
        batch_size=batch_size,
        shuffle=idx == "train",
        pin_memory=True,
    )
    for idx in ["train", "val", "test"]
}


def deactivate_dropout(net: torch.nn.Module):
    deactive_nn_instances = (
        torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)
    for module in net.modules():
        if isinstance(module, deactive_nn_instances):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False


def test(net: torch.nn.Module, loader: torch.utils.data.DataLoader, early_stop: int = -1, is_regression: bool = False):
    pred_list = []
    y_list = []
    early_stop = early_stop if early_stop > 0 else len(loader.dataset)

    if not is_regression:
        net.eval()

    for idx, batch in tqdm(enumerate(loader), total=len(loader), leave=False, desc="Testing"):
        with torch.no_grad():
            batch = batch.to(device)
            y = batch.y.float()
            pred = net(batch)
            pred = pred.view(-1) if pred.size(1) == 1 else pred
            pred_list.append(pred.detach().cpu())
            y_list.append(y.detach().cpu())
        if idx > early_stop:
            break
    pred_list = torch.cat(pred_list, dim=0)
    pred_logits = pred_list
    pred_list = torch.sigmoid(pred_list)
    y_list = torch.cat(y_list, dim=0).numpy()
    return pred_logits.numpy(), pred_list.numpy(),  y_list


#  deactivate dropout layer in regression task
if is_regression:
    deactivate_dropout(net)

net.to(device)
patience = 0
best_epoch = 0
best_val_metric = -math.inf if higher_is_better else math.inf
best_model_state = None
for epoch in range(num_epochs):
    loss_accum = count_accum = 0
    net.train()
    for idx, batch in tqdm(enumerate(data_loaders["train"]),
                           leave=False,
                           total=len(data_loaders["train"]),
                           desc=f"Epoch {epoch} =>"):

        if idx > max_round_epoch:
            break

        optimizer.zero_grad()
        batch = batch.to(device)
        pred = net(batch)
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        y = batch.y.float()
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        loss_accum += loss.item()
        count_accum += 1

    train_loss = loss_accum / count_accum
    val_logits, _, val_pred_hat = test(
        net, data_loaders["val"], is_regression=is_regression)

    val_metric = evaluate_matric_func(val_pred_hat, val_logits)

    print(
        f"==> Epcoh: {epoch} => Train Loss: {train_loss:.6f}, Val {evaluate_matric_func.__name__} Metric: {val_metric:.6f}")
    if (higher_is_better and val_metric > best_val_metric) or \
       (not higher_is_better and val_metric < best_val_metric):
        best_val_metric = val_metric
        best_epoch = epoch
        best_model_state = copy.deepcopy(net.state_dict())
        patience = 0

        test_logits, _, test_pred_hat = test(
            net, data_loaders["test"], is_regression=is_regression)
        test_metric = evaluate_matric_func(test_pred_hat, test_logits)

        print(
            f"Update the best scores => Test {evaluate_matric_func.__name__} Metric: {test_metric:.6f}")
    else:
        patience += 1
        if patience > early_stop_threshold:
            print(f"Early stopping at epoch {epoch}")
            break


net.load_state_dict(best_model_state)
test_logits, _, test_pred_hat = test(
    net, data_loaders["test"], is_regression=is_regression)
test_metric = evaluate_matric_func(test_pred_hat, test_logits)
print(
    f"Test {evaluate_matric_func.__name__} Metric: {test_metric:.6f}")
