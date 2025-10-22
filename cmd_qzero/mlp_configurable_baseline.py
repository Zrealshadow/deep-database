import torch
import math
import torch_frame
import argparse
import copy
from tqdm import tqdm

from torch_geometric.nn import MLP
from torch.nn import L1Loss, BCEWithLogitsLoss
from sklearn.metrics import mean_absolute_error, roc_auc_score
from relbench.base import TaskType

import sys
from pathlib import Path

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.base import construct_stype_encoder_dict, default_stype_encoder_cls_kwargs
from utils.resource import get_text_embedder_cfg
from utils.data import TableData
from torch_frame.nn.encoder import StypeWiseFeatureEncoder


class CustomMLP(torch.nn.Module):
    """Custom MLP with StypeWiseFeatureEncoder + torch_geometric MLP"""
    
    def __init__(self, 
                 col_names_dict, 
                 col_stats, 
                 stype_encoder_dict,
                 channels: int,
                 hidden_channels: list,
                 out_channels: int,
                 dropout: float = 0.2,
                 norm: str = "batch_norm"):
        super().__init__()
        
        # Feature encoder
        self.encoder = StypeWiseFeatureEncoder(
            out_channels=channels,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict
        )
        
        # Calculate input dimension (num_features * channels)
        num_features = sum(len(cols) for cols in col_names_dict.values())
        in_channels = num_features * channels
        
        # MLP with custom hidden layers
        # torch_geometric.MLP expects channel_list format
        # channel_list = [in_channels, hidden1, hidden2, ..., out_channels]
        channel_list = [in_channels] + hidden_channels + [out_channels]
        
        self.mlp = MLP(
            channel_list=channel_list,
            dropout=dropout,
            norm=norm,
        )
    
    def forward(self, tf: torch_frame.TensorFrame):
        # Encode features: [B, F, C]
        x, _ = self.encoder(tf)
        
        # Flatten: [B, F*C]
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # MLP
        x = self.mlp(x)
        
        return x


parser = argparse.ArgumentParser(description="Custom MLP Training Script")

parser.add_argument("--data_dir", type=str, required=True,
                    help="Path to the data directory.")
parser.add_argument("--channels", type=int, default=128,
                    help="Embedding dimension (channels for encoder).")
parser.add_argument("--hidden_channels", type=str, required=True,
                    help="Hidden layer sizes, e.g., '512-192-160-160'")
parser.add_argument("--dropout", type=float, default=0.2,
                    help="Dropout probability.")
parser.add_argument("--norm", type=str, choices=["batch_norm", "layer_norm", "none"],
                    default="batch_norm", help="Normalization type.")

# Training parameters
parser.add_argument("--batch_size", type=int, default=256,
                    help="Batch size for training.")
parser.add_argument("--lr", type=float, default=0.001,
                    help="Learning rate.")
parser.add_argument("--num_epochs", type=int, default=200,
                    help="Number of training epochs.")
parser.add_argument("--early_stop_threshold", type=int, default=10,
                    help="Number of epochs to wait for improvement before early stopping.")
parser.add_argument("--max_round_epoch", type=int, default=20,
                    help="Maximum number of epochs per round.")

args = parser.parse_args()

# Parse hidden_channels
hidden_channels = [int(x) for x in args.hidden_channels.split('-')]

# Device (auto-detect)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
table_data = TableData.load_from_dir(args.data_dir)

if not table_data.is_materialize:
    text_cfg = get_text_embedder_cfg(device="cpu")
    table_data.materilize(col_to_text_embedder_cfg=text_cfg)

# Construct encoder
stype_encoder_dict = construct_stype_encoder_dict(
    default_stype_encoder_cls_kwargs,
)

# Create model
net = CustomMLP(
    col_names_dict=table_data.col_names_dict,
    col_stats=table_data.col_stats,
    stype_encoder_dict=stype_encoder_dict,
    channels=args.channels,
    hidden_channels=hidden_channels,
    out_channels=1,
    dropout=args.dropout,
    norm=args.norm if args.norm != "none" else None,
)

# Task type and loss function
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


# Deactivate dropout layer in regression task
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
