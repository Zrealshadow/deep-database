from utils.logger import ModernLogger
import random
import torch
import math
import argparse
import copy
import numpy as np
import time
from typing import Dict

from torch_geometric.loader import NeighborLoader
from torch.nn import L1Loss, BCEWithLogitsLoss
from sklearn.metrics import mean_absolute_error, roc_auc_score
from relbench.base import TaskType

from utils.util import load_col_types
from utils.resource import get_text_embedder_cfg
from utils.builder import build_pyg_hetero_graph
from utils.data import DatabaseFactory
from utils.sample import get_node_train_table_input_with_sample
from utils.util import setup_torch
from model.aida import construct_default_AIDAXFormer

setup_torch()

parser = argparse.ArgumentParser()

# Dataset and cache
parser.add_argument('--tf_cache_dir', type=str, required=True,
                    help='TensorFrame cache directory')
parser.add_argument('--db_name', type=str, required=True, help='Database name')
parser.add_argument('--task_name', type=str, required=True, help='Task name')
parser.add_argument("--device", type=str, default="auto",
                    help="Device to use for training. Use 'auto' to randomly select from available GPUs.")

# Sampling settings
parser.add_argument('--validation_ratio', type=float,
                    default=1.0, help='Validation sampling ratio')
parser.add_argument('--test_ratio', type=float,
                    default=1.0, help='Test sampling ratio')
parser.add_argument('--num_neighbors', nargs='+', type=int,
                    default=[128, 128], help='Neighbor sampling sizes')
parser.add_argument('--sample_strategy', type=str, default='last',
                    choices=['last', 'uniform'], help='Neighbor sampling strategy')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')

# Model parameters
parser.add_argument('--base_encoder', type=str, default=None,
                    choices=['mlp', 'tabm', 'dfm', 'resnet', 'fttrans', 'armnet'],
                    help='Base encoder type for entity table')
parser.add_argument('--channels', type=int, default=128,
                    help='Hidden dimension')
parser.add_argument('--out_channels', type=int,
                    default=1, help='Output dimension')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='Dropout probability')
parser.add_argument('--feat_layer_num', type=int, default=1,
                    help='Number of feature layers')
parser.add_argument('--feat_nhead', type=int, default=1,
                    help='Number of attention heads in feature layers')
parser.add_argument('--relation_layer_num', type=int,
                    default=2, help='Number of relation layers')
parser.add_argument('--relation_aggr', type=str,
                    default='sum', help='Relation aggregation method')

# Model ablation options
parser.add_argument('--deactivate_fusion_module', action='store_true',
                    default=False, help='Deactivate fusion module')
parser.add_argument('--deactivate_relation_module', action='store_true',
                    default=False, help='Deactivate relation layers')

# Training parameters
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--num_epochs', type=int, default=500,
                    help='Number of training epochs')
parser.add_argument('--early_stop_threshold', type=int,
                    default=10, help='Early stopping patience')
parser.add_argument('--max_round_epoch', type=int,
                    default=50, help='Max batches per epoch')
parser.add_argument('--no_need_test', action='store_false',
                    default=True, help='Skip test during training')

# Log settings to activate
parser.add_argument('--step_loss_path', type=str, default=None)
parser.add_argument('--val_metric_path', type=str, default=None)
parser.add_argument('--verbose', action='store_true',
                    default=False, help='Enable verbose logging')

args = parser.parse_args()

verbose = args.verbose
# Initialize logger
logger = ModernLogger(
    name="AIDA_Former_Modeling",
    level="info" if verbose else "critical",
    rich_tracebacks=False
)

# Device selection with auto-random feature
if args.device == "auto":
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        selected_gpu = random.randint(0, num_gpus - 2)  # Exclude the last GPU
        device = torch.device(f"cuda:{selected_gpu}")
        logger.info(
            f"Auto-selected GPU {selected_gpu} from {num_gpus} available GPUs")
    else:
        device = torch.device("cpu")
        logger.warning("No GPUs available, using CPU")
else:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        logger.warning(f"CUDA not available, falling back to CPU")

# Extract arguments for readability
cache_dir = args.tf_cache_dir
db_name = args.db_name
task_name = args.task_name

# Sampling settings
validation_ratio = args.validation_ratio
test_ratio = args.test_ratio
num_neighbors = args.num_neighbors
sample_strategy = args.sample_strategy
batch_size = args.batch_size

# Model parameters
channels = args.channels
out_channels = args.out_channels
dropout = args.dropout
feat_layer_num = args.feat_layer_num
feat_nhead = args.feat_nhead
relation_layer_num = args.relation_layer_num
relation_aggr = args.relation_aggr

# Training parameters
lr = args.lr
num_epochs = args.num_epochs
early_stop_threshold = args.early_stop_threshold
max_round_epoch = args.max_round_epoch
no_need_test = args.no_need_test

deactive_relation_layer = args.deactivate_relation_module
deactive_fusion_layer = args.deactivate_fusion_module

db = DatabaseFactory.get_db(
    db_name=db_name,
    with_text_compress=True,
)


dataset = DatabaseFactory.get_dataset(
    db_name=db_name,
)


task = DatabaseFactory.get_task(
    db_name=db_name,
    task_name=task_name,
    dataset=dataset,
)

col_type_dict = load_col_types(
    cache_path=cache_dir, file_name="col_type_dict.pkl")

data, col_stats_dict = build_pyg_hetero_graph(
    db,
    col_type_dict,
    get_text_embedder_cfg(device="cpu"),
    cache_dir=cache_dir,
    verbose=False,
)

# Display task information
# logger.section(f"Task: {task.task_type.value}")
# task_info = f"Database: {db_name}\n"
# task_info += f"Task: {task_name}\n"
# task_info += f"Device: {device}\n"
# task_info += f"Base Encoder: {args.base_encoder if args.base_encoder else 'Shared Transformer'}\n"
# task_info += f"Channels: {channels}, Out Channels: {out_channels}\n"
# task_info += f"Dropout: {dropout}\n"
# task_info += f"Batch Size: {batch_size}, Learning Rate: {lr}\n"
# task_info += f"Sampling Strategy: {sample_strategy}, Neighbors: {num_neighbors}\n"
# task_info += f"Epochs: {num_epochs}, Early Stop: {early_stop_threshold}"
# logger.info_panel("Configuration", task_info)

# logger.print("Deactivate status: ")
# logger.print(f"  Relation Layer: {deactive_relation_layer}")
# logger.print(f"  Fusion Layer: {deactive_fusion_layer}")

specific_table_encoder = None
if args.base_encoder:
    specific_table_encoder = {
        task.entity_table: args.base_encoder
    }

net = construct_default_AIDAXFormer(
    data,
    col_stats_dict,
    channels=channels,
    out_channels=out_channels,
    feat_layer_num=feat_layer_num,
    feat_nhead=feat_nhead,
    relation_layer_num=relation_layer_num,
    relation_aggr=relation_aggr,
    dropout_prob=dropout,
    deactivate_fusion_module=deactive_fusion_layer,
    deactivate_relation_module=deactive_relation_layer,
    specific_table_encoder=specific_table_encoder
)
net.reset_parameters()

# init data loader
data_loader_dict: Dict[str, NeighborLoader] = {}
for split, sample_ratio, table in [
    ("train", 1, task.get_table("train")),
    ("valid", validation_ratio, task.get_table("val")),
    ("test", test_ratio, task.get_table("test", mask_input_cols=False)),
]:

    _, table_input = get_node_train_table_input_with_sample(
        table=table,
        task=task,
        sample_ratio=sample_ratio,
        shuffle=False,
    )

    data_loader_dict[split] = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        time_attr="time",
        temporal_strategy=sample_strategy,
        input_nodes=table_input.nodes,
        input_time=table_input.time,
        transform=table_input.transform,
        batch_size=batch_size,
        shuffle=split == "train"
    )

# check task type
is_regression = task.task_type == TaskType.REGRESSION


def deactivate_dropout(net: torch.nn.Module):
    """ Deactivate dropout layers in the model. for regression task
    """
    deactive_nn_instances = (
        torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)
    for module in net.modules():
        if isinstance(module, deactive_nn_instances):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
    return net


net = deactivate_dropout(net) if is_regression else net
loss_fn = L1Loss() if is_regression else BCEWithLogitsLoss()
evaluate_metric_func = mean_absolute_error if is_regression else roc_auc_score
higher_is_better = False if is_regression else True


@torch.no_grad()
def test(net: torch.nn.Module, loader: torch.utils.data.DataLoader, entity_table: str, early_stop: int = -1, is_regression: bool = False):
    pred_list = []
    y_list = []
    early_stop = early_stop if early_stop > 0 else len(loader.dataset)

    if not is_regression:
        net.eval()

    if verbose:
        progress, task_id = logger.tmp_progress(
            total=len(loader), description=f"Testing")
        progress.start()

    for idx, batch in enumerate(loader):
        with torch.no_grad():
            batch = batch.to(device)
            y = batch[entity_table].y.float()
            pred = net(batch, entity_table)
            pred = pred.view(-1) if pred.size(1) == 1 else pred

            # apply a sigmoid
            if not is_regression:
                pred = torch.sigmoid(pred)

            pred_list.append(pred.detach().cpu())
            y_list.append(y.detach().cpu())

        if verbose:
            progress.update(task_id, advance=1)

        if idx > early_stop:
            break
    if verbose:
        progress.stop()

    pred_list = pred_logits = torch.cat(pred_list, dim=0)
    pred_list = torch.sigmoid(pred_list).numpy()
    y_list = torch.cat(y_list, dim=0).numpy()
    pred_list = pred_logits.numpy() if is_regression else pred_list
    return pred_list,  y_list


optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, net.parameters()), lr=lr
)


# init training variables
net.to(device)
patience = 0
best_epoch = 0
best_val_metric = -math.inf if higher_is_better else math.inf
best_model_state = None

start_time = time.time()

# loss-step
step_loss = []
val_metrics_log = []
for epoch in range(num_epochs):
    loss_accum = count_accum = 0
    net.train()

    if verbose:
        progress, task_id = logger.tmp_progress(
            total=min(len(data_loader_dict["train"]), max_round_epoch + 1),
            description=f"Epoch {epoch}"
        )

        progress.start()

    for idx, batch in enumerate(data_loader_dict["train"]):

        if idx > max_round_epoch:
            break

        optimizer.zero_grad()
        batch = batch.to(device)
        y = batch[task.entity_table].y.float()
        pred = net(batch, task.entity_table)
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        # record
        step_loss.append(loss.item())
        loss_accum += loss.item()
        count_accum += 1

        if verbose:
            progress.update(task_id, advance=1)

    if verbose:
        progress.stop()

    train_loss = loss_accum / count_accum
    val_logits, val_pred_hat = test(
        net, data_loader_dict["test"], task.entity_table, early_stop=-1, is_regression=is_regression
    )
    val_metric = evaluate_metric_func(val_pred_hat, val_logits)
    val_metrics_log.append(val_metric)

    logger.info(
        f"Epoch: {epoch} => Train Loss: {train_loss:.6f}, Val {evaluate_metric_func.__name__}: {val_metric:.6f} | Patience: {patience}/{early_stop_threshold}")

    # best_val_metric = 0
    if (higher_is_better and val_metric > best_val_metric) or \
       (not higher_is_better and val_metric < best_val_metric):
        best_val_metric = val_metric
        best_epoch = epoch
        best_model_state = copy.deepcopy(net.state_dict())
        patience = 0

        if no_need_test:
            test_logits, test_pred_hat = test(
                net, data_loader_dict["test"], task.entity_table, is_regression=is_regression)
            test_metric = evaluate_metric_func(test_pred_hat, test_logits)

            logger.info(
                f"Updated best scores => Test {evaluate_metric_func.__name__}: {test_metric:.6f}")
        else:
            logger.info(
                f"Updated best scores"
            )
    else:
        patience += 1
        if patience > early_stop_threshold:
            logger.warning(f"Early stopping at epoch {epoch}")
            break

end_time = time.time()
training_duration = end_time - start_time

logger.success(
    f"Training completed in {training_duration:.2f}s | Best Val {evaluate_metric_func.__name__}: {best_val_metric:.6f} at epoch {best_epoch}"
)

# save the logs

if args.step_loss_path is not None:
    np.save(args.step_loss_path, np.array(step_loss))

if args.val_metric_path is not None:
    np.save(args.val_metric_path, np.array(val_metrics_log))

# print the best results
net.load_state_dict(best_model_state)


table = task.get_table("test", mask_input_cols=False)
_, table_input = get_node_train_table_input_with_sample(
    table=table,
    task=task,
    sample_ratio=1,
    shuffle=False,
)

loader = NeighborLoader(
    data,
    num_neighbors=num_neighbors,
    time_attr="time",
    temporal_strategy=sample_strategy,
    input_nodes=table_input.nodes,
    input_time=table_input.time,
    transform=table_input.transform,
    batch_size=batch_size,
    shuffle=False,
)

start_time = time.time()
test_logits, test_pred_hat = test(
    net, loader, task.entity_table, is_regression=is_regression)
test_metric = evaluate_metric_func(test_pred_hat, test_logits)
end_time = time.time()
inference_time = end_time - start_time


logger.success(
    f"Final Test {evaluate_metric_func.__name__}: {test_metric:.6f} | Inference Time: {inference_time:.2f}s")
