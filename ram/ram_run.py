import torch
import math
import argparse
import copy
from tqdm import tqdm
import numpy as np
import time
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import sort_edge_index
from torch.nn import L1Loss, BCEWithLogitsLoss
from sklearn.metrics import mean_absolute_error, roc_auc_score
from relbench.base import TaskType
from typing import Dict


from utils.util import load_col_types
from utils.resource import get_text_embedder_cfg
from utils.builder import build_pyg_hetero_graph
from utils.data import DatabaseFactory
from utils.sample import get_node_train_table_input_with_sample
from utils.util import load_np_dict

from model.rdb import RDBModel


device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()

# Dataset and cache
parser.add_argument('--tf_cache_dir', type=str, required=True)
parser.add_argument('--data_cache_dir', type=str, default=None)
parser.add_argument('--db_name', type=str, required=True)
parser.add_argument('--task_name', type=str, required=True)
parser.add_argument('--edge_path', type=str, default=None)

# Function settings
parser.add_argument('--validation_ratio', type=float, default=1)
parser.add_argument('--test_ratio', type=float, default=1)
parser.add_argument('--num_neighbors', nargs='+', type=int, default=[64, 32])
parser.add_argument('--batch_size', type=int, default=256)

# Model parameters
parser.add_argument('--channels', type=int, default=128)
parser.add_argument('--out_channels', type=int, default=1)
parser.add_argument('--aggr', type=str, default="sum")
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--feat_layer_num', type=int, default=1)
parser.add_argument('--graph_layer_num', type=int, default=2)
parser.add_argument('--feat_norm', type=str, default="layer_norm")
parser.add_argument('--head_norm', type=str, default="batch_norm")
parser.add_argument('--heads', type=int, default=4)
parser.add_argument('--pretrain_path', type=str, default=None)

# Training parameters
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--early_stop_threshold', type=int, default=10)
parser.add_argument('--max_round_epoch', type=int, default=10)
parser.add_argument('--no_need_test', action='store_false', default=True)


# Log settings to activate
parser.add_argument('--step_loss_path', type=str, default=None)
parser.add_argument('--val_metric_path', type=str, default=None)

args = parser.parse_args()


cache_dir = args.tf_cache_dir
data_cache_dir = args.data_cache_dir
db_name = args.db_name
task_name = args.task_name
edge_path = args.edge_path

# define some functions
validation_ratio = args.validation_ratio
test_ratio = args.test_ratio
num_neighbors = args.num_neighbors
batch_size = args.batch_size

# model parameters
channels = args.channels
out_channels = args.out_channels
aggr = args.aggr
dropout = args.dropout
feat_layer_num = args.feat_layer_num
graph_layer_num = args.graph_layer_num
feat_norm = args.feat_norm
head_norm = args.head_norm
pretrain_path = args.pretrain_path

# training parameters
lr = args.lr
num_epochs = args.num_epochs
early_stop_threshold = args.early_stop_threshold
max_round_epoch = args.max_round_epoch
no_need_test = args.no_need_test


db = DatabaseFactory.get_db(
    db_name=db_name,
    cache_dir=data_cache_dir,
)


dataset = DatabaseFactory.get_dataset(
    db_name=db_name,
    cache_dir=data_cache_dir,
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
    verbose=True,
)


# if additional edges are provided, load them
if edge_path is not None:
    edge_dict = load_np_dict(edge_path)
    for edge_name, edge_np in edge_dict.items():
        src_table, dst_table = edge_name.split('-')[0], edge_name.split('-')[1]
        edge_index = torch.from_numpy(edge_np.astype(int)).t()
        # [2, edge_num]
        edge_type = (src_table, f"appendix", dst_table)
        data[edge_type].edge_index = sort_edge_index(edge_index)
    data.validate()


net = RDBModel(
    data,
    col_stats_dict,
    channels=channels,
    out_channels=out_channels,
    feat_layer_num=feat_layer_num,
    feat_norm=feat_norm,
    head_norm=head_norm,
    aggr=aggr,
    graph_layer_num=graph_layer_num,
    dropout_prob=dropout
)

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
        input_nodes=table_input.nodes,
        input_time=table_input.time,
        transform=table_input.transform,
        batch_size=batch_size,
        shuffle=split == "train"
    )

# load pre-trained model
if pretrain_path is not None:
    pre_trained_state_dict = torch.load(pretrain_path)
    net.load_state_dict(pre_trained_state_dict)
    print(f"Load pre-trained model from {pretrain_path}")

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

    for idx, batch in tqdm(enumerate(loader), total=len(loader), leave=False, desc="Testing"):
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
        if idx > early_stop:
            break

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


# loss-step
step_loss = []
val_metrics_log = []
for epoch in range(num_epochs):
    loss_accum = count_accum = 0
    net.train()
    for idx, batch in tqdm(enumerate(data_loader_dict["train"]),
                           leave=False,
                           total=len(data_loader_dict["train"]),
                           desc="Training"):
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
    train_loss = loss_accum / count_accum
    val_logits, val_pred_hat = test(
        net, data_loader_dict["valid"], task.entity_table, early_stop=-1, is_regression=is_regression
    )
    val_metric = evaluate_metric_func(val_pred_hat, val_logits)
    val_metrics_log.append(val_metric)
    print(
        f"==> Epcoh: {epoch} => Train Loss: {train_loss:.6f}, Val {evaluate_metric_func.__name__} Metric: {val_metric:.6f} \t{patience}/{early_stop_threshold}")
    
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

            print(
                f"Update the best scores => Test {evaluate_metric_func.__name__} Metric: {test_metric:.6f}")
        else:
            print(
                f"Update the best scores \t "
            )
    else:
        patience += 1
        if patience > early_stop_threshold:
            print(f"Early stopping at epoch {epoch}")
            break


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


print(
    f"Test {evaluate_metric_func.__name__} Metric: {test_metric:.6f}, Inference Time: {inference_time:.2f} seconds")
