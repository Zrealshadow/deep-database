import torch
import math
import argparse
import copy
from tqdm import tqdm
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import MLP, Node2Vec
from torch_geometric.utils import sort_edge_index
from torch.nn import L1Loss, BCEWithLogitsLoss
from sklearn.metrics import mean_absolute_error, roc_auc_score
from relbench.base import TaskType
from typing import Dict

from utils.data import DatabaseFactory
from utils.builder import HomoGraph, make_homograph_from_db

device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser()

# Dataset and cache
parser.add_argument('--tf_cache_dir', type=str, required=True)
parser.add_argument('--data_cache_dir', type=str, default=None)
parser.add_argument('--db_name', type=str, required=True)
parser.add_argument('--task_name', type=str, required=True)

# Model parameters
parser.add_argument('--channels', type=int, default=128)
parser.add_argument('--out_channels', type=int, default=1)


# Training parameters
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--n2v_lr', type=float, default=0.01)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--num_epochs', type=int, default=300)
parser.add_argument('--early_stop_threshold', type=int, default=5)
parser.add_argument('--max_round_epoch', type=int, default=100)

args = parser.parse_args()

cache_dir = args.tf_cache_dir
data_cache_dir = args.data_cache_dir
db_name = args.db_name
task_name = args.task_name
print(f"db_name: {db_name}, task_name: {task_name}")

# model parameters
channels = args.channels
out_channels = args.out_channels

# training parameters
batch_size = args.batch_size
n2v_lr = args.n2v_lr
lr = args.lr
num_epochs = args.num_epochs
early_stop_threshold = args.early_stop_threshold
max_round_epoch = args.max_round_epoch


# db = DatabaseFactory.get_db(
#     db_name, cache_dir=data_cache_dir, with_text_compress=True)
# task = DatabaseFactory.get_task(db_name, task_name)

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

homoGraph = make_homograph_from_db(db, verbose=True)

# build edge
x = torch.LongTensor(homoGraph.row)
y = torch.LongTensor(homoGraph.col)
edge_index = sort_edge_index(torch.stack([x, y], dim=0))


# build X, Y
x_dict = {}
y_dict = {}
for split, table in [
    ("train", task.get_table("train")),
    ("val", task.get_table("val")),
    ("test", task.get_table("test", mask_input_cols=False)),
]:
    x_col = list(table.fkey_col_to_pkey_table.keys())[0]
    x = table.df[x_col].tolist()
    x = homoGraph.dbindex.get_global_ids(task.entity_table, x)
    y = table.df[task.target_col].tolist()
    x_dict[split] = x
    y_dict[split] = y


# declare model

class Node2VecWrapper(torch.nn.Module):
    def __init__(self, model: Node2Vec, out_channels: int):
        super().__init__()
        self.model = model
        self.head = MLP(channel_list=[
                        model.embedding_dim, model.embedding_dim, out_channels], dropout=0.2)

    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        return x


model = Node2Vec(
    edge_index,
    embedding_dim=channels,
    walk_length=10,
    context_size=5,
    walks_per_node=10,
    num_negative_samples=1,
    sparse=True,
)

net = Node2VecWrapper(model, out_channels=out_channels).to(device)


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


optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=n2v_lr)
net_optim = torch.optim.Adam(net.head.parameters(), lr=lr)

net.train()
best_val_metric = -math.inf if higher_is_better else math.inf
best_state_state = None
patience = 0
best_epoch = 0

loader = model.loader(batch_size = batch_size, shuffle = True, num_workers = 4)

for epoch in range(1, num_epochs+1):
    total_loss = 0
    cnt = 0
    for pos_rw, neg_rw in tqdm(loader, leave=False):
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        cnt += 1
        if cnt > max_round_epoch*100:
            break
    
    
    classifier_loss = 0
    for cnt, i in enumerate(range(0, len(x_dict['train']), batch_size)):
        if cnt > max_round_epoch:
            break
        net_optim.zero_grad()
        x_batch = x_dict['train'][i:i+batch_size]
        y_batch = y_dict['train'][i:i+batch_size]

        x_batch = torch.LongTensor(x_batch).to(device)
        y_batch = torch.FloatTensor(y_batch).to(device)

        pred = net(x_batch)
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        loss = loss_fn(pred, y_batch)
        loss.backward()
        net_optim.step()
        classifier_loss += loss.item()
    
    # valid ones
    pred_y = []
    for i in range(0, len(x_dict['val']), batch_size):
        x_batch = x_dict['val'][i:i+batch_size]
        x_batch = torch.LongTensor(x_batch).to(device)

        pred = net(x_batch)
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred_y.append(pred.cpu().detach())
    pred_y = torch.cat(pred_y, dim=0).numpy()
    val_metrics = evaluate_metric_func(y_dict['val'], pred_y)
    print(f"Epoch: {epoch}, Node2Vec Loss: {total_loss:.4f}, Classifer Loss: {classifier_loss:.4f} Val Metric: {val_metrics:.6f}")
    
    
    if (higher_is_better and val_metrics > best_val_metric) or \
            (not higher_is_better and val_metrics < best_val_metric):
        best_val_metric = val_metrics
        best_epoch = epoch
        best_state_state = copy.deepcopy(net.state_dict())
        patience = 0
    else:
        patience += 1
        if patience >= early_stop_threshold:
            print(f"Early stopping at epoch {epoch}")
            break
    

# test
net.load_state_dict(best_state_state)
pred_y = []
for i in range(0, len(x_dict['test']), batch_size):
    x_batch = x_dict['test'][i:i+batch_size]
    x_batch = torch.LongTensor(x_batch).to(device)

    pred = net(x_batch)
    pred = pred.view(-1) if pred.size(1) == 1 else pred
    pred_y.append(pred.cpu().detach())
pred_y = torch.cat(pred_y, dim=0).numpy()
metrics = evaluate_metric_func(y_dict['test'], pred_y)
print(f"Test Metric: {metrics:.6f}")