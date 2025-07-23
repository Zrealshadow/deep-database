import random
import time
import torch
import math
import argparse
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import sort_edge_index
from relbench.base import Database
from typing import List

from utils.util import load_col_types, to_unix_time
from utils.resource import get_text_embedder_cfg
from utils.builder import build_pyg_hetero_graph
from utils.data import DatabaseFactory
from utils.util import load_np_dict

from model.augment import HeteroGraphPermutationTool
from model.rdb import RDBModel
from model.utils import InfoNCE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()

# Dataset and cache
parser.add_argument('--tf_cache_dir', type=str, required=True)
parser.add_argument('--data_cache_dir', type=str, default=None)
parser.add_argument('--db_name', type=str, required=True)
parser.add_argument('--sample_path', type=str, required=True)
parser.add_argument('--edge_path', type=str, default=None)
parser.add_argument('--output_path', type=str, default=None)

# Function settings
parser.add_argument('--num_neighbors', nargs='+', type=int, default=[128, 64])
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
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--early_stop_threshold', type=int, default=3)
parser.add_argument('--max_round_epoch', type=int, default=10)

parser.add_argument('--temperature', type=float, default=0.01)
parser.add_argument('--negative_num', type=int, default=20)
parser.add_argument('--permutation_ratio', type=float, default=0.2)

args = parser.parse_args()

cache_dir = args.tf_cache_dir
data_cache_dir = args.data_cache_dir
db_name = args.db_name
edge_path = args.edge_path
sample_path = args.sample_path

# define some functions
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

db = DatabaseFactory.get_db(
    db_name=db_name,
    cache_dir=data_cache_dir,
)

col_type_dict = load_col_types(
    cache_path=cache_dir, file_name="col_type_dict.pkl")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data, col_stats_dict = build_pyg_hetero_graph(
    db,
    col_type_dict,
    get_text_embedder_cfg(device="cpu"),
    cache_dir=cache_dir,
    verbose=True,
)

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


sample_dict = load_np_dict(sample_path)
entity_tables = list(sample_dict.keys())


temperature = args.temperature
negative_num = args.negative_num
permutation_ratio = args.permutation_ratio


# expand sample_dict to index
positive_pair_index_dict = {}
# entity_table -> np.ndarray (positive pair matrix)

for table_name in entity_tables:
    sample_np = sample_dict[table_name]
    sample_idx = sample_np[:, 0]
    sample_pos_pair = sample_np
    n = db.table_dict[table_name].df.shape[0]
    b = sample_np.shape[1]
    arr = np.full((n, b), -1)
    arr[:, 0] = np.arange(n)
    arr[sample_idx, :] = sample_pos_pair
    positive_pair_index_dict[table_name] = arr


def neighborsample_loader(
    data: HeteroData,
    db: Database,
    entity_table: str,
    batch_size: int = 256,
    num_neighbors: List[int] = [64, 64]
):

    n = len(db.table_dict[entity_table].df)
    node_idxs = np.arange(n)
    nodes = (entity_table, torch.from_numpy(node_idxs))

    input_time = torch.from_numpy(
        to_unix_time(pd.Series([db.max_timestamp] * n)))

    if db.table_dict[entity_table].time_col:
        time_col = db.table_dict[entity_table].time_col
        time_values = db.table_dict[entity_table].df[time_col].loc[node_idxs.tolist(
        )]
        input_time = torch.from_numpy(to_unix_time(time_values))

    loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes=nodes,
        time_attr="time",
        input_time=input_time,
        batch_size=batch_size,
        temporal_strategy="uniform",
        disjoint=True,
        shuffle=True,
        num_workers=0,
    )
    return loader


# init loader dict
loader_dict = {}
for table_name in entity_tables:
    loader_dict[table_name] = neighborsample_loader(
        data,
        db,
        entity_table=table_name,
        batch_size=batch_size,
        num_neighbors=num_neighbors,
    )


def neighborsample_batch(
    data: HeteroData,
    db: Database,
    entity_table: str,
    node_idxs: np.ndarray,
    num_neighbors: List[int] = [64, 64],
):
    # node_idxs: [n]
    nodes = (entity_table, torch.from_numpy(node_idxs))
    n = node_idxs.shape[0]
    input_time = torch.from_numpy(
        to_unix_time(pd.Series([db.max_timestamp] * n)))

    if db.table_dict[entity_table].time_col:
        time_col = db.table_dict[entity_table].time_col
        time_values = db.table_dict[entity_table].df[time_col].loc[node_idxs.tolist(
        )]
        input_time = torch.from_numpy(to_unix_time(time_values))

    loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes=nodes,
        time_attr="time",
        input_time=input_time,
        batch_size=n,
        temporal_strategy="uniform",
        shuffle=False,
        disjoint=True,
        num_workers=0,
        persistent_workers=False,
    )
    return next(iter(loader))


# train
best_loss = math.inf
best_state = None
patience = 0
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
loss_fn = InfoNCE(temperature=temperature,
                  negative_mode='paired', reduction='mean')
net.to(device)
net.train()


for epoch in range(1, num_epochs + 1):
    ave_loss = 0
    random.shuffle(entity_tables)

    for sample_table in entity_tables:
        positive_pair_index = positive_pair_index_dict[sample_table]
        loader = loader_dict[sample_table]

        loss_accum = count_accum = 0
        now = time.time()

        cnt = 0
        for batch in tqdm(loader,
                          leave=False,
                          desc=f"Epoch {epoch}:{sample_table}"):
            cnt += 1
            if cnt > max_round_epoch:
                break

            optimizer.zero_grad()

            # get the positive pair batch
            input_node_idx = batch[sample_table].input_id
            B = input_node_idx.shape[0]
            batch_positive_pair_index = positive_pair_index[input_node_idx]
            batch_pos_samples_nodes = np.array(
                [np.random.choice(row[row != -1]) for row in batch_positive_pair_index])

            pos_batch = neighborsample_batch(
                data, db, sample_table, batch_pos_samples_nodes)

            batch, pos_batch = batch.to(device), pos_batch.to(device)

            permute_pos_batch = HeteroGraphPermutationTool.random_permutation(
                pos_batch,
                drop_prob=permutation_ratio,
                exclude_node_types=[sample_table]
            )

            anchor_x = net.get_node_embedding(batch, sample_table)[
                sample_table][:B]
            permute_x_dict = net.get_node_embedding(
                permute_pos_batch, sample_table)
            pos_x = permute_x_dict[sample_table][:B]
            # [B, channels]

            # construct the negative samples
            neg_indices = torch.randint(
                0, len(permute_x_dict[sample_table]), (B, negative_num))
            neg_x = permute_x_dict[sample_table][neg_indices]
            # [B, negative_num, channels]

            loss = loss_fn(anchor_x, pos_x, neg_x)
            loss.backward()
            optimizer.step()

            loss_accum += loss.detach().item() * B
            count_accum += B

        end = time.time()
        train_loss = loss_accum / count_accum
        ave_loss += train_loss
        mins, secs = divmod(end - now, 60)
        print(
            f"====> {epoch}  In {sample_table}, Train loss: {train_loss} Count accum :{count_accum}, Cost Time {mins:.0f}m {secs:.0f}s ")

    ave_loss /= len(entity_tables)
    if ave_loss < best_loss:
        best_loss = ave_loss
        best_state = copy.deepcopy(net.state_dict())
        print(f"Best loss: {best_loss}")
        patience = 0
    else:
        patience += 1
        if patience >= early_stop_threshold:
            print(f"Early stopping at epoch {epoch}")
            break
        print(
            f"Patience: {patience}/{early_stop_threshold}, Best loss: {best_loss}")


# save to output path
if args.output_path is not None:
    print(f"Save to {args.output_path}")
    torch.save(best_state, args.output_path)
