import torch
import os
import math
import argparse
import random
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import pandas as pd
from relbench.base import Database
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.loader import NeighborLoader
from torch_geometric.typing import NodeType
from typing import Dict, List


from utils.util import load_col_types, to_unix_time
from utils.resource import get_text_embedder_cfg
from utils.builder import build_pyg_hetero_graph
from utils.data import DatabaseFactory
from model import HeteroGCN
from model.dgi import HeteroDeepGraphInfomax
from model.graphcl import HeteroGraphCL

device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()

# Dataset and cache
parser.add_argument('--tf_cache_dir', type=str, required=True)
parser.add_argument('--data_cache_dir', type=str, default=None)
parser.add_argument('--db_name', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True,
                    help="Path to save the pretrained model")

# Function settings
parser.add_argument('--num_neighbors', nargs='+', type=int, default=[128, 64])
parser.add_argument('--batch_size', type=int, default=32)

# Model parameters
parser.add_argument('--channels', type=int, default=128)
parser.add_argument('--out_channels', type=int, default=1)
parser.add_argument('--norm', type=str, default="layer_norm")
parser.add_argument('--aggr', type=str, default="sum")
parser.add_argument('--edge_aggr', type=str, default="sum")
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--heads', type=int, default=4)
parser.add_argument('--model', type=str, default="GCN")
parser.add_argument('--method', type=str, default="dgi")

# Training parameters
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--num_epochs', type=int, default=300)
parser.add_argument('--early_stop_threshold', type=int, default=3)
parser.add_argument('--max_round_epoch', type=int, default=50)

args = parser.parse_args()


cache_dir = args.tf_cache_dir
data_cache_dir = args.data_cache_dir
db_name = args.db_name
output_dir = args.output_dir

# define some functions
num_neighbors = args.num_neighbors
batch_size = args.batch_size

# model parameters
channels = args.channels
out_channels = args.out_channels
norm = args.norm
aggr = args.aggr
edge_aggr = args.edge_aggr
dropout = args.dropout
num_layers = args.num_layers
heads = args.heads
model = args.model
method = args.method

# training parameters
lr = args.lr
num_epochs = args.num_epochs
early_stop_threshold = args.early_stop_threshold
max_round_epoch = args.max_round_epoch

db = DatabaseFactory.get_db(
    db_name, cache_dir=data_cache_dir, with_text_compress=True
)

col_type_dict = load_col_types(
    cache_path=cache_dir, file_name="col_type_dict.pkl"
)


data, col_stats_dict = build_pyg_hetero_graph(
    db,
    col_type_dict,
    get_text_embedder_cfg(device="cpu"),
    cache_dir=cache_dir,
    verbose=True,
)


def neighborsample_loader(
    data: HeteroData,
    db: Database,
    entity_table: NodeType,
    num_neighbors: List[int] = [64, 64],
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


if model != "GCN":
    raise NotImplementedError(
        f"Model {model} is not implemented for graph pre-trained, plz try GCN")

# declare model
encoder = HeteroGCN(
    data,
    col_stats_dict,
    channels=channels,
    out_channels=out_channels,
    num_layers=num_layers,
    aggr=aggr,
    edge_aggr=edge_aggr,
    dropout=dropout,
)

if method == "dgi":
    wrapper = HeteroDeepGraphInfomax(
        data=data,
        channel=channels,
        encoder=encoder
    )
elif method == "graphcl":
    wrapper = HeteroGraphCL(
        data=data,
        channel=channels,
        encoder=encoder
    )
else:
    raise KeyError("method should be in [dgi, graphcl]")

# start train


# initialize the loader
entity_tables = data.node_types
loader_dict = {}
for entity in entity_tables:
    loader = neighborsample_loader(data, db, entity, num_neighbors)
    loader_dict[entity] = loader

optimizer = torch.optim.Adam(wrapper.parameters(), lr=lr)

best_encoder_state_dict = None
best_loss = math.inf
best_epoch = 0
patience = 0


wrapper.to(device)

for epoch in range(1, num_epochs + 1):
    total_loss = 0
    cnt_accum = 0
    random.shuffle(entity_tables)
    for entity_table in entity_tables:
        cnt = 0
        loader = loader_dict[entity_table]

        for batch in tqdm(loader,
                          leave=False,
                          desc=f"Epoch {epoch}:{entity_table}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            x_dict_pos, x_dict_neg, summary = wrapper(
                batch,
                entity_table,
            )
            loss = wrapper.loss(x_dict_pos, x_dict_neg, summary)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item()
            cnt_accum += 1
            cnt += 1
            if cnt > max_round_epoch:
                break

    ave_loss = total_loss / cnt_accum

    if ave_loss < best_loss:
        best_loss = ave_loss
        best_epoch = epoch
        best_encoder_state_dict = deepcopy(encoder.state_dict())
        patience = 0
    else:
        patience += 1

    print(
        f"Epoch {epoch}: loss {ave_loss:.8f}, patience {patience}/{early_stop_threshold}")

    if patience >= early_stop_threshold:
        print("Early stopping")
        break


# save the best model

output_path = os.path.join(
    output_dir, method, f"{db_name}-{method}-encoder.pt")

with open(output_path, "wb") as f:
    torch.save(best_encoder_state_dict, f)
