from relbench.base import Table
import torch
import math
import os
import argparse
import numpy as np
import copy
from tqdm import tqdm
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import MLP, Node2Vec
from torch_geometric.utils import sort_edge_index

from torch_frame import stype
from torch.nn import L1Loss, BCEWithLogitsLoss
from sklearn.metrics import mean_absolute_error, roc_auc_score
from relbench.base import TaskType
from typing import Dict
from utils.preprocess import infer_type_in_table
from utils.util import remove_pkey_fkey

from utils.data import DatabaseFactory, TableData
from utils.builder import HomoGraph, make_homograph_from_db

from typing import List

from utils.resource import get_text_embedder_cfg


device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()

# Dataset and cache
parser.add_argument('--db_name', type=str, required=True)
parser.add_argument('--task_name', type=str, required=True)
parser.add_argument('--table_output_dir', type=str, default=None)
parser.add_argument('--sample_size', type=int, default=100_000)

# Model parameters
parser.add_argument('--channels', type=int, default=64)
parser.add_argument('--out_channels', type=int, default=1)

# Training parameters
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--n2v_lr', type=float, default=0.01)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--early_stop_threshold', type=int, default=5)
parser.add_argument('--max_round_epoch', type=int, default=10_000)
parser.add_argument('--verbose', action='store_true')


args = parser.parse_args()


db_name = args.db_name
task_name = args.task_name
sample_size = args.sample_size

# model parameters
channels = args.channels
out_channels = args.out_channels

# training parameters
batch_size = args.batch_size
n2v_lr = args.n2v_lr
num_epochs = args.num_epochs
max_round_epoch = args.max_round_epoch

table_output_dir = args.table_output_dir
verbose = args.verbose

db = DatabaseFactory.get_db(
    db_name=db_name,
)

dataset = DatabaseFactory.get_dataset(
    db_name=db_name,
)

task = DatabaseFactory.get_task(db_name, task_name, dataset=dataset)


homoGraph = make_homograph_from_db(db, verbose=verbose)

x = torch.LongTensor(homoGraph.row)
y = torch.LongTensor(homoGraph.col)
edge_index = sort_edge_index(torch.stack([x, y], dim=0))

entity_table = db.table_dict[task.entity_table]
entity_df = entity_table.df

x_dict = {}
y_dict = {}
dfs = {}

for split, table in [
    ("train", task.get_table("train")),
    ("val", task.get_table("val")),
    ("test", task.get_table("test", mask_input_cols=False)),
]:
    if split == "train" and args.sample_size is not None:
        sample_idx = np.random.permutation(len(table.df))[:sample_size]
        table.df = table.df.iloc[sample_idx].reset_index(drop=True)

    x_col = list(table.fkey_col_to_pkey_table.keys())[0]
    x = table.df[x_col].tolist()
    x = homoGraph.dbindex.get_global_ids(task.entity_table, x)
    y = table.df[task.target_col].tolist()
    x_dict[split] = x
    y_dict[split] = y

    # join with entity_table
    left_entity = task.entity_col

    entity_df = entity_df.astype(
        {entity_table.pkey_col: table.df[left_entity].dtype})

    dfs[split] = table.df.merge(
        entity_df,
        how="left",
        left_on=left_entity,
        right_on=entity_table.pkey_col,
        suffixes=('', '_extend')
    )

    # "user" and "user_id"
    # the join column is not same, there will be duplicated columns
    if left_entity != entity_table.pkey_col:
        dfs[split].drop(columns=[entity_table.pkey_col], inplace=True)


model = Node2Vec(
    edge_index,
    embedding_dim=channels,
    walk_length=10,
    context_size=5,
    walks_per_node=10,
    num_negative_samples=1,
    sparse=True,
)

optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=n2v_lr)
loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=4)


model = model.to(device)

# Start timing Node2Vec training
import time
n2v_train_start = time.time()

for epoch in range(1, num_epochs + 1):
    total_loss = 0
    cnt = 0

    for pos_rw, neg_rw in tqdm(loader, leave=False, desc=f"Epoch {epoch}"):
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        cnt += 1

        if cnt > max_round_epoch:
            break

n2v_train_time = time.time() - n2v_train_start
print(f"Node2Vec training completed in {n2v_train_time:.2f}s ({n2v_train_time/60:.2f} min)")

# then we get a trained node2vec model.
# we concatenate the vector from node2vec as feature
# and make it as an table dataset.
# follow Leva's way to boost machine learning model.

with torch.no_grad():
    node_embeddings = model.embedding.weight.cpu().numpy()

train_batch_embeddings = []


def get_embeddings(node_ids: List, batch_size: int):
    embs = []
    for i in range(0, len(node_ids), batch_size):
        x_batch = node_ids[i: i + batch_size]
        emb_batch = node_embeddings[x_batch]
        embs.append(emb_batch)
    return np.vstack(embs)


train_embs = get_embeddings(x_dict['train'], batch_size)
val_embs = get_embeddings(x_dict['val'], batch_size)
test_embs = get_embeddings(x_dict['test'], batch_size)

# concatenate the embeddings with original features

# configure the column types

# ------------------ Construct Table for type inference

train_table = task.get_table("train")

fkey_col_to_pkey_table = copy.deepcopy(train_table.fkey_col_to_pkey_table)
fkey_col_to_pkey_table.update(entity_table.fkey_col_to_pkey_table)


# there pkey_col can not be considered as pkey
# there pkey should be (pkey, timestamp)
# But there, we just for type infer, so ignore it.

object_table = Table(
    df=dfs["train"],
    fkey_col_to_pkey_table=fkey_col_to_pkey_table,
    pkey_col=train_table.pkey_col,
    time_col=train_table.time_col,
)


# ------------------- configure the column types
table_col_types = infer_type_in_table(
    object_table,
    verbose=verbose
)

remove_pkey_fkey(table_col_types, object_table)

# then add the embedding column types

dfs["train"]['node2vec_emb'] = list(train_embs)
dfs["val"]['node2vec_emb'] = list(val_embs)
dfs["test"]['node2vec_emb'] = list(test_embs)


table_col_types['node2vec_emb'] = stype.embedding


# construct tabular data
data = TableData(
    train_df=dfs["train"],
    val_df=dfs["val"],
    test_df=dfs["test"],
    col_to_stype=table_col_types,
    target_col=task.target_col,
    task_type=task.task_type,
)

dirname = db_name + "-" + task_name

if table_output_dir:

    path = os.path.join(table_output_dir, dirname)
    text_embedder_cfg = get_text_embedder_cfg()
    data.materilize(
        col_to_text_embedder_cfg=text_embedder_cfg,
    )

    data.save_to_dir(
        path
    )
    print(f"Leva table data saved to {path}")
