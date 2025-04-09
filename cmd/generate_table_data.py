import os
import argparse
import pandas as pd
import numpy as np
import torch_frame
from relbench.base import TaskType
from torch_frame import stype

from typing import Dict, Any

import sys
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.preprocess import infer_type_in_table
from utils.util import remove_pkey_fkey
from utils.data import DatabaseFactory, TableData




parser = argparse.ArgumentParser(description="Process user attendance task.")
    
parser.add_argument("--dbname", type=str, default="event", help="Name of the database.")
parser.add_argument("--task_name", type=str, default="user-attendance", help="Name of the task.")
parser.add_argument("--db_cache_dir", type=str, default=None, help="Path to DB cache directory.")
parser.add_argument("--sample_size", type=int, default=-1, help="Sample size for processing. -1 means all.")
parser.add_argument("--table_output_dir", type=str, required=True, help="Directory to output tables.")
args = parser.parse_args()


dbname = args.dbname
task_name = args.task_name
db_cache_dir = args.db_cache_dir
sample_size = args.sample_size
table_output_dir = args.table_output_dir

db = DatabaseFactory.get_db(
    db_name=dbname,
    cache_dir=db_cache_dir,
)

task = DatabaseFactory.get_task(
    db_name=dbname,
    task_name=task_name,
)

entity_table = db.table_dict[task.entity_table]
entity_df = entity_table.df

table_col_types = infer_type_in_table(
    entity_table,
    task.entity_table,
    verbose=True
)

remove_pkey_fkey(
    table_col_types,
    entity_table,
)

train_table = task.get_table("train")
val_table = task.get_table("val")
test_table = task.get_table("test", mask_input_cols=False)


if task.task_type == TaskType.BINARY_CLASSIFICATION:
    table_col_types[task.target_col] = stype.categorical
elif task.task_type == TaskType.REGRESSION:
    table_col_types[task.target_col] = stype.numerical
elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
    table_col_types[task.target_col] = stype.embedding
else:
    raise ValueError(f"Unsupported task type called {task.task_type}")


# sample_training data
if sample_size > 0:
    sampled_idx = np.random.permutation(len(train_table.df))[:sample_size]
    train_table.df = train_table.df.iloc[sampled_idx]


dfs: Dict[str, pd.DataFrame] = {}

for split, table in [
    ("train", train_table),
    ("val", val_table),
    ("test", test_table),
]:
    left_entity = list(table.fkey_col_to_pkey_table.keys())[0]
    entity_df = entity_df.astype(
        {entity_table.pkey_col: table.df[left_entity].dtype})
    dfs[split] = table.df.merge(
        entity_df,
        how="left",
        left_on=left_entity,
        right_on=entity_table.pkey_col,
    )

# assign the time column in col_types_dict
if task.time_col:
    table_col_types[task.time_col] = stype.timestamp


data = TableData(
    train_df=dfs["train"],
    val_df=dfs["val"],
    test_df=dfs["test"],
    col_to_stype=table_col_types,
    target_col = task.target_col,
)


dirname = dbname + "-" + task_name
path = os.path.join(table_output_dir, dirname)
data.save_to_dir(
    path
)

