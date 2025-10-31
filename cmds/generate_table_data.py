# -*- coding: utf-8 -*-
from utils.logger import ModernLogger
from utils.resource import get_text_embedder_cfg
from utils.preprocess import infer_type_in_table
from utils.util import remove_pkey_fkey
from utils.data import DatabaseFactory, TableData

import os
import argparse
import pandas as pd
import numpy as np
import torch_frame
from relbench.base import TaskType, Table
from torch_frame import stype
import featuretools as ft
from featuretools.selection import (
    remove_highly_correlated_features,
    remove_highly_null_features,
    remove_single_value_features,
)

import time

from typing import Dict, Any

import sys
from pathlib import Path

# set np random seed
np.random.seed(2025)

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))


parser = argparse.ArgumentParser(description="Process user attendance task.")

parser.add_argument("--dbname", type=str, default="event",
                    help="Name of the database.")
parser.add_argument("--task_name", type=str,
                    default="user-attendance", help="Name of the task.")
parser.add_argument("--db_cache_dir", type=str, default=None,
                    help="Path to DB cache directory.")
parser.add_argument("--sample_size", type=int, default=-1,
                    help="Sample size for processing. -1 means all.")
parser.add_argument("--table_output_dir", type=str,
                    default=None, help="Directory to output tables.")

# activate dfs
parser.add_argument("--dfs", action='store_true', default=False)
parser.add_argument("--selection", action='store_true', default=False,
                    help="Activate feature selection after dfs.")
parser.add_argument("--max_depth", type=int, default=2,
                    help="Max depth for deep feature synthesis.")
parser.add_argument("--max_features", type=int, default=1000,
                    help="Max number of features to generate.")
parser.add_argument("--n_timedelta", type=int, default=-1,
                    help="Number of days for time delta window.")
parser.add_argument("--time_budget", type=int, default=-1,
                    help="Time budget in minutes for processing. -1 means unlimited.")
parser.add_argument("--n_jobs", type=int, default=1,
                    help="Number of parallel jobs for processing. Default is 1.")
# a flag, if add the args in output dir name
parser.add_argument("--cfg", action='store_true', default=False)
parser.add_argument("--verbose", action="store_false", default=True,
                    help="Enable verbose logging.")


if __name__ == "__main__":
    args = parser.parse_args()

    verbose = args.verbose

    # Initialize logger
    logger = ModernLogger(
        name="GenerateTableData",
        level="info" if verbose else "critical"
    )

    dbname = args.dbname
    task_name = args.task_name
    db_cache_dir = args.db_cache_dir
    sample_size = args.sample_size
    table_output_dir = args.table_output_dir

    use_dfs = args.dfs
    dfs_max_depth = args.max_depth
    dfs_max_features = args.max_features
    dfs_number_timedelta = args.n_timedelta
    dfs_time_budget = args.time_budget
    dfs_n_jobs = args.n_jobs
    dfs_with_args = args.cfg

    selection_cfg = args.selection

    # Display configuration
    logger.section(f"Generate Table Data: {dbname} - {task_name}")
    config_info = f"Database: {dbname}\n"
    config_info += f"Task: {task_name}\n"
    config_info += f"Sample Size: {'All' if sample_size == -1 else sample_size}\n"
    config_info += f"DFS Enabled: {use_dfs}\n"
    if use_dfs:
        config_info += f"DFS Max Depth: {dfs_max_depth}\n"
        config_info += f"DFS Max Features: {dfs_max_features}\n"
        config_info += f"DFS Time Window: {'Unlimited' if dfs_number_timedelta == -1 else f'{dfs_number_timedelta} days'}\n"
        config_info += f"Feature Selection: {selection_cfg}\n"
        config_info += f"Parallel Jobs: {dfs_n_jobs}"
    logger.info_panel("Configuration", config_info)

    logger.info("Loading database and task...")
    db = DatabaseFactory.get_db(
        db_name=dbname,
        cache_dir=db_cache_dir,
        upto_test_timestamp=False,
    )

    dataset = DatabaseFactory.get_dataset(
        db_name=dbname,
        cache_dir=db_cache_dir,

    )

    task = DatabaseFactory.get_task(
        db_name=dbname,
        task_name=task_name,
        dataset=dataset,
    )
    logger.success("Database and task loaded successfully")

    # ------------TODO: code support "avito" dataset ---------------
    # because there are many null in foreign key columns, which raise error in featuretools dfs
    if dbname == "avito":
        from utils.data.avito_dataset import preprocess_avito_database
        # after preprocessing it, there is no null in foreign key columns
        # NOTE: after registration, this part of logic should be removed
        preprocess_avito_database(db)
        
    if (dbname in ['ratebeer', 'amazon', 'avito', 'hm']) and use_dfs:
        # WARNING: isolated drop nan is dangerous,
        # for example in avito, drop some "AdsInfo",
        # leads to modification to other table which contains fkey to "AdsInfo"
        # to keep pky-fkey integrity, however there is no logic to do that now.
        # Here is a temporary solution for avito dataset for dfs only
        for table_name, table in db.table_dict.items():
            fkey_cols = list(table.fkey_col_to_pkey_table.keys())
            df_ = table.df.dropna(subset=fkey_cols, how='any')
            dropped_instance_num = table.df.shape[0] - df_.shape[0]
            table.df = df_
            if dropped_instance_num > 0:
                logger.warning(
                    f"Table {table_name} dropped {dropped_instance_num} rows with null foreign keys")
            # not allowed to reindex

        # for no-pky table, add a column as pkey
        for table_name, table in db.table_dict.items():
            if not table.pkey_col:
                pkey_name = table_name + "_id"
                table.df[pkey_name] = range(len(table.df))
                table.pkey_col = pkey_name
                logger.info(
                    f"Table {table_name} has no primary key, added column {pkey_name} as primary key")

    if dbname == "amazon":
        # TODO put it in dataset class
        db.table_dict['product'].df['category'] = \
            db.table_dict['product'].df['category'].astype('string')

    # ------------- move this part of logic into the data class for preprocessing.

    entity_table = db.table_dict[task.entity_table]
    entity_df = entity_table.df

    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test", mask_input_cols=False)

    # --------------------------- sample_training data
    if sample_size > 0 and train_table.df.shape[0] > sample_size:
        sampled_idx = np.random.permutation(len(train_table.df))[:sample_size]
        train_table.df = train_table.df.iloc[sampled_idx]

    if sample_size > 0 and val_table.df.shape[0] > sample_size:
        sampled_idx = np.random.permutation(len(val_table.df))[:sample_size]
        val_table.df = val_table.df.iloc[sampled_idx]

    if sample_size > 0 and test_table.df.shape[0] > sample_size:
        sampled_idx = np.random.permutation(len(test_table.df))[:sample_size]
        test_table.df = test_table.df.iloc[sampled_idx]

    # ------------------------ preprocess the tabular data
    dfs: Dict[str, pd.DataFrame] = {}

    # construct entity set
    process_start_time = time.time()  # overall processing start time
    build_es_start_time = time.time()

    dataframes = {}
    relationships = []
    es = None
    dfs_kwargs: Dict[str, Any] = {}

    selected_columns = []
    if use_dfs:
        logger.section("Deep Feature Synthesis (DFS)")

        # initialize the EntitySet for featuretools dfs.
        for table_name, table in db.table_dict.items():
            dataframes[table_name] = (table.df, table.pkey_col) \
                if not table.time_col else (table.df, table.pkey_col, table.time_col)

            for fkey_col, pkey_table in table.fkey_col_to_pkey_table.items():
                pkey_table_pkey_col = db.table_dict[pkey_table].pkey_col

                # need to check the relation contains nulls
                # if Nan value is included will leads to Int64Dtype() -> int64 fail
                # =================== TODO:code supports "event" dataset =================
                if dbname == "event" and (db.table_dict[pkey_table].df[pkey_table_pkey_col].hasnans or
                                          db.table_dict[table_name].df[fkey_col].hasnans):
                    logger.warning(
                        f"Null values in relationship: {table_name}.{fkey_col} -> {pkey_table}.{pkey_table_pkey_col}, skipping")
                    continue
                # ====[Special case] need incorporates the logic into Dataset class =============

                relationships.append(
                    (pkey_table, pkey_table_pkey_col, table_name, fkey_col)
                )

        es = ft.EntitySet(
            id=dbname,
            dataframes=dataframes,
            relationships=relationships,
        )

        # set for last time index
        es.add_last_time_indexes()
        time_window = dfs_number_timedelta * \
            task.timedelta if dfs_number_timedelta > 0 else None

        build_es_end_time = time.time()
        build_es_duration = build_es_end_time - build_es_start_time
        logger.success(f"EntitySet built in {build_es_duration:.2f}s")

        dfs_kwargs = {
            'entityset': es,
            'target_dataframe_name': task.entity_table,
            'max_depth': dfs_max_depth,
            'max_features': dfs_max_features,
            'training_window': time_window,
            'n_jobs': dfs_n_jobs,
            'verbose': verbose,
            'agg_primitives': ["sum", "max", "min", "mean", "count", "percent_true", "num_unique", "mode"]
        }
        # Log DFS parameters
        logger.info(
            f"DFS Parameters: max_depth={dfs_max_depth}, max_features={dfs_max_features}, time_window={time_window}, n_jobs={dfs_n_jobs}, time_budget={'unlimited' if dfs_time_budget == -1 else f'{dfs_time_budget} min'}")

    # ---------------- generate the feature matrix for each split
    feature_matrix_n = None  # number of features in feature matrix in dfs

    for split, table in [
        ("train", train_table),
        ("val", val_table),
        ("test", test_table),
    ]:
        left_entity = task.entity_col

        if not use_dfs:
            entity_df = entity_df.astype(
                {entity_table.pkey_col: table.df[left_entity].dtype})
            dfs[split] = table.df.merge(
                entity_df,
                how="left",
                left_on=left_entity,
                right_on=entity_table.pkey_col,
                suffixes=('', '_extend')
            )
            continue

        # use dfs to generate feature matrix

        # construct the cutoff time and instance
        cutoff_times = table.df.copy()

        # TODO: after refine and standardize the dataset and task part, don't need this filter

        # specifically remove "index" column in "event" db "user-ignore"
        involved_cols = [task.time_col, left_entity, task.target_col]
        cutoff_times = cutoff_times[involved_cols]
        # keep the column name is same as the entity pkey column
        cutoff_times = cutoff_times.rename(
            columns={left_entity: entity_table.pkey_col}
        )
        cutoff_times['time'] = table.df[task.time_col]

        split_start_time = time.time()
        logger.info(f"Starting DFS for {split} split...")

        dfs_kwargs['cutoff_time'] = cutoff_times

        feature_matrix, feature_instances = ft.dfs(**dfs_kwargs)

        split_end_time = time.time()
        split_duration = split_end_time - split_start_time
        logger.success(
            f"DFS for {split} split completed in {split_duration:.2f}s")

        # assertion that generated features in training is equal to val/test
        if not feature_matrix_n:
            feature_matrix_n = len(feature_matrix.columns)
        else:
            assert feature_matrix_n == len(feature_matrix.columns), \
                f"Feature matrix in {split} has different number of features {len(feature_matrix.columns)} from previous {feature_matrix_n}"

        if split == "train":
            if not selection_cfg:
                selected_columns = feature_matrix.columns.tolist()
            else:
                start_time = time.time()
                fm = remove_highly_null_features(feature_matrix)
                fm = remove_single_value_features(fm)
                fm = remove_highly_correlated_features(fm)
                selected_columns = fm.columns.tolist()
                end_time = time.time()
                duration = end_time - start_time
                logger.success(
                    f"Feature selection completed in {duration:.2f}s")
                logger.success(
                    f"Selected {len(selected_columns)}/{len(feature_matrix.columns)} features")

        feature_matrix = feature_matrix[selected_columns]

        # change the categorical dtype in feature_matrix to object
        # WARNING: the featuretools will automatically assign the dtype for input dataframe.
        # Some categorical dtype will raise issue for type inference especially through sampling
        for col in feature_matrix.columns:
            if str(feature_matrix[col].dtype) == "category":
                feature_matrix[col] = feature_matrix[col].astype("object")

        dfs[split] = feature_matrix.reset_index()
        # convert entity_col from index to column

    process_end_time = time.time()
    overall_duration = process_end_time - process_start_time
    logger.section("Processing Complete")
    logger.success(
        f"Overall processing completed in {overall_duration:.2f}s ({overall_duration/60:.2f} min)")

    # ------------------ Construct Table for type inference
    object_table = Table(
        df=dfs["train"],
        fkey_col_to_pkey_table=train_table.fkey_col_to_pkey_table,
        pkey_col=train_table.pkey_col,
        time_col=train_table.time_col,
    )

    # ------------------- configure the column types
    table_col_types = infer_type_in_table(
        object_table,
        " ",
        verbose=verbose
    )

    remove_pkey_fkey(
        table_col_types,
        object_table,
    )

    if task.task_type == TaskType.BINARY_CLASSIFICATION:
        table_col_types[task.target_col] = stype.categorical
    elif task.task_type == TaskType.REGRESSION:
        table_col_types[task.target_col] = stype.numerical
    elif task.task_type == TaskType.MULTILABEL_CLASSIFICATION:
        table_col_types[task.target_col] = stype.embedding
    else:
        raise ValueError(f"Unsupported task type called {task.task_type}")

    # assign the time column in col_types_dict
    if task.time_col:
        table_col_types[task.time_col] = stype.timestamp

    data = TableData(
        train_df=dfs["train"],
        val_df=dfs["val"],
        test_df=dfs["test"],
        col_to_stype=table_col_types,
        target_col=task.target_col,
        task_type=task.task_type,
    )

    dirname = dbname + "-" + task_name

    if use_dfs and dfs_with_args:
        dirname += f"-dfs-depth{dfs_max_depth}-feat{dfs_max_features}-tw{dfs_number_timedelta}"

    if table_output_dir:
        path = os.path.join(table_output_dir, dirname)

        text_embedder_cfg = get_text_embedder_cfg()
        data.materilize(
            col_to_text_embedder_cfg=text_embedder_cfg,
        )

        data.save_to_dir(
            path
        )

        logger.file_saved(path, f"{dbname}-{task_name}")
