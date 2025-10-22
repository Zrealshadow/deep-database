#!/usr/bin/env python3
"""
TabPFN Baseline for Temporal Degradation
使用滑动窗口策略：预测每个版本时，用该版本之前最近的1000条数据作为context
"""

from tabpfn import TabPFNClassifier, TabPFNRegressor
import argparse
import numpy as np
import pandas as pd
import time
import json
from pathlib import Path
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.metrics import roc_auc_score, mean_absolute_error
import warnings
from utils.data import TableData
import os

warnings.filterwarnings('ignore')


def set_seed(seed=42):
    np.random.seed(seed)


def load_all_data_with_timestamps(data_dir):
    """
    Load data for TabPFN: ONLY use numerical features from train_df.
    We filter by StatType.NUMERICAL to ensure we only get numeric columns.
    """
    # Load TableData (do NOT materialize)
    table_data = TableData.load_from_dir(data_dir)
    
    print(f"\n  📊 Feature Extraction (Numerical Only):")
    print(f"  Total columns in train_df: {len(table_data.train_df.columns)}")
    
    # Get ONLY numerical columns from col_to_stype
    # Count all types for verification
    type_counts = {}
    for col, stype in table_data.col_to_stype.items():
        stype_str = str(stype)
        type_counts[stype_str] = type_counts.get(stype_str, 0) + 1
    
    print(f"  Column types: {type_counts}")
    
    # Filter ONLY numerical (compare as string)
    numerical_cols = [col for col, stype in table_data.col_to_stype.items() 
                      if 'numerical' in str(stype).lower()]
    
    print(f"  ✅ Selected numerical columns: {len(numerical_cols)}")
    
    if len(numerical_cols) == 0:
        raise ValueError(f"No numerical features found! Available types: {type_counts}")
    
    # Extract features from train_df (these are from CSV, not materialized)
    X = table_data.train_df[numerical_cols].fillna(0).values.astype(np.float32)
    y = table_data.train_df[table_data.target_col].values
    timestamps = table_data.train_df['timestamp'].values
    
    # Check for issues
    if np.isnan(X).any():
        nan_count = np.isnan(X).sum()
        print(f"  ⚠️  Found {nan_count} NaN after fillna, filling again with 0")
        X = np.nan_to_num(X, nan=0.0)
    
    if np.isinf(X).any():
        inf_count = np.isinf(X).sum()
        print(f"  ⚠️  Found {inf_count} Inf values, replacing with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"  Final shape: X={X.shape}, y={y.shape}, timestamps={timestamps.shape}")
    
    return X, y, timestamps, numerical_cols


def get_context_data(X_all, y_all, timestamps_all, before_dates, max_samples=1000):
    """
    Get context data: samples with timestamps in before_dates, taking the most recent max_samples.
    
    Args:
        X_all: All features
        y_all: All labels
        timestamps_all: All timestamps (may not be sorted!)
        before_dates: List of dates to include
        max_samples: Maximum number of samples to return (most recent)
    
    Returns:
        X_context, y_context: Most recent max_samples from before_dates
    """
    # Get all samples matching before_dates
    mask = np.isin(timestamps_all, before_dates)
    indices = np.where(mask)[0]
    
    if len(indices) == 0:
        return None, None
    
    # Sort indices by timestamp to get chronological order
    timestamps_subset = timestamps_all[indices]
    sorted_order = np.argsort(timestamps_subset)
    indices_sorted = indices[sorted_order]
    
    # Take the most recent max_samples (last ones chronologically)
    if len(indices_sorted) > max_samples:
        indices_sorted = indices_sorted[-max_samples:]
    
    return X_all[indices_sorted], y_all[indices_sorted]


def main():
    parser = argparse.ArgumentParser(description="TabPFN Temporal Analysis - Config-Based")
    parser.add_argument("--dataset_id", type=str, required=True, 
                        help="Dataset ID from config (e.g., 'trial-study-outcome')")
    parser.add_argument("--config_file", type=str, default="./qzero_config.json", 
                        help="Path to config JSON")

    # all those can be default,
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu or cuda")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./result_raw_from_server/tabpfn",
                        help="Output directory for results")
    parser.add_argument("--max_features", type=int, default=100,
                        help="Max features for SelectKBest")
    parser.add_argument("--max_context_samples", type=int, default=1000,
                        help="Max context samples for sliding window")
    args = parser.parse_args()
    
    # Load config
    with open(args.config_file, 'r') as f:
        config = json.load(f)

    if args.dataset_id not in config:
        raise ValueError(f"Dataset {args.dataset_id} not found in config")

    dataset_config = config[args.dataset_id]

    # Extract from config
    data_dir = dataset_config['data_dir']
    task_type = dataset_config['task_type']
    groups = dataset_config['groups']
    
    # Convert groups to test_dates_list format
    parts = []
    for group in groups:
        if len(group) > 1:
            parts.append(','.join(group))
        else:
            parts.append(group[0])
    test_dates_list = ';'.join(parts)

    print(f"📋 Config: {args.dataset_id}")
    print(f"   Groups: {len(groups)}, Task: {task_type}, Data: {data_dir}")

    set_seed(args.seed)

    print("=" * 80)
    print("TabPFN Temporal Analysis - Sliding Window Context")
    print("=" * 80)
    print(f"🎯 Task: {task_type}")
    print(f"🎲 Seed: {args.seed}")
    print(f"🖥️  Device: {args.device}")
    print(f"📏 Max context: {args.max_context_samples} samples")
    print(f"📏 Max features: {args.max_features}")

    test_dates_groups = [d.strip().split(",") for d in test_dates_list.split(";")]
    print(f"\n📅 Will predict {len(test_dates_groups)} versions")

    print(f"\n📊 Loading all data...")
    X_all, y_all, timestamps_all, feature_cols = load_all_data_with_timestamps(data_dir)
    print(f"  Total samples: {len(X_all):,}")
    print(f"  Total features: {X_all.shape[1]}")
    print(f"  Unique timestamps: {len(np.unique(timestamps_all))}")
    print(f"  Target range: [{y_all.min():.6f}, {y_all.max():.6f}], mean={y_all.mean():.6f}")

    all_dates = sorted(np.unique(timestamps_all))
    print(f"  Date range: {all_dates[0]} to {all_dates[-1]}")

    # Feature selection based on task type
    if X_all.shape[1] > args.max_features:
        print(f"\n🔧 Selecting top {args.max_features} features...")
        if task_type == "classification":
            selector = SelectKBest(f_classif, k=args.max_features)
        else:
            selector = SelectKBest(f_regression, k=args.max_features)
        X_all_selected = selector.fit_transform(X_all, y_all)
        print(f"  ✅ Features: {X_all.shape[1]} → {X_all_selected.shape[1]}")
    else:
        selector = None
        X_all_selected = X_all

    # Create model based on task type
    print(f"\n🏗️  Creating TabPFN model...")
    if task_type == "classification":
        clf = TabPFNClassifier(device=args.device)
        metric_name = "roc_auc"
    else:
        clf = TabPFNRegressor(device=args.device)
        metric_name = "mae"

    results = []
    print(f"\n📊 Testing with sliding context:")
    print("=" * 80)

    for i, test_dates in enumerate(test_dates_groups):
        # ⏱️ 开始计时 - 每个test version从头到尾的时间
        start_time = time.time()
        
        if test_dates[0] == "dummy":
            continue

        mask_test = np.isin(timestamps_all, test_dates)
        X_test = X_all_selected[mask_test]
        y_test = y_all[mask_test]
        if len(X_test) == 0:
            print(f"  Version {i + 1}: No data, skipping")
            continue

        test_date_indices = [all_dates.index(d) for d in test_dates if d in all_dates]
        if not test_date_indices:
            print(f"  Version {i + 1}: Date not found, skipping")
            continue

        min_test_idx = min(test_date_indices)
        context_dates = all_dates[:min_test_idx]

        if len(context_dates) == 0:
            print(f"  Version {i + 1} ({','.join(test_dates)}): No context available, skipping")
            continue

        X_context, y_context = get_context_data(
            X_all_selected, y_all, timestamps_all, context_dates, args.max_context_samples
        )
        if X_context is None or len(X_context) == 0:
            print(f"  Version {i + 1}: No context data, skipping")
            continue

        if len(np.unique(y_context)) < 2:
            print(f"  Version {i + 1}: Context has single class, skipping")
            continue

        # ⏱️ 记录fit开始时间
        fit_start = time.time()
        
        # Try to fit - may fail if context has constant features
        try:
            clf.fit(X_context, y_context)
            fit_time = time.time() - fit_start
        except ValueError as e:
            if "constant" in str(e).lower():
                print(f"  Version {i + 1} ({','.join(test_dates)}): Context has constant features, skipping")
                continue
            else:
                raise  # Re-raise if it's a different error
        
        # ⏱️ 记录inference开始时间
        inference_start = time.time()
        
        # Predict based on task type
        if task_type == "classification":
            if len(np.unique(y_test)) < 2:
                print(f"  Version {i + 1} ({','.join(test_dates)}): Test single class, skipping")
                continue
            test_pred_proba = clf.predict_proba(X_test)[:, 1]
            inference_time = time.time() - inference_start
            test_metric = roc_auc_score(y_test, test_pred_proba)
            context_pred_proba = clf.predict_proba(X_context)[:, 1]
            context_metric = roc_auc_score(y_context, context_pred_proba)
            metric_display = f"AUC={test_metric:.4f}"
        else:  # regression
            test_pred = clf.predict(X_test)
            inference_time = time.time() - inference_start
            test_metric = mean_absolute_error(y_test, test_pred)
            context_pred = clf.predict(X_context)
            context_metric = mean_absolute_error(y_context, context_pred)
            metric_display = f"MAE={test_metric:.4f}"
        
        # ⏱️ 总时间（从头到尾）
        total_time = time.time() - start_time

        print(
            f"  Version {i + 1} ({','.join(test_dates)}): "
            f"Context={len(X_context):,}, Test {metric_display}, "
            f"Time={total_time:.2f}s (fit={fit_time:.2f}s, infer={inference_time:.4f}s)"
        )

        results.append(
            {
                "test_version_index": i + 1,
                "test_dates": ",".join(test_dates),
                "test_samples": len(X_test),
                "test_metric": test_metric,
                "context_samples": len(X_context),
                "context_dates": f"{context_dates[0]}~{context_dates[-1]}",
                "context_metric": context_metric,
                "total_time_seconds": total_time,
                "fit_time_seconds": fit_time,
                "inference_time_seconds": inference_time,
                "inference_time_per_sample_ms": (inference_time / len(X_test)) * 1000,
                "task_type": task_type,
            }
        )

    os.makedirs(args.output_dir, exist_ok=True)
    dataset_name = Path(data_dir).name
    model_name = f"TabPFN"

    if len(results) == 0:
        print(f"\n⚠️  No valid results")
        return

    for result in results:
        result.update(
            {
                "dataset": dataset_name,
                "model_name": model_name,
                "model": "TabPFN",
                "max_features": args.max_features,
                "max_context_samples": args.max_context_samples,
                "metric_name": metric_name,
                "seed": args.seed,
            }
        )

    csv_filename = f"{dataset_name}_TabPFN_sliding_results.csv"
    result_file = os.path.join(args.output_dir, csv_filename)
    result_df = pd.DataFrame(results)

    if os.path.exists(result_file):
        existing_df = pd.read_csv(result_file)
        result_df = pd.concat([existing_df, result_df], ignore_index=True)

    result_df.to_csv(result_file, index=False)
    print(f"\n💾 Results saved: {result_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()



"""
# Test command (Config-Based):
python ./evl_cmd/qzero_tabpfn.py --dataset_id trial-study-outcome

# Or with custom output:
python ./evl_cmd/qzero_tabpfn.py --dataset_id trial-study-outcome --output_dir ./test_output
"""