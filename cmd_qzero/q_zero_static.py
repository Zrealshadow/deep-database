#!/usr/bin/env python3
"""
Temporal Degradation: Train Once, Test on Multiple Versions
训练一次，在多个时间版本上测试
"""

import torch
import math
import argparse
import copy
import pandas as pd
import os
import json
import numpy as np
import random
from tqdm import tqdm
from pathlib import Path

from torch_frame.nn.models import MLP, ResNet, FTTransformer
from torch.nn import L1Loss, BCEWithLogitsLoss
from sklearn.metrics import mean_absolute_error, roc_auc_score
from relbench.base import TaskType
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import torch_frame

from model.base import construct_stype_encoder_dict, default_stype_encoder_cls_kwargs
from utils.resource import get_text_embedder_cfg
from utils.data import TableData


def set_seed(seed=42):
    """设置所有随机种子以保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def deactivate_dropout(net: torch.nn.Module):
    """Deactivate dropout layers in the model for regression task"""
    deactive_nn_instances = (
        torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)
    for module in net.modules():
        if isinstance(module, deactive_nn_instances):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False


def evaluate(net: torch.nn.Module, loader: torch.utils.data.DataLoader, 
            device: torch.device, is_regression: bool = False):
    """Evaluate model on a dataset"""
    pred_list = []
    y_list = []

    if not is_regression:
        net.eval()

    for batch in tqdm(loader, leave=False, desc="Evaluating"):
        with torch.no_grad():
            batch = batch.to(device)
            y = batch.y.float()
            pred = net(batch)
            pred = pred.view(-1) if pred.size(1) == 1 else pred
            pred_list.append(pred.detach().cpu())
            y_list.append(y.detach().cpu())
    
    # 检查是否有数据
    if len(pred_list) == 0:
        # 返回空数组
        return np.array([]), np.array([]), np.array([])
    
    pred_list = torch.cat(pred_list, dim=0)
    pred_logits = pred_list
    pred_list = torch.sigmoid(pred_list)
    y_list = torch.cat(y_list, dim=0).numpy()
    return pred_logits.numpy(), pred_list.numpy(), y_list


def get_temporal_indices(data_dir, dates):
    """获取指定日期的数据索引"""
    train_csv = pd.read_csv(os.path.join(data_dir, 'train.csv'), low_memory=False)
    mask = train_csv['timestamp'].isin(dates)
    return train_csv[mask].index.tolist()


def main():
    parser = argparse.ArgumentParser(description="Retrain Strategy - Config-Based")
    
    # Config-based arguments
    parser.add_argument("--dataset_id", type=str, required=True,
                        help="Dataset ID from config (e.g., 'trial-study-outcome')")
    parser.add_argument("--train_group_idx", type=int, required=True,
                        help="Train group index (0-based, e.g., 0 for first group)")
    parser.add_argument("--config_file", type=str, default="./qzero_config.json",
                        help="Path to config JSON")
    parser.add_argument("--val_split_ratio", type=float, default=0.1)
    
    # Model
    parser.add_argument("--model", type=str, choices=["MLP", "ResNet", "FTTransformer"], default="ResNet")
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout_prob", type=float, default=0.2)
    parser.add_argument("--normalization", type=str, default="layer_norm")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--early_stop_threshold", type=int, default=15)
    parser.add_argument("--max_batches_per_epoch", type=int, default=20,
                        help="Max batches per epoch ")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./result_raw_from_server/retrain")
    parser.add_argument("--model_name", type=str, default=None)
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    
    if args.dataset_id not in config:
        raise ValueError(f"Dataset {args.dataset_id} not found in config")
    
    dataset_config = config[args.dataset_id]
    data_dir = dataset_config['data_dir']
    task_type_str = dataset_config['task_type']
    all_groups = dataset_config['groups']
    
    # Validate train_group_idx
    if args.train_group_idx < 0 or args.train_group_idx >= len(all_groups):
        raise ValueError(f"train_group_idx {args.train_group_idx} out of range [0, {len(all_groups)-1}]")
    
    # Get train dates from config
    train_dates = all_groups[args.train_group_idx]
    
    # Get test dates: all groups after train_group_idx
    test_dates_groups = all_groups[args.train_group_idx + 1:]
    
    if len(test_dates_groups) == 0:
        print(f"⚠️  No test groups available after train_group_idx={args.train_group_idx}")
        return
    
    # Setup
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    
    print("=" * 80)
    print("Retrain Strategy: Train Once, Test on Multiple Versions")
    print("=" * 80)
    print(f"📋 Dataset: {args.dataset_id}")
    print(f"🎯 Task: {task_type_str}")
    print(f"🎲 Seed: {args.seed}")
    print(f"🖥️  Device: {device}")
    
    print(f"\n📅 Dates:")
    print(f"  Train group [{args.train_group_idx}]: {train_dates}")
    print(f"  Test: {len(test_dates_groups)} groups")
    for i, dates in enumerate(test_dates_groups[:3], 1):
        print(f"    Test {i}: {dates}")
    if len(test_dates_groups) > 3:
        print(f"    ... and {len(test_dates_groups)-3} more")
    
    # Load data
    table_data = TableData.load_from_dir(data_dir)
    if not table_data.is_materialize:
        print(" Materialize dataset ing....")
        text_cfg = get_text_embedder_cfg(device="cpu")
        table_data.materilize(col_to_text_embedder_cfg=text_cfg)
    else:
        print(" Skip materialize dataset ing....")
    
    # Get temporal train indices
    temporal_train_indices = get_temporal_indices(data_dir, train_dates)
    print(f"\n📊 Training data: {len(temporal_train_indices):,} samples")
    
    # Split train/val - 按时间顺序，最后20%作为val
    temporal_train_y = table_data.train_tf.y[temporal_train_indices].numpy()
    
    # 按时间顺序split（不random shuffle）
    n_total = len(temporal_train_indices)
    n_train = int(n_total * (1 - args.val_split_ratio))
    
    train_idx = list(range(n_train))  # 前80%
    val_idx = list(range(n_train, n_total))  # 后20%
    
    print(f"  Train: {len(train_idx):,} samples (前{int((1-args.val_split_ratio)*100)}%, 按时间顺序)")
    print(f"  Val:   {len(val_idx):,} samples (后{int(args.val_split_ratio*100)}%, 按时间顺序)")
    
    # Create train/val loaders
    temporal_train_tf = Subset(table_data.train_tf, temporal_train_indices)
    train_subset = Subset(temporal_train_tf, train_idx)
    val_subset = Subset(temporal_train_tf, val_idx)
    
    # 检查是否有数据
    if len(train_idx) == 0:
        print(f"\n⚠️  Training set is empty! Skipping this experiment.")
        return
    
    train_loader = torch_frame.data.DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    
    # Validation可能为空
    if len(val_idx) > 0:
        val_loader = torch_frame.data.DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    else:
        val_loader = None
        print(f"\n⚠️  Validation set is empty! Will skip validation.")
    
    # Setup task
    if table_data.task_type == TaskType.REGRESSION:
        loss_fn = L1Loss()
        evaluate_func = mean_absolute_error
        higher_is_better = False
        is_regression = True
    else:
        pos_count = temporal_train_y.sum()
        neg_count = len(temporal_train_y) - pos_count
        
        print(f"\n⚖️  Class balance:")
        print(f"  Negative: {neg_count:,} ({neg_count/len(temporal_train_y)*100:.1f}%)")
        print(f"  Positive: {pos_count:,} ({pos_count/len(temporal_train_y)*100:.1f}%)")
        
        loss_fn = BCEWithLogitsLoss()
        evaluate_func = roc_auc_score
        higher_is_better = True
        is_regression = False
    
    # Build model
    stype_encoder_dict = construct_stype_encoder_dict(default_stype_encoder_cls_kwargs)
    
    out_channels = 1
    model_args = {
        "channels": args.channels,
        "out_channels": out_channels,
        "num_layers": args.num_layers,
        "col_names_dict": table_data.col_names_dict,
        "stype_encoder_dict": stype_encoder_dict,
        "col_stats": table_data.col_stats,
    }
    
    if args.model.lower() in ["resnet", "mlp"]:
        model_args.update({
            "dropout_prob": args.dropout_prob,
            "normalization": args.normalization,
        })
    
    print(f"\n🏗️  Model: {args.model}")
    print(f"  Channels: {args.channels}, Layers: {args.num_layers}")
    
    # Create model
    if args.model.lower() == "mlp":
        net = MLP(**model_args)
    elif args.model.lower() == "resnet":
        net = ResNet(**model_args)
    elif args.model.lower() in ["fttransformer", "fttrans"]:
        net = FTTransformer(**model_args)
    else:
        raise "not model matched exisint options, mlp, resnet, fttransformer"
    
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    
    if is_regression:
        deactivate_dropout(net)
    
    # Training
    print(f"\n🚀 Training...")
    best_val_metric = -math.inf if higher_is_better else math.inf
    best_epoch = 0
    best_model_state = copy.deepcopy(net.state_dict())  # 初始化为当前模型，防止None
    patience = 0
    
    for epoch in range(args.num_epochs):
        net.train()
        loss_accum = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, leave=False, desc=f"Epoch {epoch}")):
            # Limit max batches per epoch (like dnn_baseline)
            if batch_idx >= args.max_batches_per_epoch:
                break
            
            optimizer.zero_grad()
            batch = batch.to(device)
            pred = net(batch).view(-1)
            y = batch.y.float()
            loss = loss_fn(pred, y)
            
            # 检查loss是否为NaN或inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n⚠️  Warning: Invalid loss detected, skipping batch...")
                continue
                
            loss.backward()
            optimizer.step()
            loss_accum += loss.item()
            batch_count += 1
        
        train_loss = loss_accum / max(batch_count, 1)
        
        # Validate (如果有validation set)
        if val_loader is None:
            # 没有validation，跳过early stopping，训练到底
            print(f"Epoch {epoch:3d} | Loss: {train_loss:.6f} | No validation")
            continue
        
        _, val_pred, val_y = evaluate(net, val_loader, device, is_regression)
        
        # 检查NaN
        if np.isnan(val_pred).any():
            print(f"\n⚠️  Warning: NaN detected in predictions at epoch {epoch}, skipping...")
            continue
        
        # 检查validation set是否只有一个类别（无法计算AUC）
        if not is_regression and len(np.unique(val_y)) < 2:
            print(f"Epoch {epoch:3d} | Loss: {train_loss:.6f} | Val: Only one class, skipping validation")
            # 当validation失效时，跳过early stopping，训练到底
            # 不更新best_model_state，用初始模型
            continue
            
        val_metric = evaluate_func(val_y, val_pred)
        
        print(f"Epoch {epoch:3d} | Loss: {train_loss:.6f} | Val AUC: {val_metric:.6f}")
        
        is_better = (val_metric > best_val_metric) if higher_is_better else (val_metric < best_val_metric)
        if is_better:
            best_val_metric = val_metric
            best_epoch = epoch
            best_model_state = copy.deepcopy(net.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= args.early_stop_threshold:
                print(f"\n⚠️  Early stopping at epoch {epoch}")
                break
    
    # Load best model
    net.load_state_dict(best_model_state)
    
    # Evaluate on train
    _, train_pred, train_y = evaluate(net, train_loader, device, is_regression)
    train_metric = evaluate_func(train_y, train_pred)
    
    print(f"\n" + "=" * 80)
    print(f"✅ Best Model (Epoch {best_epoch}):")
    print(f"   Train AUC: {train_metric:.6f}")
    print(f"   Val AUC:   {best_val_metric:.6f}")
    
    # Test on all versions (on CPU, with timing)
    results = []
    metric_name = "MAE" if is_regression else "AUC"
    
    if len(test_dates_groups) > 0 and test_dates_groups[0][0] != 'dummy':
        print(f"\n📊 Testing on {len(test_dates_groups)} versions (CPU):")
        print("=" * 80)
        
        # Move model to CPU for testing
        cpu_device = torch.device('cpu')
        net_cpu = net.to(cpu_device)
        net_cpu.eval()
        
        for i, test_dates in enumerate(test_dates_groups):
            test_indices = get_temporal_indices(data_dir, test_dates)
            
            # 跳过空的test set
            if len(test_indices) == 0:
                print(f"  Test version {i+1} ({','.join(test_dates)}): No data, skipping")
                continue
            
            test_tf = Subset(table_data.train_tf, test_indices)
            test_loader = torch_frame.data.DataLoader(test_tf, batch_size=args.batch_size, shuffle=False)
            
            # ⏱️ Time the test (CPU)
            import time
            test_start_time = time.time()
            _, test_pred, test_y = evaluate(net_cpu, test_loader, cpu_device, is_regression)
            test_time = time.time() - test_start_time
            
            # 检查是否返回空数组
            if len(test_y) == 0:
                print(f"  Test version {i+1} ({','.join(test_dates)}): Empty dataset, skipping")
                continue
            
            test_metric = evaluate_func(test_y, test_pred)
            
            print(f"  Test version {i+1} ({','.join(test_dates)}): "
                  f"{metric_name} = {test_metric:.6f}, Time = {test_time:.2f}s")
            
            results.append({
                'test_version_index': i+1,
                'test_dates': ','.join(test_dates),
                'test_samples': len(test_indices),
                'test_metric': test_metric,
                'test_time_seconds': test_time,
                'test_time_per_sample_ms': (test_time / len(test_indices)) * 1000,
            })
    else:
        print(f"\n⚠️  No test data, only validation results will be saved.")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成模型名称，包含数据集名和训练日期
    dataset_name = Path(data_dir).name
    
    # 用日期生成版本字符串 (例如: 2015-04-25,26,27 → 0425-0427)
    if len(train_dates) == 1:
        train_version_str = train_dates[0].replace('-', '')[-4:]  # 例如: 0425
    else:
        first_date = train_dates[0].replace('-', '')[-4:]  # 0425
        last_date = train_dates[-1].replace('-', '')[-4:]  # 0427
        train_version_str = f"{first_date}-{last_date}"  # 0425-0427
    
    model_name = args.model_name or f"{dataset_name}_{args.model}_train{train_version_str}"
    
    # 即使没有test结果，也保存一行记录训练信息
    if len(results) == 0:
        print(f"\n⚠️  No valid test results, saving training info only")
        results.append({
            'test_version_index': 0,
            'test_dates': 'N/A',
            'test_samples': 0,
            'test_metric': np.nan,
        })
    
    # Save all test results
    for result in results:
        result.update({
            'dataset': dataset_name,
            'model_name': model_name,
            'model': args.model,
            'channels': args.channels,
            'num_layers': args.num_layers,
            'train_dates': ','.join(train_dates),
            'val_split_ratio': args.val_split_ratio,
            'train_samples': len(train_idx),
            'val_samples': len(val_idx),
            'best_epoch': best_epoch,
            'train_metric': train_metric,
            'val_metric': best_val_metric,
            'metric_name': evaluate_func.__name__,
            'seed': args.seed,
        })
    
    # CSV文件名包含数据集和训练日期
    csv_filename = f"{dataset_name}_train{train_version_str}_results.csv"
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

