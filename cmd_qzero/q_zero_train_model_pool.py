#!/usr/bin/env python3
"""
Train Model Pool: Diverse MLP and ResNet Architectures

Train a pool of models with different architectures (small, medium, large)
Following the same training configuration as dnn_baseline_table_data.py

No temporal split - standard train/val/test split from TableData
"""

import argparse
import json
import os
import time
import math
import copy
import pandas as pd
import torch
import torch.nn as nn
import torch_frame
from sklearn.metrics import mean_absolute_error, roc_auc_score
from relbench.base import TaskType

from utils.data import TableData
from model.base import construct_stype_encoder_dict, default_stype_encoder_cls_kwargs
from q_zero.search_space import QZeroMLP, QZeroResNet


def test(net, loader, device, is_regression=False):
    """Test function from dnn_baseline_table_data.py"""
    pred_list = []
    y_list = []

    if not is_regression:
        net.eval()

    for batch in loader:
        with torch.no_grad():
            batch = batch.to(device)
            y = batch.y.float()
            pred = net(batch)
            pred = pred.view(-1) if pred.size(1) == 1 else pred
            pred_list.append(pred.detach().cpu())
            y_list.append(y.detach().cpu())
    
    pred_list = torch.cat(pred_list, dim=0)
    pred_logits = pred_list
    pred_list = torch.sigmoid(pred_list)
    y_list = torch.cat(y_list, dim=0).numpy()
    
    return pred_logits.numpy(), pred_list.numpy(), y_list


def deactivate_dropout(net):
    """Deactivate dropout for regression - from dnn_baseline_table_data.py"""
    deactive_nn_instances = (
        torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)
    for module in net.modules():
        if isinstance(module, deactive_nn_instances):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False


def main():
    parser = argparse.ArgumentParser(description='Train Model Pool')
    parser.add_argument('--dataset_id', type=str, required=True, help='Dataset ID')
    parser.add_argument('--space_name', type=str, required=True, choices=['mlp', 'resnet'], help='Search space')
    parser.add_argument('--architecture', type=str, required=True, help='Architecture (e.g., "64-128-256")')
    parser.add_argument('--model_size', type=str, required=True, choices=['small', 'medium', 'large'], help='Model size category')
    parser.add_argument('--config_file', type=str, default='./q_zero_config.json', help='Dataset config file')
    parser.add_argument('--output_dir', type=str, default='./result_raw_from_server/q_zero_model_performance_pool', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    # Training parameters from dnn_baseline_table_data.py
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--early_stop_threshold', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--max_round_epoch', type=int, default=20, help='Max batches per epoch')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    
    print("=" * 80)
    print("🏋️ Training Model Pool")
    print("=" * 80)
    print(f"Dataset: {args.dataset_id}")
    print(f"Space: {args.space_name.upper()}")
    print(f"Architecture: {args.architecture}")
    print(f"Size: {args.model_size}")
    print(f"Device: {device}")
    
    # Load config
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    
    if args.dataset_id not in config:
        raise ValueError(f"Dataset {args.dataset_id} not found in config")
    
    dataset_config = config[args.dataset_id]
    data_dir = dataset_config['data_dir']
    
    print(f"Data dir: {data_dir}")
    
    # Load data - same as dnn_baseline_table_data.py
    print(f"\n{'─'*80}")
    print("📥 Loading Data")
    print(f"{'─'*80}")
    
    table_data = TableData.load_from_dir(data_dir)
    
    task_type = table_data.task_type
    is_regression = (task_type == TaskType.REGRESSION)
    
    print(f"Task: {task_type}")
    print(f"Is regression: {is_regression}")
    
    # Create data loaders - EXACTLY like dnn_baseline_table_data.py
    data_loaders = {
        idx: torch_frame.data.DataLoader(
            getattr(table_data, f"{idx}_tf"),
            batch_size=args.batch_size,
            shuffle=idx == "train",
            pin_memory=True,
        )
        for idx in ["train", "val", "test"]
    }
    
    print(f"Train samples: {len(table_data.train_tf)}")
    print(f"Val samples: {len(table_data.val_tf)}")
    print(f"Test samples: {len(table_data.test_tf)}")
    
    # Get dataset info
    num_cols = sum(len(v) for v in table_data.col_names_dict.values())
    col_stats = table_data.col_stats
    
    # Parse architecture
    arch_list = [int(x) for x in args.architecture.split('-')]
    
    # Auto-determine encoder_channels (same formula as q_zero_filter.py)
    if args.space_name == 'mlp':
        channel_choices = QZeroMLP.channel_choices
        num_blocks = len(arch_list) + 1  # MLP: num_layers = len(hidden_dims) + 1
    else:
        channel_choices = QZeroResNet.channel_choices
        num_blocks = len(arch_list)  # ResNet: num_layers = len(block_widths)
    
    min_channel = min(channel_choices)
    max_channel = max(channel_choices)
    encoder_channels = int(min_channel * math.sqrt(num_cols / 10))
    encoder_channels = max(min_channel, min(encoder_channels, max_channel))
    
    print(f"\n📐 Architecture Info:")
    print(f"  Features (num_cols): {num_cols}")
    print(f"  Encoder channels: {encoder_channels} (auto-scaled)")
    print(f"  Architecture: {args.architecture}")
    print(f"  Num blocks: {num_blocks}")
    
    # Create stype_encoder_dict
    stype_encoder_dict = construct_stype_encoder_dict(default_stype_encoder_cls_kwargs)
    
    # Create model
    print(f"\n{'─'*80}")
    print("🏗️ Creating Model")
    print(f"{'─'*80}")
    
    if args.space_name == 'mlp':
        net = QZeroMLP(
            channels=encoder_channels,
            out_channels=1,
            num_layers=len(arch_list) + 1,  # MLP: num_layers = len(hidden_dims) + 1
            col_stats=col_stats,
            col_names_dict=table_data.col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
            hidden_dims=arch_list,
            normalization='layer_norm',
            dropout_prob=0.2,
        )
    else:  # resnet
        net = QZeroResNet(
            channels=encoder_channels,
            out_channels=1,
            num_layers=len(arch_list),  # ResNet: num_layers = len(block_widths)
            col_stats=col_stats,
            col_names_dict=table_data.col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
            block_widths=arch_list,
            normalization='layer_norm',
            dropout_prob=0.2,
        )
    
    print(f"✅ Model created")
    
    # Setup loss and metrics - EXACTLY like dnn_baseline_table_data.py
    if is_regression:
        loss_fn = nn.L1Loss()
        evaluate_metric_func = mean_absolute_error
        higher_is_better = False
    else:
        loss_fn = nn.BCEWithLogitsLoss()
        evaluate_metric_func = roc_auc_score
        higher_is_better = True
    
    # Deactivate dropout for regression - EXACTLY like dnn_baseline_table_data.py
    if is_regression:
        deactivate_dropout(net)
    
    net.to(device)
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)
    
    # Train - EXACTLY like dnn_baseline_table_data.py
    print(f"\n{'─'*80}")
    print("🏋️ Training")
    print(f"{'─'*80}")
    
    patience = 0
    best_epoch = 0
    best_val_metric = -math.inf if higher_is_better else math.inf
    best_model_state = None
    
    train_start_time = time.time()
    
    for epoch in range(args.num_epochs):
        loss_accum = count_accum = 0
        net.train()
        
        for idx, batch in enumerate(data_loaders["train"]):
            
            if idx > args.max_round_epoch:
                break
            
            optimizer.zero_grad()
            batch = batch.to(device)
            pred = net(batch)
            pred = pred.view(-1) if pred.size(1) == 1 else pred
            y = batch.y.float()
            loss = loss_fn(pred, y)
            
            loss.backward()
            optimizer.step()
            loss_accum += loss.item()
            count_accum += 1
        
        train_loss = loss_accum / count_accum
        val_logits, _, val_pred_hat = test(
            net, data_loaders["val"], device, is_regression=is_regression)
        
        val_metric = evaluate_metric_func(val_pred_hat, val_logits)
        
        print(
            f"==> Epoch: {epoch} => Train Loss: {train_loss:.6f}, Val {evaluate_metric_func.__name__} Metric: {val_metric:.6f}")
        
        if (higher_is_better and val_metric > best_val_metric) or \
           (not higher_is_better and val_metric < best_val_metric):
            best_val_metric = val_metric
            best_epoch = epoch
            best_model_state = copy.deepcopy(net.state_dict())
            patience = 0
            
            test_logits, _, test_pred_hat = test(
                net, data_loaders["test"], device, is_regression=is_regression)
            test_metric = evaluate_metric_func(test_pred_hat, test_logits)
            
            print(
                f"Update the best scores => Test {evaluate_metric_func.__name__} Metric: {test_metric:.6f}")
        else:
            patience += 1
            if patience > args.early_stop_threshold:
                print(f"Early stopping at epoch {epoch}")
                break
    
    train_time = time.time() - train_start_time
    
    # Final test with best model
    net.load_state_dict(best_model_state)
    
    test_start_time = time.time()
    test_logits, _, test_pred_hat = test(
        net, data_loaders["test"], device, is_regression=is_regression)
    test_metric = evaluate_metric_func(test_pred_hat, test_logits)
    test_time = time.time() - test_start_time
    
    print(f"✅ Test {evaluate_metric_func.__name__} Metric: {test_metric:.6f}")
    
    # Save result
    os.makedirs(args.output_dir, exist_ok=True)
    
    result = {
        'dataset': args.dataset_id,
        'space_name': args.space_name,
        'model_size': args.model_size,
        'num_blocks': num_blocks,
        'encoder_channels': encoder_channels,
        'architecture': args.architecture,
        'train_time': train_time,
        'test_time': test_time,
        'total_time': train_time + test_time,
    }
    
    if is_regression:
        result['mae'] = test_metric
    else:
        result['auc'] = test_metric
    
    # Append to CSV
    output_file = os.path.join(args.output_dir, f'{args.dataset_id}_{args.space_name}_pool.csv')
    
    result_df = pd.DataFrame([result])
    
    if os.path.exists(output_file):
        result_df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        result_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*80}")
    print(f"✅ Result saved to: {output_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
