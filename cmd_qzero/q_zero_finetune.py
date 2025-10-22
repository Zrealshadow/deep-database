#!/usr/bin/env python3
"""
Fine-tune Strategy: Incremental fine-tuning on CPU
- Train initial model on group 0 (GPU)
- Test on group 1 (CPU, inference only)
- Fine-tune on group 1 + Test on group 2 (CPU, time = finetune + inference)
- Fine-tune on group 2 + Test on group 3 (CPU, time = finetune + inference)
- ...
"""

import argparse
import json
import os
import sys
import time
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_frame
from torch.nn import L1Loss, BCEWithLogitsLoss
from torch.utils.data import Subset
from sklearn.metrics import mean_absolute_error, roc_auc_score
from relbench.base import TaskType
from torch_frame.nn.models import MLP, ResNet, FTTransformer

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.base import construct_stype_encoder_dict, default_stype_encoder_cls_kwargs
from utils.data import TableData


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_temporal_indices(data_dir, dates):
    """Get indices for given dates from train_df"""
    import warnings
    warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
    table_data = TableData.load_from_dir(data_dir)
    train_df = table_data.train_df
    
    # Convert dates to list if single string
    if isinstance(dates, str):
        dates = [dates]
    
    # Find matching indices
    mask = train_df['timestamp'].isin(dates)
    indices = train_df[mask].index.tolist()
    
    return indices


def evaluate(model, loader, device, is_regression):
    """Evaluate model on loader"""
    model.eval()
    preds = []
    ys = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            pred = pred.view(-1) if pred.size(1) == 1 else pred
            
            if is_regression:
                preds.append(pred.cpu())
            else:
                preds.append(torch.sigmoid(pred).cpu())
            
            ys.append(batch.y.cpu())
    
    if len(preds) == 0:
        return 0.0, np.array([]), np.array([])
    
    pred_tensor = torch.cat(preds, dim=0).numpy()
    y_tensor = torch.cat(ys, dim=0).numpy()
    
    return 0.0, pred_tensor, y_tensor


def finetune_on_data(model, loader, optimizer, loss_fn, device, max_batches=20, num_epochs=10):
    """
    Fine-tune model on new data
    Returns: time spent on fine-tuning
    """
    model.train()
    start_time = time.time()
    
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            
            optimizer.zero_grad()
            batch = batch.to(device)
            pred = model(batch)
            pred = pred.view(-1) if pred.size(1) == 1 else pred
            y = batch.y.float()
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
    
    finetune_time = time.time() - start_time
    return finetune_time


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Strategy - Config-Based")
    
    # Config-based arguments
    parser.add_argument("--dataset_id", type=str, required=True,
                        help="Dataset ID from config (e.g., 'trial-study-outcome')")
    parser.add_argument("--config_file", type=str, default="./qzero_config.json",
                        help="Path to config JSON")
    parser.add_argument("--val_split_ratio", type=float, default=0.1)
    
    # Model
    parser.add_argument("--model", type=str, choices=["MLP", "ResNet", "FTTransformer"], default="ResNet")
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout_prob", type=float, default=0.2)
    parser.add_argument("--normalization", type=str, default="layer_norm")
    
    # Training (initial)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--early_stop_threshold", type=int, default=15)
    
    # Fine-tuning
    parser.add_argument("--finetune_epochs", type=int, default=10,
                        help="Number of epochs for fine-tuning")
    parser.add_argument("--finetune_max_batches", type=int, default=20,
                        help="Max batches per epoch for fine-tuning")
    parser.add_argument("--finetune_lr", type=float, default=0.0001,
                        help="Learning rate for fine-tuning")
    
    # Device
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./result_raw_from_server/finetune")
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
    
    if len(all_groups) < 2:
        print(f"⚠️  Need at least 2 groups for fine-tuning")
        return
    
    # Setup
    set_seed(args.seed)
    train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    cpu_device = torch.device("cpu")
    
    print("=" * 80)
    print("Fine-tune Strategy: Incremental Fine-tuning")
    print("=" * 80)
    print(f"📋 Dataset: {args.dataset_id}")
    print(f"🎯 Task: {task_type_str}")
    print(f"🎲 Seed: {args.seed}")
    print(f"🖥️  Initial Training Device: {train_device}")
    print(f"🖥️  Fine-tuning & Testing Device: CPU")
    print(f"\n📅 Groups: {len(all_groups)}")
    print(f"  Initial train: group 0")
    print(f"  Fine-tune & test: groups 1..{len(all_groups)-1}")
    
    # Load data
    table_data = TableData.load_from_dir(data_dir)
    if not table_data.is_materialize:
        from utils.resource import get_text_embedder_cfg
        text_cfg = get_text_embedder_cfg(device="cpu")
        table_data.materilize(col_to_text_embedder_cfg=text_cfg)
    
    # Initial training on group 0
    train_dates = all_groups[0]
    temporal_train_indices = get_temporal_indices(data_dir, train_dates)
    print(f"\n📊 Initial training data (group 0): {len(temporal_train_indices):,} samples")
    
    # Split train/val
    n_total = len(temporal_train_indices)
    n_train = int(n_total * (1 - args.val_split_ratio))
    train_idx = temporal_train_indices[:n_train]
    val_idx = temporal_train_indices[n_train:]
    
    print(f"  Train: {len(train_idx):,} samples")
    print(f"  Val:   {len(val_idx):,} samples")
    
    # Setup task
    if table_data.task_type == TaskType.REGRESSION:
        loss_fn = L1Loss()
        evaluate_func = mean_absolute_error
        higher_is_better = False
        is_regression = True
        metric_name = "MAE"
    else:
        loss_fn = BCEWithLogitsLoss()
        evaluate_func = roc_auc_score
        higher_is_better = True
        is_regression = False
        metric_name = "AUC"
    
    # Build model
    stype_encoder_dict = construct_stype_encoder_dict(default_stype_encoder_cls_kwargs)
    model_args = {
        "channels": args.channels,
        "out_channels": 1,
        "num_layers": args.num_layers,
        "col_names_dict": table_data.col_names_dict,
        "stype_encoder_dict": stype_encoder_dict,
        "col_stats": table_data.col_stats,
    }
    
    if args.model == "MLP":
        model_args["dropout_prob"] = args.dropout_prob
        model_args["normalization"] = args.normalization
        net = MLP(**model_args)
    elif args.model == "ResNet":
        model_args["dropout_prob"] = args.dropout_prob
        model_args["normalization"] = args.normalization
        net = ResNet(**model_args)
    elif args.model == "FTTransformer":
        net = FTTransformer(**model_args)
    
    net = net.to(train_device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    
    print(f"\n🏗️  Model: {args.model}")
    print(f"  Channels: {args.channels}, Layers: {args.num_layers}")
    
    # Create loaders
    train_subset = Subset(table_data.train_tf, train_idx)
    val_subset = Subset(table_data.train_tf, val_idx)
    train_loader = torch_frame.data.DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch_frame.data.DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
    
    # Initial training
    print(f"\n🚀 Initial Training (GPU)...")
    best_val_metric = -float('inf') if higher_is_better else float('inf')
    best_epoch = 0
    best_model_state = None
    patience = 0
    
    for epoch in range(args.num_epochs):
        net.train()
        epoch_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= args.finetune_max_batches:  # Same limit as fine-tuning
                break
            
            optimizer.zero_grad()
            batch = batch.to(train_device)
            pred = net(batch)
            pred = pred.view(-1) if pred.size(1) == 1 else pred
            y = batch.y.float()
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Validation
        _, val_pred, val_y = evaluate(net, val_loader, train_device, is_regression)
        if len(val_y) > 0:
            val_metric = evaluate_func(val_y, val_pred)
            
            # Check improvement
            improved = (higher_is_better and val_metric > best_val_metric) or \
                      (not higher_is_better and val_metric < best_val_metric)
            
            if improved:
                best_val_metric = val_metric
                best_epoch = epoch
                best_model_state = copy.deepcopy(net.state_dict())
                patience = 0
                print(f"Epoch {epoch:3d} | Loss: {epoch_loss/min(batch_idx+1, args.finetune_max_batches):.6f} | Val {metric_name}: {val_metric:.6f} ✓")
            else:
                patience += 1
                if epoch % 5 == 0:
                    print(f"Epoch {epoch:3d} | Loss: {epoch_loss/min(batch_idx+1, args.finetune_max_batches):.6f} | Val {metric_name}: {val_metric:.6f}")
            
            if patience >= args.early_stop_threshold:
                print(f"\n⚠️  Early stopping at epoch {epoch}")
                break
    
    # Load best model
    if best_model_state is not None:
        net.load_state_dict(best_model_state)
    
    print(f"\n✅ Initial model trained (best epoch: {best_epoch}, val {metric_name}: {best_val_metric:.6f})")
    
    # Move to CPU for fine-tuning and testing
    net = net.to(cpu_device)
    optimizer_finetune = torch.optim.Adam(net.parameters(), lr=args.finetune_lr)
    
    # Fine-tune and test incrementally
    results = []
    
    print(f"\n📊 Fine-tuning & Testing (CPU):")
    print("=" * 80)
    
    for i in range(1, len(all_groups)):
        current_group = all_groups[i]
        current_indices = get_temporal_indices(data_dir, current_group)
        
        if len(current_indices) == 0:
            print(f"  Group {i} ({','.join(current_group)}): No data, skipping")
            continue
        
        current_tf = Subset(table_data.train_tf, current_indices)
        current_loader = torch_frame.data.DataLoader(current_tf, batch_size=args.batch_size, shuffle=True)
        
        # For group 1: only test (no fine-tuning data yet)
        if i == 1:
            # Test only
            test_start = time.time()
            _, test_pred, test_y = evaluate(net, current_loader, cpu_device, is_regression)
            test_time = time.time() - test_start
            
            if len(test_y) == 0:
                print(f"  Group {i}: Empty test data, skipping")
                continue
            
            test_metric = evaluate_func(test_y, test_pred)
            
            print(f"  Group {i} ({','.join(current_group)}): "
                  f"Test {metric_name}={test_metric:.4f}, Time={test_time:.2f}s (inference only)")
            
            results.append({
                'test_version_index': i,
                'test_dates': ','.join(current_group),
                'test_samples': len(current_indices),
                'test_metric': test_metric,
                'finetune_time_seconds': 0.0,
                'inference_time_seconds': test_time,
                'total_time_seconds': test_time,
                'finetune_group': 'none',
            })
        else:
            # Fine-tune on previous group, then test on current group
            prev_group = all_groups[i-1]
            prev_indices = get_temporal_indices(data_dir, prev_group)
            
            if len(prev_indices) == 0:
                print(f"  Group {i}: No fine-tune data from group {i-1}, skipping")
                continue
            
            # Fine-tune loader
            finetune_tf = Subset(table_data.train_tf, prev_indices)
            finetune_loader = torch_frame.data.DataLoader(finetune_tf, batch_size=args.batch_size, shuffle=True)
            
            # Fine-tune
            finetune_time = finetune_on_data(
                net, finetune_loader, optimizer_finetune, loss_fn, 
                cpu_device, args.finetune_max_batches, args.finetune_epochs
            )
            
            # Test
            test_loader = torch_frame.data.DataLoader(current_tf, batch_size=args.batch_size, shuffle=False)
            test_start = time.time()
            _, test_pred, test_y = evaluate(net, test_loader, cpu_device, is_regression)
            inference_time = time.time() - test_start
            
            if len(test_y) == 0:
                print(f"  Group {i}: Empty test data, skipping")
                continue
            
            test_metric = evaluate_func(test_y, test_pred)
            total_time = finetune_time + inference_time
            
            print(f"  Group {i} ({','.join(current_group)}): "
                  f"Test {metric_name}={test_metric:.4f}, "
                  f"Time={total_time:.2f}s (finetune={finetune_time:.2f}s + infer={inference_time:.2f}s)")
            
            results.append({
                'test_version_index': i,
                'test_dates': ','.join(current_group),
                'test_samples': len(current_indices),
                'test_metric': test_metric,
                'finetune_time_seconds': finetune_time,
                'inference_time_seconds': inference_time,
                'total_time_seconds': total_time,
                'finetune_group': ','.join(prev_group),
                'finetune_samples': len(prev_indices),
            })
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    dataset_name = Path(data_dir).name
    train_dates = all_groups[0]
    
    if len(train_dates) == 1:
        train_version_str = train_dates[0].replace('-', '')[-4:]
    else:
        first_date = train_dates[0].replace('-', '')[-4:]
        last_date = train_dates[-1].replace('-', '')[-4:]
        train_version_str = f"{first_date}-{last_date}"
    
    model_name = args.model_name or f"{dataset_name}_{args.model}_finetune"
    
    # Add metadata to all results
    for result in results:
        result.update({
            'dataset': dataset_name,
            'model_name': model_name,
            'model': args.model,
            'channels': args.channels,
            'num_layers': args.num_layers,
            'initial_train_dates': ','.join(train_dates),
            'initial_train_samples': len(train_idx),
            'val_samples': len(val_idx),
            'best_epoch': best_epoch,
            'val_metric': best_val_metric,
            'finetune_epochs': args.finetune_epochs,
            'finetune_max_batches': args.finetune_max_batches,
            'metric_name': metric_name,
            'seed': args.seed,
        })
    
    # Save to CSV
    csv_filename = f"{dataset_name}_train{train_version_str}_finetune_results.csv"
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

