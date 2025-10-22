#!/usr/bin/env python3
"""
P3 Baseline: Dynamic Model Ensembling (Retrain + EMA)

Strategy:
- Curr model: RETRAIN from scratch on PREVIOUS group V_{i-1} (full training, no freezing)
- EMA model: Exponential moving average of historical Curr weights
- Dynamic blending: α_t selected via validation set grid search
- Test on NEXT: Predict V_i using ensemble

Key differences from P2 (Fine-tune):
- P2: Fine-tune (partial epochs, smaller LR) → Single model
- P3: Retrain (full epochs, from scratch) → Ensemble (Curr + EMA)

Training flow:
  t=1: Curr retrained on V_0       → EMA initialized → Test on V_1
  t=2: Curr retrained on V_1       → EMA updated     → Test on V_2
  t=3: Curr retrained on V_2       → EMA updated     → Test on V_3
  
Training config (same as P1):
  - Batch size: 256
  - Max epochs: 200
  - Max batches/epoch: 20
  - Early stopping: patience=15
  - Learning rate: 0.001
  - Device: CPU (both training and testing)
  
After each Curr retraining, update EMA: θ_EMA ← β*θ_EMA + (1-β)*θ_Curr
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

from model.base import construct_stype_encoder_dict, default_stype_encoder_cls_kwargs
from utils.data import TableData


def safe_roc_auc(y_true, y_score):
    """Safe AUC that handles single-class edge cases"""
    y_true = np.asarray(y_true)
    if y_true.ndim != 1:
        y_true = y_true.reshape(-1)
    
    # Single-class guard
    if len(np.unique(y_true)) < 2:
        return np.nan
    
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan


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
    
    if isinstance(dates, str):
        dates = [dates]
    
    mask = train_df['timestamp'].isin(dates)
    indices = train_df[mask].index.tolist()
    
    return indices


def reset_model_weights(model):
    """Reset model weights to random initialization (recursively)"""
    def _init(m):
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
    model.apply(_init)


def update_ema(ema_model, curr_model, beta=0.9):
    """Update EMA model weights: θ_EMA ← β*θ_EMA + (1-β)*θ_Curr"""
    with torch.no_grad():
        for ema_param, curr_param in zip(ema_model.parameters(), curr_model.parameters()):
            ema_param.data.mul_(beta).add_(curr_param.data, alpha=1-beta)


def evaluate_single(model, loader, device, is_regression):
    """Evaluate a single model and return predictions"""
    model.eval()
    pred_list = []
    y_list = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            
            # Robust dimension handling
            if pred.dim() == 2 and pred.size(1) == 1:
                pred = pred.squeeze(1)
            elif pred.dim() > 1:
                pred = pred.reshape(pred.size(0), -1).squeeze(1)
            
            pred_list.append(pred.cpu())
            y_list.append(batch.y.cpu())
    
    if len(pred_list) == 0:
        return np.array([]), np.array([])
    
    pred_tensor = torch.cat(pred_list, dim=0)
    y_tensor = torch.cat(y_list, dim=0)
    
    # Apply sigmoid for classification (after concatenation)
    if not is_regression:
        pred_tensor = torch.sigmoid(pred_tensor)
    
    return pred_tensor.numpy(), y_tensor.numpy()


def evaluate_blend(curr_model, ema_model, loader, device, is_regression, alpha):
    """
    Evaluate blended prediction: α*Curr + (1-α)*EMA
    Returns: pred_array, y_array
    """
    # Get predictions from both models
    curr_pred, curr_y = evaluate_single(curr_model, loader, device, is_regression)
    ema_pred, ema_y = evaluate_single(ema_model, loader, device, is_regression)
    
    if len(curr_pred) == 0:
        return np.array([]), np.array([])
    
    # Blend predictions (already sigmoidized for classification)
    blended_pred = alpha * curr_pred + (1 - alpha) * ema_pred
    
    return blended_pred, curr_y


def train_curr_model(model, train_loader, val_loader, optimizer, loss_fn, device, 
                     is_regression, evaluate_func, higher_is_better,
                     max_epochs=200, max_batches=20, early_stop_patience=15):
    """
    Retrain Curr model from scratch (full training like P1)
    Returns: training_time, best_epoch, best_val_metric
    """
    model.train()
    start_time = time.time()
    
    best_val_metric = -float('inf') if higher_is_better else float('inf')
    best_epoch = 0
    best_state = None
    patience_counter = 0
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= max_batches:
                break
            
            batch = batch.to(device)
            optimizer.zero_grad()
            
            pred = model(batch)
            
            # Robust dimension handling
            if pred.dim() == 2 and pred.size(1) == 1:
                pred = pred.squeeze(1)
            elif pred.dim() > 1:
                pred = pred.reshape(pred.size(0), -1).squeeze(1)
            
            loss = loss_fn(pred, batch.y.float())
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        # Validation
        model.eval()
        val_preds = []
        val_ys = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                
                # Robust dimension handling
                if pred.dim() == 2 and pred.size(1) == 1:
                    pred = pred.squeeze(1)
                elif pred.dim() > 1:
                    pred = pred.reshape(pred.size(0), -1).squeeze(1)
                
                if is_regression:
                    val_preds.append(pred.cpu())
                else:
                    val_preds.append(torch.sigmoid(pred).cpu())
                
                val_ys.append(batch.y.cpu())
        
        if len(val_preds) > 0:
            val_pred_tensor = torch.cat(val_preds, dim=0).numpy()
            val_y_tensor = torch.cat(val_ys, dim=0).numpy()
            val_metric = evaluate_func(val_y_tensor, val_pred_tensor)
            
            # Handle NaN from safe_roc_auc
            if np.isnan(val_metric):
                val_metric = best_val_metric  # Keep previous best
        else:
            val_metric = 0.0
        
        # Check if improved
        is_better = (val_metric > best_val_metric) if higher_is_better else (val_metric < best_val_metric)
        
        if is_better:
            best_val_metric = val_metric
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            break
    
    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    
    train_time = time.time() - start_time
    return train_time, best_epoch, best_val_metric


def main():
    parser = argparse.ArgumentParser(description="P3: Curr+EMA Ensemble Baseline")
    
    # Config
    parser.add_argument("--dataset_id", type=str, required=True)
    parser.add_argument("--config_file", type=str, default="./q_zero_config.json")
    
    # Model
    parser.add_argument("--model", type=str, choices=["MLP", "ResNet", "FTTransformer"], default="ResNet")
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout_prob", type=float, default=0.2)
    parser.add_argument("--normalization", type=str, default="layer_norm")
    
    # Training (same as P1/Static)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=200, help="Max epochs for retraining")
    parser.add_argument("--max_batches_per_epoch", type=int, default=20)
    parser.add_argument("--early_stop_patience", type=int, default=15)
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    
    # EMA
    parser.add_argument("--ema_beta", type=float, default=0.9, help="EMA decay rate")
    parser.add_argument("--alpha_grid", type=str, default="0.25,0.5,0.75", 
                        help="Grid for blending weight search")
    
    # Device
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./result_raw_from_server/ensemble")
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
        print(f"⚠️  Need at least 2 groups")
        return
    
    # Setup
    set_seed(args.seed)
    device = torch.device(args.device)
    alpha_grid = [float(x) for x in args.alpha_grid.split(',')]
    
    print("=" * 80)
    print("P3: Dynamic Model Ensembling (Curr + EMA)")
    print("=" * 80)
    print(f"📋 Dataset: {args.dataset_id}")
    print(f"🎯 Task: {task_type_str}")
    print(f"🎲 Seed: {args.seed}")
    print(f"🖥️  Device: {device}")
    print(f"\n📅 Groups: {len(all_groups)}")
    print(f"  Max epochs: {args.num_epochs}")
    print(f"  Early stop patience: {args.early_stop_patience}")
    print(f"  EMA beta: {args.ema_beta}")
    print(f"  Alpha grid: {alpha_grid}")
    
    # Load data
    table_data = TableData.load_from_dir(data_dir)
    if not table_data.is_materialize:
        from utils.resource import get_text_embedder_cfg
        text_cfg = get_text_embedder_cfg(device="cpu")
        table_data.materilize(col_to_text_embedder_cfg=text_cfg)
    
    # Task setup
    if table_data.task_type == TaskType.REGRESSION:
        loss_fn = L1Loss()
        evaluate_func = mean_absolute_error
        higher_is_better = False
        is_regression = True
        metric_name = "MAE"
    else:
        loss_fn = BCEWithLogitsLoss()
        evaluate_func = safe_roc_auc  # Use safe version
        higher_is_better = True
        is_regression = False
        metric_name = "AUC"
    
    # Build model (same style as q_zero_static.py)
    stype_encoder_dict = construct_stype_encoder_dict(default_stype_encoder_cls_kwargs)
    
    # Model arguments dict (consistent style)
    out_channels = 1
    model_args = {
        "channels": args.channels,
        "out_channels": out_channels,
        "num_layers": args.num_layers,
        "col_names_dict": table_data.col_names_dict,
        "stype_encoder_dict": stype_encoder_dict,
        "col_stats": table_data.col_stats,  # Don't move to device, let model handle it
    }
    
    # Add dropout/norm only for ResNet/MLP (not FTTransformer)
    if args.model.lower() in ["resnet", "mlp"]:
        model_args.update({
            "dropout_prob": args.dropout_prob,
            "normalization": args.normalization,
        })
    
    # Create Curr model
    if args.model == "ResNet":
        curr_model = ResNet(**model_args).to(device)
    elif args.model == "FTTransformer":
        curr_model = FTTransformer(**model_args).to(device)
    else:  # MLP
        curr_model = MLP(**model_args).to(device)
    
    # Clone for EMA (will be updated later)
    ema_model = None
    
    # Model name
    if args.model_name is None:
        args.model_name = f"{args.model}-Medium-Ensemble"
    
    print(f"\n🏗️  Model: {args.model}")
    print(f"  Channels: {args.channels}, Layers: {args.num_layers}")
    
    # Results storage
    results = []
    
    # Process each group (start from group 1, use group 0...i-1 as history)
    print(f"\n📊 Processing {len(all_groups)} groups...")
    print("=" * 80)
    
    for i in range(1, len(all_groups)):
        group_start = time.time()
        
        # Training data: PREVIOUS group only (V_{i-1})
        train_dates = all_groups[i-1]
        train_all_indices = get_temporal_indices(data_dir, train_dates)
        
        # Current test group (V_i)
        test_dates = all_groups[i]
        test_indices = get_temporal_indices(data_dir, test_dates)
        
        # Split train data into train/val
        n_train_all = len(train_all_indices)
        n_train = int(n_train_all * (1 - args.val_split))
        
        # Create temporal subset first, then split into train/val
        temporal_train_tf = Subset(table_data.train_tf, train_all_indices)
        train_idx_local = list(range(n_train))
        val_idx_local = list(range(n_train, n_train_all))
        
        train_subset = Subset(temporal_train_tf, train_idx_local)
        val_subset = Subset(temporal_train_tf, val_idx_local)
        
        train_loader = torch_frame.data.DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        val_loader = torch_frame.data.DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        
        print(f"\n  Group {i}: Test on ({','.join(test_dates[:3])}{'...' if len(test_dates)>3 else ''}): {len(test_indices)} samples")
        print(f"    Train on previous group {i-1} ({','.join(train_dates[:3])}{'...' if len(train_dates)>3 else ''}): {n_train_all} samples")
        print(f"      Train: {len(train_idx_local)}, Val: {len(val_idx_local)}")
        
        # Reset Curr model weights (retrain from scratch!)
        reset_model_weights(curr_model)
        
        # Setup optimizer
        optimizer = torch.optim.Adam(curr_model.parameters(), lr=args.lr)
        
        # Retrain Curr model from scratch
        train_time, best_epoch, best_val_metric_train = train_curr_model(
            curr_model, train_loader, val_loader, optimizer, loss_fn, 
            device, is_regression, evaluate_func, higher_is_better,
            args.num_epochs, args.max_batches_per_epoch, args.early_stop_patience
        )
        
        print(f"    ✅ Curr retrained: {train_time:.2f}s (best epoch={best_epoch}, val {metric_name}={best_val_metric_train:.4f})")
        
        # Update EMA model
        if ema_model is None:
            # First group: initialize EMA by creating new model and copying weights
            if args.model == "ResNet":
                ema_model = ResNet(**model_args).to(device)
            elif args.model == "FTTransformer":
                ema_model = FTTransformer(**model_args).to(device)
            else:  # MLP
                ema_model = MLP(**model_args).to(device)
            
            # Copy weights from Curr to EMA using state_dict
            ema_model.load_state_dict(curr_model.state_dict())
            ema_model.eval()
            print(f"    ✅ EMA initialized (copy of Curr via state_dict)")
        else:
            # Update EMA weights
            update_ema(ema_model, curr_model, args.ema_beta)
            print(f"    ✅ EMA updated (β={args.ema_beta})")
        
        # Grid search for best alpha on validation set
        best_alpha = None
        best_val_metric_blend = -float('inf') if higher_is_better else float('inf')
        
        for alpha in alpha_grid:
            val_pred, val_y = evaluate_blend(
                curr_model, ema_model, val_loader, device, is_regression, alpha
            )
            
            if len(val_y) == 0:
                continue
            
            val_metric = evaluate_func(val_y, val_pred)
            
            # Skip NaN results
            if np.isnan(val_metric):
                continue
            
            is_better = (val_metric > best_val_metric_blend) if higher_is_better else (val_metric < best_val_metric_blend)
            if is_better:
                best_val_metric_blend = val_metric
                best_alpha = alpha
        
        # Fallback if all alphas yielded NaN
        if best_alpha is None:
            best_alpha = 0.5
            best_val_metric_blend = np.nan
        
        print(f"    ✅ Best α={best_alpha:.2f} (blend val {metric_name}={best_val_metric_blend:.4f})")
        
        # Test on FUTURE group (V_i) with best alpha
        temporal_test_tf = Subset(table_data.train_tf, test_indices)
        test_loader = torch_frame.data.DataLoader(temporal_test_tf, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        
        inference_start = time.time()
        test_pred, test_y = evaluate_blend(
            curr_model, ema_model, test_loader, device, is_regression, best_alpha
        )
        inference_time = time.time() - inference_start
        
        test_metric = evaluate_func(test_y, test_pred) if len(test_y) > 0 else 0.0
        
        total_time = time.time() - group_start
        
        print(f"    Test {metric_name}={test_metric:.4f}, Time={total_time:.2f}s (train={train_time:.2f}s, infer={inference_time:.2f}s)")
        
        # Store results
        results.append({
            'test_group_index': i,
            'train_group_index': i-1,
            'test_dates': ','.join(test_dates),
            'train_dates': ','.join(train_dates),
            'n_test_samples': len(test_indices),
            'n_train_all': n_train_all,
            'n_train': len(train_idx_local),
            'n_val': len(val_idx_local),
            'test_metric': test_metric,
            'best_alpha': best_alpha,
            'best_epoch': best_epoch,
            'train_val_metric': best_val_metric_train,
            'blend_val_metric': best_val_metric_blend,
            'retrain_time_seconds': train_time,
            'inference_time_seconds': inference_time,
            'total_time_seconds': total_time,
            'dataset': args.dataset_id,
            'model_name': args.model_name,
            'model': args.model,
            'channels': args.channels,
            'num_layers': args.num_layers,
            'num_epochs': args.num_epochs,
            'early_stop_patience': args.early_stop_patience,
            'ema_beta': args.ema_beta,
            'metric_name': metric_name,
            'seed': args.seed
        })
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_date_str = all_groups[0][0].replace('-', '')[:4]
    result_file = os.path.join(
        args.output_dir,
        f"{args.dataset_id}_train{train_date_str}_ensemble_results.csv"
    )
    
    result_df = pd.DataFrame(results)
    
    # Append to existing file if exists
    if os.path.exists(result_file):
        existing_df = pd.read_csv(result_file)
        result_df = pd.concat([existing_df, result_df], ignore_index=True)
    
    result_df.to_csv(result_file, index=False)
    
    print(f"\n💾 Results saved: {result_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()

