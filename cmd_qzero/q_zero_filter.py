#!/usr/bin/env python3
"""
Q-Zero Filter: Model Selection + Fine-tuning Strategy

Workflow for each test timestamp t (t >= 2):
1. Model Selection: Sample 500 architectures, evaluate on t-1 data (256 samples)
2. Select top-1, 5, 10, 20 + 1 random
3. For each selected model:
   - Train on timestamp 1
   - Fine-tune on timestamp t-1
   - Test on timestamp t
4. Save all results to CSV
"""

import argparse
import json
import os
import time
import copy
import random
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch.nn import L1Loss, BCEWithLogitsLoss
from torch.utils.data import Subset
from sklearn.metrics import mean_absolute_error, roc_auc_score
import torch_frame.data

from model.base import construct_stype_encoder_dict, default_stype_encoder_cls_kwargs
from utils.data import TableData
from q_zero.search_space import QZeroMLP, QZeroResNet
from q_zero.proxies.expressflow import express_flow_score


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
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


def sample_architecture(space_name: str, num_blocks: int, all_channels: List[int]) -> List[int]:
    """
    Sample an architecture from search space
    
    Args:
        space_name: 'mlp' or 'resnet'
        num_blocks: number of blocks/layers
        all_channels: list of possible channel sizes
    
    Returns:
        List of channel sizes for each block
    """
    # Each block can have different dimension
    return [random.choice(all_channels) for _ in range(num_blocks)]


def architecture_to_str(arch: List[int]) -> str:
    """Convert architecture to string: [128, 256, 128] -> '128-256-128'"""
    return '-'.join(map(str, arch))


def evaluate_architecture_with_proxy(
    arch: List[int],
    space_name: str,
    sample_batch_x: torch.Tensor,
    col_stats: Dict,
    col_names_dict: Dict,
    stype_encoder_dict: Dict,
    out_channels: int,
    device: str,
) -> float:
    """
    Evaluate architecture using zero-cost proxy
    
    Args:
        arch: architecture (list of channel sizes)
        space_name: 'mlp' or 'resnet'
        sample_batch_x: encoded features
            - MLP: [B, channels]
            - ResNet: [B, channels * num_cols]
        ...
    
    Returns:
        proxy score (higher is better)
    """
    num_layers = len(arch) + 1  # arch is hidden dims, num_layers = len(hidden) + 1
    
    # Create model based on space
    if space_name == 'mlp':
        # MLP: sample_batch_x is [B, channels] after mean pooling
        model = QZeroMLP(
            channels=sample_batch_x.shape[1],
            out_channels=out_channels,
            num_layers=num_layers,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
            hidden_dims=arch,  # Custom architecture
            normalization='layer_norm',
            dropout_prob=0.2,
        ).to(device)
        net_for_proxy = model.mlp
    else:  # resnet
        # ResNet: sample_batch_x is [B, channels * num_cols] after flatten
        # Need to compute channels per column
        num_cols = sum(len(v) for v in col_names_dict.values())
        pre_backbone_dim = sample_batch_x.shape[1]
        
        assert pre_backbone_dim % num_cols == 0, \
            f"Encoded dim {pre_backbone_dim} not divisible by num_cols={num_cols}"
        
        channels = pre_backbone_dim // num_cols
        
        model = QZeroResNet(
            channels=channels,  # ← FIXED: use per-column channels, not total
            out_channels=out_channels,
            num_layers=len(arch),  # For ResNet, num_layers is number of blocks
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
            block_widths=arch,  # Custom block widths
            normalization='layer_norm',
            dropout_prob=0.2,
        ).to(device)
        net_for_proxy = model.backbone
    
    # Compute zero-cost proxy score (new API)
    try:
        score, elapsed_time = express_flow_score(
            arch=net_for_proxy,
            batch_data=sample_batch_x,
            device=device,
            use_wo_embedding=False,  # Already encoded, net_for_proxy is just MLP/backbone
            linearize_target=None,   # Linearize entire network
            epsilon=1e-5,
            weight_mode="traj_width",  # Use trajectory + width
            use_fp64=False,
        )
    except Exception as e:
        print(f"  ⚠️  Error computing proxy for arch {arch}: {e}")
        score = -1e10  # Very low score for failed architectures
    
    # Clean up
    del model
    if device.startswith('cuda'):  # ← FIXED: handle 'cuda:0' etc.
        torch.cuda.empty_cache()
    
    return float(score)


def model_selection(
    space_name: str,
    table_data: TableData,
    prev_group_indices: List[int],
    device: str,
    num_samples: int = 500,
    sample_batch_size: int = 256,
) -> Tuple[List[Tuple[List[int], float]], List[int], int]:
    """
    Model selection using zero-cost proxy
    
    Args:
        space_name: 'mlp' or 'resnet'
        table_data: loaded TableData
        prev_group_indices: indices from previous timestamp
        device: 'cuda' or 'cpu'
        num_samples: number of architectures to sample
        sample_batch_size: batch size for proxy evaluation

    Returns:
        ranked_archs: list of (architecture, score) sorted by score
        random_arch: one random architecture for baseline
        encoder_channels: encoder channels used for this dataset
    """
    print(f"\n{'='*80}")
    print(f"🔍 Model Selection: Sampling {num_samples} architectures")
    print(f"{'='*80}")
    
    # Prepare sample batch from previous timestamp
    actual_batch_size = min(sample_batch_size, len(prev_group_indices))
    prev_subset = Subset(table_data.train_tf, prev_group_indices[:actual_batch_size])
    
    # Use torch_frame.data.DataLoader (handles TensorFrame automatically)
    prev_loader = torch_frame.data.DataLoader(
        prev_subset, 
        batch_size=actual_batch_size, 
        shuffle=False,
    )
    
    # Get one batch and encode it
    batch = next(iter(prev_loader)).to(device)
    
    # Get search space choices from model class
    stype_encoder_dict = construct_stype_encoder_dict(default_stype_encoder_cls_kwargs)
    out_channels = 1  # Binary classification or regression
    
    if space_name == 'mlp':
        blocks_choices = QZeroMLP.blocks_choices
        channel_choices = QZeroMLP.channel_choices
    else:
        blocks_choices = QZeroResNet.blocks_choices
        channel_choices = QZeroResNet.channel_choices
    
    # Get number of features in dataset
    num_cols = sum(len(v) for v in table_data.col_names_dict.values())
    
    # Auto-determine encoder channels based on dataset feature count
    # More features → larger encoder channels (to capture dataset complexity)
    # Formula: scale with sqrt(num_cols), clamped to search space range
    import math
    min_channel = min(channel_choices)
    max_channel = max(channel_choices)
    # Scale encoder channels with feature count (but stay within search space)
    encoder_channels = int(min_channel * math.sqrt(num_cols / 10))  # Baseline: 10 features
    encoder_channels = max(min_channel, min(encoder_channels, max_channel))
    
    print(f"   Dataset features: {num_cols}")
    print(f"   Search space: blocks={blocks_choices}, channels={channel_choices}")
    print(f"   Auto encoder_channels={encoder_channels} (scaled by feature count)")
    
    # Create a temporary model to encode features
    if space_name == 'mlp':
        temp_model = QZeroMLP(
            channels=encoder_channels,
            out_channels=out_channels,
            num_layers=2,
            col_stats=table_data.col_stats,
            col_names_dict=table_data.col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
            hidden_dims=[encoder_channels],  # 1 hidden layer (num_layers=2: input→hidden→output)
            normalization='layer_norm',
            dropout_prob=0.2,
        ).to(device)
    else:
        temp_model = QZeroResNet(
            channels=encoder_channels,
            out_channels=out_channels,
            num_layers=2,  # 2 residual blocks
            col_stats=table_data.col_stats,
            col_names_dict=table_data.col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
            block_widths=[encoder_channels, encoder_channels],  # ← FIXED: 2 blocks need 2 widths!
            normalization='layer_norm',
            dropout_prob=0.2,
        ).to(device)
    
    # Encode features
    with torch.no_grad():
        x_encoded, _ = temp_model.encoder(batch)
        if space_name == 'mlp':
            x_encoded = torch.mean(x_encoded, dim=1)  # [B, encoder_channels]
        else:
            x_encoded = x_encoded.view(x_encoded.size(0), -1)  # [B, encoder_channels * num_cols]
    
    # Get actual channels from encoded features (will be used for proxy evaluation)
    actual_channels = x_encoded.shape[1]
    
    del temp_model
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    
    if space_name == 'mlp':
        print(f"✅ Encoded features: {x_encoded.shape}")
        print(f"   MLP uses mean pooling → dimension={actual_channels} (same as encoder_channels)")
    else:
        print(f"✅ Encoded features: {x_encoded.shape}")
        print(f"   ResNet flattens → dimension={actual_channels} = {encoder_channels} × {num_cols} features")
        print(f"   This captures both encoder capacity AND feature count!")
    
    # Sample architectures
    sampled_archs = []
    for _ in range(num_samples):
        num_blocks = random.choice(blocks_choices)
        arch = sample_architecture(space_name, num_blocks, channel_choices)
        sampled_archs.append(arch)
    
    print(f"✅ Sampled {len(sampled_archs)} architectures")
    print(f"   Example: {sampled_archs[:3]}")
    
    # Evaluate each architecture
    arch_scores = []
    for i, arch in enumerate(sampled_archs):
        if (i + 1) % 50 == 0:
            print(f"   Evaluating {i+1}/{num_samples}...")
        
        score = evaluate_architecture_with_proxy(
            arch=arch,
            space_name=space_name,
            sample_batch_x=x_encoded,
            col_stats=table_data.col_stats,
            col_names_dict=table_data.col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
            out_channels=out_channels,
            device=device,
        )
        
        arch_scores.append((arch, score))
    
    # Sort by score (descending)
    ranked_archs = sorted(arch_scores, key=lambda x: x[1], reverse=True)
    
    # Pick one random architecture for baseline
    random_arch = random.choice(sampled_archs)
    
    print(f"\n📊 Ranking Complete!")
    print(f"   Top-1:  {architecture_to_str(ranked_archs[0][0])} (score={ranked_archs[0][1]:.4f})")
    print(f"   Top-5:  {[architecture_to_str(a[0]) for a in ranked_archs[:5]]}")
    print(f"   Random: {architecture_to_str(random_arch)}")
    
    # Return ranked architectures, random architecture, and encoder_channels
    return ranked_archs, random_arch, encoder_channels


def train_model(
    model: torch.nn.Module,
    train_loader: torch_frame.data.DataLoader,
    val_loader: torch_frame.data.DataLoader,
    device: str,
    is_regression: bool,
    num_epochs: int = 200,
    lr: float = 0.001,
    max_batches_per_epoch: int = 20,
    early_stop_patience: int = 10,  # ← FIXED: Same as dnn_baseline_table_data.py
) -> Tuple[torch.nn.Module, float]:
    """
    Train model from scratch
    
    Returns:
        trained_model, training_time
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = L1Loss() if is_regression else BCEWithLogitsLoss()
    
    # ← FIXED: Regression uses inf (smaller is better), Classification uses -inf (larger is better)
    best_val_metric = float('inf') if is_regression else float('-inf')
    best_state = None
    best_epoch = 0
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training
        epoch_loss = 0.0
        batch_count = 0
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= max_batches_per_epoch:
                break
            
            batch = batch.to(device)
            optimizer.zero_grad()
            
            pred = model(batch)
            pred = pred.view(-1) if pred.size(1) == 1 else pred
            
            # ← FIXED: Ensure labels are float for BCEWithLogitsLoss
            loss = loss_fn(pred, batch.y.float())
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        
        # Validation
        if val_loader is not None and (epoch + 1) % 5 == 0:
            model.eval()
            val_preds = []
            val_ys = []
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    pred = model(batch)
                    pred = pred.view(-1) if pred.size(1) == 1 else pred
                    
                    if is_regression:
                        val_preds.append(pred.cpu())
                    else:
                        val_preds.append(torch.sigmoid(pred).cpu())
                    
                    val_ys.append(batch.y.cpu())
            
            if len(val_preds) > 0:
                val_pred = torch.cat(val_preds).numpy()
                val_y = torch.cat(val_ys).numpy()
                
                # ← FIXED: Use positive MAE for regression, directly compare
                if is_regression:
                    val_metric = mean_absolute_error(val_y, val_pred)  # Positive MAE
                    is_better = val_metric < best_val_metric  # Smaller is better
                    metric_name = 'MAE'
                else:
                    if len(np.unique(val_y)) > 1:
                        val_metric = roc_auc_score(val_y, val_pred)
                    else:
                        val_metric = 0.5
                    is_better = val_metric > best_val_metric  # Larger is better
                    metric_name = 'AUC'
                
                if is_better:
                    best_val_metric = val_metric
                    best_state = copy.deepcopy(model.state_dict())
                    best_epoch = epoch + 1
                    patience_counter = 0
                    print(f"      Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f}, val_{metric_name}={val_metric:.4f} ⭐ (best)")
                else:
                    patience_counter += 1
                    print(f"      Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f}, val_{metric_name}={val_metric:.4f} (patience={patience_counter}/{early_stop_patience})")
                
                if patience_counter >= early_stop_patience:
                    print(f"      Early stopping at epoch {epoch+1}")
                    break
            
            model.train()
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"      Restored best model from epoch {best_epoch}")
    
    training_time = time.time() - start_time
    print(f"      Total epochs run: {epoch+1}/{num_epochs}, Total time: {training_time:.2f}s")
    
    return model, training_time


def finetune_model(
    model: torch.nn.Module,
    finetune_loader: torch_frame.data.DataLoader,
    device: str,
    is_regression: bool,
    num_epochs: int = 10,
    lr: float = 0.0001,
    max_batches: int = 20,
) -> Tuple[torch.nn.Module, float]:
    """
    Fine-tune model on new data
    
    Returns:
        finetuned_model, finetuning_time
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = L1Loss() if is_regression else BCEWithLogitsLoss()
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_count = 0
        for batch_idx, batch in enumerate(finetune_loader):
            if batch_idx >= max_batches:
                break
            
            batch = batch.to(device)
            optimizer.zero_grad()
            
            pred = model(batch)
            pred = pred.view(-1) if pred.size(1) == 1 else pred
            
            # ← FIXED: Ensure labels are float for BCEWithLogitsLoss
            loss = loss_fn(pred, batch.y.float())
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        if (epoch + 1) % 5 == 0:
            print(f"      Finetune epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f}")
    
    finetuning_time = time.time() - start_time
    print(f"      Finetune complete: {num_epochs} epochs, {finetuning_time:.2f}s")
    
    return model, finetuning_time


def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch_frame.data.DataLoader,
    device: str,
    is_regression: bool,
) -> Tuple[float, float]:
    """
    Evaluate model on test data
    
    Returns:
        metric (AUC or MAE), inference_time
    """
    model.eval()
    preds = []
    ys = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)
            pred = pred.view(-1) if pred.size(1) == 1 else pred
            
            if is_regression:
                preds.append(pred.cpu())
            else:
                preds.append(torch.sigmoid(pred).cpu())
            
            ys.append(batch.y.cpu())
    
    inference_time = time.time() - start_time
    
    if len(preds) == 0:
        return 0.0, inference_time
    
    pred_array = torch.cat(preds).numpy()
    y_array = torch.cat(ys).numpy()
    
    if is_regression:
        metric = mean_absolute_error(y_array, pred_array)
    else:
        if len(np.unique(y_array)) > 1:
            metric = roc_auc_score(y_array, pred_array)
        else:
            metric = 0.5
    
    return metric, inference_time


def main():
    parser = argparse.ArgumentParser(description='Q-Zero Filter: Model Selection + Fine-tuning')
    parser.add_argument('--dataset_id', type=str, required=True, help='Dataset ID (e.g., avito-ad-ctr)')
    parser.add_argument('--space_name', type=str, required=True, choices=['mlp', 'resnet'], help='Search space')
    parser.add_argument('--config_file', type=str, default='./q_zero_config.json', help='Config file path')
    parser.add_argument('--output_dir', type=str, default='./result_raw_from_server/q_zero_filter', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_samples', type=int, default=500, help='Number of architectures to sample')
    parser.add_argument('--sample_batch_size', type=int, default=8, help='Batch size for proxy evaluation')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    print("=" * 80)
    print(f"🎯 Q-Zero Filter: {args.dataset_id} ({args.space_name})")
    print("=" * 80)
    
    # Load config
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    
    if args.dataset_id not in config:
        raise ValueError(f"Dataset {args.dataset_id} not found in config")
    
    dataset_config = config[args.dataset_id]
    data_dir = dataset_config['data_dir']
    task_type = dataset_config['task_type']
    groups = dataset_config['groups']
    
    is_regression = (task_type == 'regression')
    
    print(f"📊 Dataset: {args.dataset_id}")
    print(f"   Task: {task_type}")
    print(f"   Groups: {len(groups)}")
    print(f"   Device: {args.device}")
    
    # Load data
    print(f"\n📂 Loading data from {data_dir}...")
    table_data = TableData.load_from_dir(data_dir)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare results CSV with full configuration in filename
    import datetime
    import glob
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(
        args.output_dir, 
        f'{args.dataset_id}_{args.space_name}_n{args.num_samples}_b{args.sample_batch_size}_{timestamp}.csv'
    )
    results = []
    
    # Load ALL existing CSVs for this dataset+space (regardless of n/b config)
    # Because: same (dataset, test_group, architecture) → same performance!
    csv_pattern = os.path.join(
        args.output_dir,
        f'{args.dataset_id}_{args.space_name}_*.csv'
    )
    existing_csvs = glob.glob(csv_pattern)
    
    # Combine all existing results for cache lookup
    if len(existing_csvs) > 0:
        print(f"📂 Found {len(existing_csvs)} existing CSV(s) for caching:")
        for csv_file in existing_csvs:
            print(f"   - {os.path.basename(csv_file)}")
        existing_results_list = [pd.read_csv(f) for f in existing_csvs]
        all_existing_results = pd.concat(existing_results_list, ignore_index=True)
        print(f"   Total cached experiments: {len(all_existing_results)}")
    else:
        all_existing_results = None
        print(f"📂 No existing CSVs found, will train all architectures")
    
    print(f"📄 Results will be saved to: {os.path.basename(results_file)}")
    
    # Main loop: for each test timestamp (starting from group 2)
    for test_group_idx in range(2, len(groups)):
        print(f"\n{'='*80}")
        print(f"📅 Test Group {test_group_idx} / {len(groups)-1}")
        print(f"{'='*80}")
        
        # Get indices
        group_0_indices = get_temporal_indices(data_dir, groups[0])
        prev_group_indices = get_temporal_indices(data_dir, groups[test_group_idx - 1])
        test_group_indices = get_temporal_indices(data_dir, groups[test_group_idx])
        
        print(f"   Group 0 (train):       {len(group_0_indices)} samples")
        print(f"   Group {test_group_idx-1} (finetune):    {len(prev_group_indices)} samples")
        print(f"   Group {test_group_idx} (test):        {len(test_group_indices)} samples")
        
        # Step 1: Model Selection on previous group
        ranked_archs, random_arch, encoder_channels = model_selection(
            space_name=args.space_name,
            table_data=table_data,
            prev_group_indices=prev_group_indices,
            device=args.device,
            num_samples=args.num_samples,
            sample_batch_size=args.sample_batch_size,
        )
        
        print(f"\n📝 Using encoder_channels={encoder_channels} for training (from proxy phase)")
        
        # Step 2: Collect all unique architectures to train
        # Top-k overlaps: top-20 includes top-10, top-5, top-1
        # We need to train each unique architecture only once
        
        all_archs_to_train = {}  # arch_str -> (arch, proxy_score, selection_methods)
        
        # Collect top-1
        arch_str = architecture_to_str(ranked_archs[0][0])
        all_archs_to_train[arch_str] = (ranked_archs[0][0], ranked_archs[0][1], ['top1'])
        
        # Collect top-5
        for i in range(min(5, len(ranked_archs))):
            arch, score = ranked_archs[i]
            arch_str = architecture_to_str(arch)
            if arch_str in all_archs_to_train:
                all_archs_to_train[arch_str][2].append('top5')  # Add to selection methods
            else:
                all_archs_to_train[arch_str] = (arch, score, ['top5'])
        
        # Collect top-10
        for i in range(min(10, len(ranked_archs))):
            arch, score = ranked_archs[i]
            arch_str = architecture_to_str(arch)
            if arch_str in all_archs_to_train:
                all_archs_to_train[arch_str][2].append('top10')
            else:
                all_archs_to_train[arch_str] = (arch, score, ['top10'])
        
        # Collect top-20
        for i in range(min(20, len(ranked_archs))):
            arch, score = ranked_archs[i]
            arch_str = architecture_to_str(arch)
            if arch_str in all_archs_to_train:
                all_archs_to_train[arch_str][2].append('top20')
            else:
                all_archs_to_train[arch_str] = (arch, score, ['top20'])
        
        # Add random
        random_arch_str = architecture_to_str(random_arch)
        if random_arch_str in all_archs_to_train:
            all_archs_to_train[random_arch_str][2].append('random')
        else:
            all_archs_to_train[random_arch_str] = (random_arch, 0.0, ['random'])
        
        print(f"\n📊 Unique architectures to train: {len(all_archs_to_train)}")
        print(f"   (Top-20 includes top-10/5/1, so we deduplicate)")
        
        # Step 3: Train, finetune, test each unique architecture
        for arch_idx, (arch_str, (arch, proxy_score, selection_methods)) in enumerate(all_archs_to_train.items()):
            print(f"\n{'─'*80}")
            print(f"🔧 [{arch_idx+1}/{len(all_archs_to_train)}] Architecture: enc={encoder_channels}, arch={arch_str}")
            print(f"   Proxy score: {proxy_score:.4f}")
            print(f"   Selected by: {', '.join(selection_methods)}")
            print(f"{'─'*80}")
            
            # Check if this experiment already exists in CSV (skip if cached)
            cache_key = {
                'dataset': args.dataset_id,
                'space_name': args.space_name,
                'test_group': test_group_idx,
                'architecture': arch_str,
            }
            
            # Load existing results to check cache (from all historical CSVs)
            if all_existing_results is not None:
                is_cached = (
                    (all_existing_results['dataset'] == cache_key['dataset']) &
                    (all_existing_results['space_name'] == cache_key['space_name']) &
                    (all_existing_results['test_group'] == cache_key['test_group']) &
                    (all_existing_results['architecture'] == cache_key['architecture'])
                ).any()
                
                if is_cached:
                    print(f"  ⏭️  Skipping training (found in cache)")
                    
                    # Get cached results
                    cached_row = all_existing_results[
                        (all_existing_results['dataset'] == cache_key['dataset']) &
                        (all_existing_results['space_name'] == cache_key['space_name']) &
                        (all_existing_results['test_group'] == cache_key['test_group']) &
                        (all_existing_results['architecture'] == cache_key['architecture'])
                    ].iloc[0]
                    
                    # Add results for each selection method this architecture belongs to
                    for method in selection_methods:
                        metric_col = 'mae' if is_regression else 'auc'
                        results.append({
                            'dataset': args.dataset_id,
                            'space_name': args.space_name,
                            'test_group': test_group_idx,
                            'selection_method': method,
                            'arch_idx': arch_idx,
                            'num_blocks': len(arch),
                            'encoder_channels': encoder_channels,  # Record encoder channels
                            'architecture': arch_str,
                            'proxy_score': proxy_score,
                            'train_time': cached_row['train_time'],
                            'finetune_time': cached_row['finetune_time'],
                            'inference_time': cached_row['inference_time'],
                            'total_time': cached_row['total_time'],
                            metric_col: cached_row[metric_col],
                        })
                    continue
            
            # Train this architecture (not cached)
            print(f"  🚀 Training new architecture...")
            
            # Create model using encoder_channels from proxy phase
            stype_encoder_dict = construct_stype_encoder_dict(default_stype_encoder_cls_kwargs)
            out_channels = 1
            
            # Use same encoder_channels as proxy phase (consistent!)
            if args.space_name == 'mlp':
                model = QZeroMLP(
                    channels=encoder_channels,  # Same as proxy phase
                    out_channels=out_channels,
                    num_layers=len(arch) + 1,
                    col_stats=table_data.col_stats,
                    col_names_dict=table_data.col_names_dict,
                    stype_encoder_dict=stype_encoder_dict,
                    hidden_dims=arch,
                    normalization='layer_norm',
                    dropout_prob=0.2,
                ).to(args.device)
            else:  # resnet
                model = QZeroResNet(
                    channels=encoder_channels,  # Same as proxy phase
                    out_channels=out_channels,
                    num_layers=len(arch),
                    col_stats=table_data.col_stats,
                    col_names_dict=table_data.col_names_dict,
                    stype_encoder_dict=stype_encoder_dict,
                    block_widths=arch,
                    normalization='layer_norm',
                    dropout_prob=0.2,
                ).to(args.device)
            
            # 3a. Train on group 0
            train_subset = Subset(table_data.train_tf, group_0_indices)
            train_size = int(0.9 * len(train_subset))
            val_size = len(train_subset) - train_size
            train_subset_split, val_subset_split = torch.utils.data.random_split(
                train_subset, [train_size, val_size]
            )
            
            train_loader = torch_frame.data.DataLoader(train_subset_split, batch_size=256, shuffle=True)
            val_loader = torch_frame.data.DataLoader(val_subset_split, batch_size=256, shuffle=False)
            
            print(f"    Training on group 0...")
            model, train_time = train_model(
                model, train_loader, val_loader, args.device, is_regression
            )
            print(f"    ✅ Training complete: {train_time:.2f}s")
            
            # 3b. Fine-tune on previous group
            finetune_subset = Subset(table_data.train_tf, prev_group_indices)
            finetune_loader = torch_frame.data.DataLoader(finetune_subset, batch_size=256, shuffle=True)
            
            print(f"    Fine-tuning on group {test_group_idx-1}...")
            model, finetune_time = finetune_model(
                model, finetune_loader, args.device, is_regression
            )
            print(f"    ✅ Fine-tuning complete: {finetune_time:.2f}s")
            
            # 3c. Test on current group
            test_subset = Subset(table_data.train_tf, test_group_indices)
            test_loader = torch_frame.data.DataLoader(test_subset, batch_size=256, shuffle=False)
            
            print(f"    Testing on group {test_group_idx}...")
            metric, inference_time = evaluate_model(
                model, test_loader, args.device, is_regression
            )
            
            metric_name = 'MAE' if is_regression else 'AUC'
            print(f"    ✅ Testing complete: {metric_name}={metric:.4f}, time={inference_time:.2f}s")
            
            # Save result for each selection method
            for method in selection_methods:
                results.append({
                    'dataset': args.dataset_id,
                    'space_name': args.space_name,
                    'test_group': test_group_idx,
                    'selection_method': method,
                    'arch_idx': arch_idx,
                    'num_blocks': len(arch),
                    'encoder_channels': encoder_channels,  # Record encoder channels
                    'architecture': arch_str,
                    'proxy_score': proxy_score,
                    'train_time': train_time,
                    'finetune_time': finetune_time,
                    'inference_time': inference_time,
                    'total_time': train_time + finetune_time + inference_time,
                    metric_name.lower(): metric,
                })
            
            # Clean up
            del model
            if args.device.startswith('cuda'):
                torch.cuda.empty_cache()
        
        # Save results after each test group
        df = pd.DataFrame(results)
        df.to_csv(results_file, index=False)
        print(f"\n💾 Results saved to: {results_file}")
    
    print(f"\n{'='*80}")
    print(f"🎉 All experiments complete!")
    print(f"{'='*80}")
    print(f"📊 Total results: {len(results)}")
    print(f"📁 Saved to: {results_file}")


if __name__ == "__main__":
    main()


"""
# 🚀 最简单的调试命令（10个架构，CPU，约2-3分钟）
python q_zero_filter.py --dataset_id avito-ad-ctr --space_name mlp --config_file ./q_zero_config.json --device cpu --num_samples 5 --sample_batch_size 32 --seed 42
"""