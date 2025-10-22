#!/usr/bin/env python3
"""
Train static models on specific timestamps and ensemble predictions.

Strategy:
1. Train 3 ResNet models (small/medium/large) on timestamps 3, 5, 7
2. Each model trained on ONE timestamp (randomly assigned)
3. Use first 1000 samples from timestamps 4, 6, 9 for validation
4. Test on the LAST timestamp
5. Ensemble using:
   - Average Ensemble
   - Bayesian Ensemble
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_frame
from torch.nn import L1Loss, BCEWithLogitsLoss
from torch.utils.data import Subset
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from relbench.base import TaskType

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.base import construct_stype_encoder_dict, default_stype_encoder_cls_kwargs
from utils.data import TableData
from q_zero.search_space.resnet import QZeroResNet


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
        return None
    
    preds = torch.cat(preds).numpy()
    ys = torch.cat(ys).numpy()
    
    if is_regression:
        metric = mean_absolute_error(ys, preds)
    else:
        metric = roc_auc_score(ys, preds)
    
    return metric, preds, ys


def train_one_epoch(model, loader, optimizer, criterion, device, max_round_epoch=20):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for idx, batch in enumerate(loader):
        # Limit batches per epoch
        if idx >= max_round_epoch:
            break
        
        batch = batch.to(device)
        optimizer.zero_grad()
        
        pred = model(batch)
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        
        # Convert target to float for loss computation
        target = batch.y.float()
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def create_resnet_model(architecture, encoder_channels, num_cols, out_channels, 
                        col_stats, col_names_dict, stype_encoder_dict):
    """
    Create a ResNet model with given architecture.
    
    Args:
        architecture: String like "2-32" or "3-128-128-64"
        encoder_channels: Encoder output channels
        num_cols: Number of features
        out_channels: Output dimension (1 for binary/regression)
        col_stats, col_names_dict, stype_encoder_dict: TensorFrame configs
    
    Returns:
        QZeroResNet model
    """
    parts = architecture.split('-')
    num_blocks = int(parts[0])
    block_channels = [int(x) for x in parts[1:]]
    
    if len(block_channels) != num_blocks:
        raise ValueError(f"Architecture {architecture} has {len(block_channels)} channels but {num_blocks} blocks")
    
    model = QZeroResNet(
        channels=encoder_channels,
        out_channels=out_channels,
        num_layers=num_blocks,
        col_stats=col_stats,
        col_names_dict=col_names_dict,
        stype_encoder_dict=stype_encoder_dict,
        block_widths=block_channels
    )
    
    return model


def train_static_model(dataset_name, data_dir, task_type_str, train_timestamp, val_timestamps, 
                       architecture, device, max_epochs=200, patience=10):
    """
    Train a single static model on one timestamp.
    
    Args:
        dataset_name: Dataset name
        data_dir: Data directory
        task_type_str: Task type ('classification' or 'regression')
        train_timestamp: Single timestamp to train on (e.g., 3)
        val_timestamps: List of timestamps for validation (e.g., [4, 6, 9])
        architecture: ResNet architecture (e.g., "2-32")
        device: torch device
        max_epochs: Maximum training epochs
        patience: Early stopping patience
    
    Returns:
        Trained model, validation metric
    """
    print(f"\n{'='*80}")
    print(f"Training {architecture} on timestamp {train_timestamp}")
    print(f"{'='*80}")
    
    # Load data
    table_data = TableData.load_from_dir(data_dir)
    
    # Materialize if needed
    if not table_data.is_materialize:
        from utils.resource import get_text_embedder_cfg
        text_cfg = get_text_embedder_cfg(device="cpu")
        table_data.materilize(col_to_text_embedder_cfg=text_cfg)
    
    is_regression = (task_type_str == 'regression')
    
    # Get available timestamps
    all_timestamps = sorted(table_data.train_df['timestamp'].unique())
    
    # Get train indices
    train_date = all_timestamps[train_timestamp]
    train_indices = get_temporal_indices(data_dir, train_date)
    
    # Get validation indices (first 500 samples from each val timestamp)
    val_indices = []
    for val_ts in val_timestamps:
        if val_ts < len(all_timestamps):
            val_date = all_timestamps[val_ts]
            val_ts_indices = get_temporal_indices(data_dir, val_date)
            val_indices.extend(val_ts_indices[:500])  # Only first 500
    
    print(f"  Train samples: {len(train_indices)} (timestamp {train_timestamp})")
    print(f"  Val samples: {len(val_indices)} (from timestamps {val_timestamps}, 500 each)")
    
    # Create TensorFrame datasets
    train_subset = Subset(table_data.train_tf, train_indices)
    val_subset = Subset(table_data.train_tf, val_indices)
    
    # Create loaders
    train_loader = torch_frame.data.DataLoader(train_subset, batch_size=128, shuffle=True)
    val_loader = torch_frame.data.DataLoader(val_subset, batch_size=512, shuffle=False)
    
    # Create encoder dict
    num_cols = sum([len(names) for names in table_data.col_names_dict.values()])
    encoder_channels = min(256, max(32, int(32 * np.sqrt(num_cols / 10))))
    
    stype_encoder_dict = construct_stype_encoder_dict(default_stype_encoder_cls_kwargs)
    
    # Create model
    out_channels = 1
    model = create_resnet_model(
        architecture=architecture,
        encoder_channels=encoder_channels,
        num_cols=num_cols,
        out_channels=out_channels,
        col_stats=table_data.col_stats,
        col_names_dict=table_data.col_names_dict,
        stype_encoder_dict=stype_encoder_dict
    )
    model = model.to(device)
    
    print(f"  Model: {architecture}, Encoder channels: {encoder_channels}")
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = L1Loss() if is_regression else BCEWithLogitsLoss()
    
    # Training loop
    best_val_metric = None
    patience_counter = 0
    
    for epoch in range(max_epochs):
        # Train (limit to 20 batches per epoch)
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, max_round_epoch=20)
        
        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0:
            val_metric, _, _ = evaluate(model, val_loader, device, is_regression)
            
            # Check improvement
            improved = False
            if best_val_metric is None:
                best_val_metric = val_metric
                improved = True
            else:
                if is_regression:
                    if val_metric < best_val_metric:
                        best_val_metric = val_metric
                        improved = True
                else:
                    if val_metric > best_val_metric:
                        best_val_metric = val_metric
                        improved = True
            
            if improved:
                patience_counter = 0
                print(f"  Epoch {epoch+1:3d}: Loss={train_loss:.4f}, Val={'MAE' if is_regression else 'AUC'}={val_metric:.4f} ✨")
            else:
                patience_counter += 1
                if (epoch + 1) % 20 == 0:
                    print(f"  Epoch {epoch+1:3d}: Loss={train_loss:.4f}, Val={'MAE' if is_regression else 'AUC'}={val_metric:.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    return model, best_val_metric


def extract_features_from_loader(loader, device):
    """
    Extract raw features from TensorFrame loader for K-NN indexing
    
    Returns:
        features: numpy array (N x D)
        labels: numpy array (N,)
    """
    all_features = []
    all_labels = []
    
    for batch in loader:
        batch = batch.to(device)
        
        # Concatenate all feature types into a single vector
        feat_list = []
        for stype, feat_tensor in batch.feat_dict.items():
            # Handle different types of tensors
            try:
                # Try direct conversion first
                if torch.is_tensor(feat_tensor):
                    # Handle sparse tensors
                    if feat_tensor.is_sparse:
                        feat_np = feat_tensor.to_dense().cpu().numpy()
                    else:
                        feat_np = feat_tensor.cpu().numpy()
                # Handle TensorFrame special types
                elif hasattr(feat_tensor, 'values'):
                    # Check if values is callable (a method) or a tensor
                    if callable(feat_tensor.values):
                        try:
                            feat_val = feat_tensor.values()
                            if feat_val.is_sparse:
                                feat_np = feat_val.to_dense().cpu().numpy()
                            else:
                                feat_np = feat_val.cpu().numpy()
                        except RuntimeError:
                            # If values() fails, try direct access
                            feat_np = feat_tensor.cpu().numpy()
                    else:
                        if feat_tensor.values.is_sparse:
                            feat_np = feat_tensor.values.to_dense().cpu().numpy()
                        else:
                            feat_np = feat_tensor.values.cpu().numpy()
                else:
                    # Fallback
                    feat_np = feat_tensor.cpu().numpy()
                
                # Flatten to 2D
                if feat_np.ndim == 1:
                    feat_np = feat_np.reshape(-1, 1)
                elif feat_np.ndim == 3:
                    N, num_cols, emb_dim = feat_np.shape
                    feat_np = feat_np.reshape(N, num_cols * emb_dim)
                
                feat_list.append(feat_np)
                
            except Exception as e:
                # Skip problematic features
                print(f"    Warning: Skipping feature type {stype}: {e}")
                continue
        
        if len(feat_list) > 0:
            batch_features = np.concatenate(feat_list, axis=1)
            all_features.append(batch_features)
        
        all_labels.append(batch.y.cpu().numpy())
    
    if len(all_features) == 0:
        return np.array([]), np.array([])
    
    return np.vstack(all_features), np.concatenate(all_labels)


def build_knn_index(train_loaders, device, k_neighbors=20):
    """
    Build K-NN index from all training data
    
    Args:
        train_loaders: List of DataLoaders, one per model
        device: torch device
        k_neighbors: number of neighbors
    
    Returns:
        nn_model: fitted NearestNeighbors
        model_ids: which model each sample belongs to
    """
    print("\n" + "="*80)
    print("📊 Building K-NN Index for Bayesian Ensemble")
    print("="*80)
    
    X_all = []
    model_ids = []
    
    for model_idx, loader in enumerate(train_loaders):
        print(f"  Extracting features from model {model_idx+1} training data...")
        X_train, _ = extract_features_from_loader(loader, device)
        
        if len(X_train) > 0:
            X_all.append(X_train)
            model_ids.extend([model_idx] * len(X_train))
            print(f"    Model {model_idx+1}: {len(X_train)} samples × {X_train.shape[1]} features")
    
    if len(X_all) == 0:
        print("  ⚠️  No training data found!")
        return None, None
    
    X_all = np.vstack(X_all)
    model_ids = np.array(model_ids)
    
    print(f"\n  Total indexed: {len(X_all)} samples")
    print(f"  Feature dimension: {X_all.shape[1]}")
    
    # Handle NaN values (replace with 0)
    if np.isnan(X_all).any():
        nan_count = np.isnan(X_all).sum()
        print(f"  ⚠️  Found {nan_count} NaN values, replacing with 0...")
        X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"  Building K-NN model (k={k_neighbors})...")
    
    nn_model = NearestNeighbors(n_neighbors=min(k_neighbors, len(X_all)), 
                                algorithm='auto', metric='euclidean')
    nn_model.fit(X_all)
    
    print("  ✅ K-NN index built!")
    print("="*80)
    
    return nn_model, model_ids


def ensemble_average(predictions_list):
    """Simple average ensemble"""
    return np.mean(predictions_list, axis=0)


def ensemble_bayesian(predictions_list, val_metrics, train_timestamps, test_timestamp, is_regression):
    """
    Temporal-Weighted ensemble (simplified Bayesian-style weighting)
    
    Weight = performance_weight × temporal_weight
    
    Performance weight:
      For AUC: exp(auc - 1) → higher AUC = higher weight
      For MAE: exp(-mae) → lower MAE = higher weight
    
    Temporal weight:
      exp(-distance / 2) → closer to test timestamp = higher weight
      Example: train_ts=7, test_ts=9 → distance=2 → weight=exp(-1)=0.368
    """
    weights = []
    for val_metric, train_ts in zip(val_metrics, train_timestamps):
        # Performance-based weight
        if is_regression:
            # MAE: lower is better
            perf_weight = np.exp(-val_metric)
        else:
            # AUC: higher is better, normalize to [0, 1]
            perf_weight = np.exp(val_metric - 1.0)
        
        # Temporal distance weight (closer = higher weight)
        distance = abs(test_timestamp - train_ts)
        temporal_weight = np.exp(-distance / 2.0)  # Decay factor = 2.0
        
        # Combined weight
        combined_weight = perf_weight * temporal_weight
        weights.append(combined_weight)
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    # Weighted average
    weighted_pred = np.zeros_like(predictions_list[0])
    for i, pred in enumerate(predictions_list):
        weighted_pred += weights[i] * pred
    
    return weighted_pred, weights


def ensemble_knn_bayesian(
    test_features,
    predictions_list,
    val_metrics,
    train_timestamps,
    test_timestamp,
    nn_model,
    model_ids,
    is_regression,
    k_neighbors=20,
    time_decay=0.9
):
    """
    K-NN + Bayesian Ensemble (True Bayesian posterior)
    
    W_i(x) ∝ π_i × sim_i(x) × conf_i
    
    Where:
      - π_i: Prior (time decay weight)
      - sim_i(x): Likelihood (data similarity via K-NN)
      - conf_i: Model confidence (from validation performance)
    
    Args:
        test_features: Test data features (N x D numpy array)
        predictions_list: List of predictions from each model (each is N-dim array)
        val_metrics: List of validation metrics
        train_timestamps: Training timestamps for each model
        test_timestamp: Test timestamp
        nn_model: Fitted NearestNeighbors
        model_ids: Array indicating which model each training sample belongs to
        is_regression: bool
        k_neighbors: number of neighbors
        time_decay: prior decay factor
    
    Returns:
        ensemble_predictions (numpy array), average_weights (numpy array)
    """
    n_models = len(predictions_list)
    n_test = len(test_features)
    
    # Step 1: Calculate priors (time decay)
    priors = np.array([time_decay ** (n_models - 1 - i) for i in range(n_models)])
    
    # Step 2: Calculate confidence from validation metrics
    conf = np.zeros(n_models)
    for i in range(n_models):
        if is_regression:
            # MAE: lower is better → higher confidence
            conf[i] = 1.0 / (1.0 + val_metrics[i])
        else:
            # AUC: higher is better
            conf[i] = val_metrics[i]
    
    print(f"\n  🎲 Priors (time_decay={time_decay}): {priors}")
    print(f"  💪 Confidence: {conf}")
    
    # Step 3: Per-sample ensemble (calculate similarity via K-NN)
    ensemble_preds = []
    all_weights = []
    
    print(f"  🔍 Computing sample-specific weights via K-NN...")
    
    for i, x in enumerate(test_features):
        # K-NN retrieval: find similar historical samples
        distances, indices = nn_model.kneighbors([x], n_neighbors=k_neighbors)
        neighbor_model_ids = model_ids[indices[0]]
        
        # Calculate sim_i: proportion of neighbors from each model's training data
        sim = np.zeros(n_models)
        for model_id in range(n_models):
            count = np.sum(neighbor_model_ids == model_id)
            sim[model_id] = count / k_neighbors
        
        # Bayesian weight: π_i × sim_i × conf_i
        weights = priors * sim * conf
        
        # Normalize
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            weights = weights / weight_sum
        else:
            # Fallback: uniform weights
            weights = np.ones(n_models) / n_models
        
        # Weighted prediction for this sample
        sample_preds = [predictions_list[j][i] for j in range(n_models)]
        ensemble_pred = np.dot(weights, sample_preds)
        
        ensemble_preds.append(ensemble_pred)
        all_weights.append(weights)
    
    ensemble_preds = np.array(ensemble_preds)
    all_weights = np.array(all_weights)
    
    # Calculate average weights across all samples
    avg_weights = all_weights.mean(axis=0)
    print(f"  📊 Average weights (across {n_test} samples): {avg_weights}")
    
    return ensemble_preds, avg_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['avito-user-clicks-01', 'event-user-ignore', 
                               'trial-site-success', 'event-user-attendance'],
                       help='Dataset name')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--output_dir', type=str, 
                       default='./result_raw_from_server/q_zero_ensemble_static',
                       help='Output directory')
    
    args = parser.parse_args()
    
    set_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("🏋️  Training Static Models for Ensemble")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")
    print("="*80 + "\n")
    
    # Load config
    with open('./q_zero_config.json', 'r') as f:
        config = json.load(f)
    
    # Get dataset config
    if args.dataset not in config:
        print(f"❌ Dataset {args.dataset} not found in config")
        return
    
    dataset_config = config[args.dataset]
    data_dir = dataset_config['data_dir']
    task_type_str = dataset_config['task_type']
    
    # Define 3 ResNet architectures (different diversity)
    architectures = [
        "2-128-256",        # 2 blocks, large channels
        "4-32-64-32-64",    # 4 blocks, alternating channels
        "3-64-128-64"       # 3 blocks, up-down pattern
    ]
    
    # Training timestamps (spread out more)
    train_timestamps = [1, 4, 7]
    np.random.shuffle(train_timestamps)  # Random assignment
    
    # Validation timestamps (use first 500 samples each)
    val_timestamps = [2, 5, 8]
    
    # Test timestamp (fixed at 10)
    test_timestamp = 10
    
    print(f"📋 Model Configuration:")
    print(f"  Task: {task_type_str}")
    for i, (arch, train_ts) in enumerate(zip(architectures, train_timestamps)):
        print(f"  Model {i+1}: {arch:20} → Train on timestamp {train_ts}")
    print(f"\n  Validation: First 500 samples from timestamps {val_timestamps}")
    print(f"  Test: Timestamp {test_timestamp}")
    print()
    
    # Train all 3 models
    trained_models = []
    val_metrics = []
    train_loaders = []  # Save for K-NN index building
    
    for arch, train_ts in zip(architectures, train_timestamps):
        model, val_metric = train_static_model(
            dataset_name=args.dataset,
            data_dir=data_dir,
            task_type_str=task_type_str,
            train_timestamp=train_ts,
            val_timestamps=val_timestamps,
            architecture=arch,
            device=device,
            max_epochs=200,
            patience=10
        )
        trained_models.append(model)
        val_metrics.append(val_metric)
        
        # Create train loader for this model (needed for K-NN index)
        table_data_temp = TableData.load_from_dir(data_dir)
        if not table_data_temp.is_materialize:
            from utils.resource import get_text_embedder_cfg
            text_cfg = get_text_embedder_cfg(device="cpu")
            table_data_temp.materilize(col_to_text_embedder_cfg=text_cfg)
        
        all_timestamps_temp = sorted(table_data_temp.train_df['timestamp'].unique())
        train_date = all_timestamps_temp[train_ts]
        train_indices = get_temporal_indices(data_dir, train_date)
        train_subset = Subset(table_data_temp.train_tf, train_indices)
        train_loader = torch_frame.data.DataLoader(train_subset, batch_size=512, shuffle=False)
        train_loaders.append(train_loader)
    
    print(f"\n{'='*80}")
    print(f"✅ All 3 models trained!")
    print(f"{'='*80}\n")
    
    # Test on timestamp 9
    table_data = TableData.load_from_dir(data_dir)
    
    # Materialize if needed
    if not table_data.is_materialize:
        from utils.resource import get_text_embedder_cfg
        text_cfg = get_text_embedder_cfg(device="cpu")
        table_data.materilize(col_to_text_embedder_cfg=text_cfg)
    
    all_timestamps = sorted(table_data.train_df['timestamp'].unique())
    test_date = all_timestamps[test_timestamp]
    
    test_indices = get_temporal_indices(data_dir, test_date)
    test_subset = Subset(table_data.train_tf, test_indices)
    test_loader = torch_frame.data.DataLoader(test_subset, batch_size=512, shuffle=False)
    
    print(f"{'='*80}")
    print(f"🧪 Testing on Timestamp {test_timestamp}")
    print(f"{'='*80}")
    print(f"  Test samples: {len(test_indices)}\n")
    
    is_regression = (task_type_str == 'regression')
    metric_name = "MAE" if is_regression else "AUC"
    
    # Get individual predictions and measure inference time
    all_predictions = []
    individual_metrics = []
    individual_times = []
    
    for i, model in enumerate(trained_models):
        # Measure GPU inference time
        start_time = time.time()
        metric, preds, ys = evaluate(model, test_loader, device, is_regression)
        gpu_time = time.time() - start_time
        
        # Measure CPU inference time
        model_cpu = model.to('cpu')
        start_time = time.time()
        metric_cpu, _, _ = evaluate(model_cpu, test_loader, torch.device('cpu'), is_regression)
        cpu_time = time.time() - start_time
        model.to(device)  # Move back to GPU
        
        all_predictions.append(preds)
        individual_metrics.append(metric)
        individual_times.append(cpu_time)  # Use CPU time
        print(f"  Model {i+1} ({architectures[i]:20}): {metric_name}={metric:.4f}, GPU={gpu_time:.3f}s, CPU={cpu_time:.3f}s")
    
    # Average Ensemble (measure time)
    start_time = time.time()
    avg_ensemble_preds = ensemble_average(all_predictions)
    if is_regression:
        avg_ensemble_metric = mean_absolute_error(ys, avg_ensemble_preds)
    else:
        avg_ensemble_metric = roc_auc_score(ys, avg_ensemble_preds)
    avg_ensemble_time = time.time() - start_time
    
    # Total time = 3x inference + ensemble computation
    total_avg_time = sum(individual_times) + avg_ensemble_time
    
    print(f"\n  📊 Average Ensemble: {metric_name}={avg_ensemble_metric:.4f}, Time={total_avg_time:.3f}s (3x{sum(individual_times)/3:.3f}s + {avg_ensemble_time:.3f}s)")
    
    # Bayesian Ensemble (measure time)
    start_time = time.time()
    bayesian_ensemble_preds, bayesian_weights = ensemble_bayesian(
        all_predictions, val_metrics, train_timestamps, test_timestamp, is_regression
    )
    if is_regression:
        bayesian_ensemble_metric = mean_absolute_error(ys, bayesian_ensemble_preds)
    else:
        bayesian_ensemble_metric = roc_auc_score(ys, bayesian_ensemble_preds)
    bayesian_ensemble_time = time.time() - start_time
    
    # Total time = 3x inference + bayesian computation
    total_bayesian_time = sum(individual_times) + bayesian_ensemble_time
    
    print(f"  📊 Bayesian Ensemble: {metric_name}={bayesian_ensemble_metric:.4f}, Time={total_bayesian_time:.3f}s (3x{sum(individual_times)/3:.3f}s + {bayesian_ensemble_time:.3f}s)")
    
    # Show Bayesian weights
    print(f"\n  🎯 Bayesian Weights (performance × temporal):")
    for i, (arch, train_ts, weight) in enumerate(zip(architectures, train_timestamps, bayesian_weights)):
        distance = abs(test_timestamp - train_ts)
        print(f"     Model {i+1} (train_ts={train_ts}, distance={distance}): {weight:.4f} ({weight*100:.1f}%)")
    
    # K-NN + Bayesian Ensemble
    print("\n" + "="*80)
    print("🔬 K-NN + Bayesian Ensemble (True Bayesian Posterior)")
    print("="*80)
    
    # Build K-NN index from training data
    nn_model, model_ids = build_knn_index(train_loaders, device, k_neighbors=20)
    
    if nn_model is not None:
        # Extract test features
        print(f"\n  Extracting test features...")
        test_features, test_labels = extract_features_from_loader(test_loader, device)
        print(f"    Test samples: {len(test_features)}")
        
        # Handle NaN in test features
        if np.isnan(test_features).any():
            nan_count = np.isnan(test_features).sum()
            print(f"    ⚠️  Found {nan_count} NaN values in test features, replacing with 0...")
            test_features = np.nan_to_num(test_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Run K-NN + Bayesian ensemble
        start_time = time.time()
        knn_bayesian_preds, knn_bayesian_weights = ensemble_knn_bayesian(
            test_features,
            all_predictions,
            val_metrics,
            train_timestamps,
            test_timestamp,
            nn_model,
            model_ids,
            is_regression,
            k_neighbors=20,
            time_decay=0.9
        )
        
        if is_regression:
            knn_bayesian_metric = mean_absolute_error(ys, knn_bayesian_preds)
        else:
            knn_bayesian_metric = roc_auc_score(ys, knn_bayesian_preds)
        
        knn_bayesian_time = time.time() - start_time
        total_knn_bayesian_time = sum(individual_times) + knn_bayesian_time
        
        print(f"\n  📊 K-NN + Bayesian: {metric_name}={knn_bayesian_metric:.4f}, Time={total_knn_bayesian_time:.3f}s (3x{sum(individual_times)/3:.3f}s + {knn_bayesian_time:.3f}s)")
        print(f"  🎯 Average weights (sample-specific): {knn_bayesian_weights}")
        
        print("="*80)
    else:
        knn_bayesian_metric = None
        knn_bayesian_weights = None
        knn_bayesian_time = None
        total_knn_bayesian_time = None
    
    # Compare to single model
    avg_single_time = sum(individual_times) / 3
    print(f"\n  ⏱️  Time Comparison:")
    print(f"     Single Model (avg): {avg_single_time:.3f}s")
    print(f"     Ensemble (3 models): {total_avg_time:.3f}s ({total_avg_time/avg_single_time:.1f}x slower)")
    print(f"     Overhead: {avg_ensemble_time:.3f}s ({avg_ensemble_time/total_avg_time*100:.1f}%)")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'dataset': args.dataset,
        'train_timestamps': train_timestamps,  # Already a list
        'val_timestamps': val_timestamps,
        'test_timestamp': test_timestamp,
        'architectures': architectures,
        'individual_metrics': individual_metrics,
        'val_metrics': val_metrics,
        'average_ensemble_metric': avg_ensemble_metric,
        'bayesian_ensemble_metric': bayesian_ensemble_metric,
        'bayesian_weights': bayesian_weights.tolist(),
        'knn_bayesian_metric': knn_bayesian_metric,
        'knn_bayesian_weights': knn_bayesian_weights.tolist() if knn_bayesian_weights is not None else None,
        'metric_type': metric_name,
        'individual_inference_times': individual_times,
        'avg_ensemble_time': avg_ensemble_time,
        'bayesian_ensemble_time': bayesian_ensemble_time,
        'knn_bayesian_time': knn_bayesian_time,
        'total_avg_time': total_avg_time,
        'total_bayesian_time': total_bayesian_time,
        'total_knn_bayesian_time': total_knn_bayesian_time
    }
    
    # Add config identifier to filename to avoid overwriting
    config_id = f"train{train_timestamps[0]}{train_timestamps[1]}{train_timestamps[2]}_test{test_timestamp}"
    output_file = output_dir / f'{args.dataset}_ensemble_{config_id}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ Results saved to: {output_file}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

