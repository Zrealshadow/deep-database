#!/usr/bin/env python3
"""
Zero-Training Expert Ensemble via K-NN + Bayesian Weighting

Strategy:
- Build K-NN index from ALL historical training data (offline, not counted in time)
- For each test sample x:
  1. K-NN retrieval: Find similar samples in history
  2. Calculate sim_i(x): proportion of neighbors from model i
  3. Calculate conf_i(x): model prediction confidence
  4. Calculate W_i(x) ∝ π_i × sim_i × conf_i
  5. Ensemble prediction: ŷ = Σ W_i × M_i(x)

No training during inference! Only retrieval + computation.
"""

import argparse
import json
import os
import sys
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_frame
from torch.utils.data import Subset
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from relbench.base import TaskType
from torch_frame.nn.models import MLP, ResNet, FTTransformer

from model.base import construct_stype_encoder_dict, default_stype_encoder_cls_kwargs
from utils.data import TableData


def safe_roc_auc(y_true, y_score):
    """Safe AUC that handles single-class edge cases"""
    y_true = np.asarray(y_true)
    if y_true.ndim != 1:
        y_true = y_true.reshape(-1)
    
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


def build_knn_index(table_data, all_groups, device):
    """
    Build K-NN index from ALL historical training data
    This is OFFLINE preparation, not counted in inference time
    
    Returns:
        nn_model: fitted NearestNeighbors
        X_all: all training features
        model_ids: which group each sample belongs to
    """
    print("\n" + "=" * 80)
    print("📊 Building K-NN Index (Offline, not timed)")
    print("=" * 80)
    
    X_all = []
    model_ids = []
    
    for group_idx, dates in enumerate(all_groups):
        # Get TensorFrame for this group
        indices = get_temporal_indices(table_data.dataset_dir, dates)
        if len(indices) == 0:
            continue
        
        temporal_tf = Subset(table_data.train_tf, indices)
        loader = torch_frame.data.DataLoader(temporal_tf, batch_size=len(indices), shuffle=False)
        
        # Extract features (use materialize's features)
        for batch in loader:
            batch = batch.to(device)
            
            # Extract all features as a flat vector
            feat_list = []
            for stype, feat_tensor in batch.feat_dict.items():
                if hasattr(feat_tensor, 'values'):
                    feat_np = feat_tensor.values.cpu().numpy()
                else:
                    feat_np = feat_tensor.cpu().numpy()
                
                if feat_np.ndim == 1:
                    feat_np = feat_np.reshape(-1, 1)
                elif feat_np.ndim == 3:
                    N, num_cols, emb_dim = feat_np.shape
                    feat_np = feat_np.reshape(N, num_cols * emb_dim)
                
                feat_list.append(feat_np)
            
            if len(feat_list) > 0:
                X_group = np.concatenate(feat_list, axis=1)
                X_all.append(X_group)
                model_ids.extend([group_idx] * len(X_group))
        
        print(f"  Group {group_idx}: {len(indices)} samples, {X_group.shape[1]} features")
    
    # Concatenate all groups
    X_all = np.vstack(X_all)
    model_ids = np.array(model_ids)
    
    print(f"\n  Total indexed: {len(X_all)} samples from {len(all_groups)} groups")
    print(f"  Feature dimension: {X_all.shape[1]}")
    
    # Build K-NN model
    print(f"\n  Building NearestNeighbors index...")
    nn_model = NearestNeighbors(n_neighbors=20, algorithm='auto', metric='euclidean')
    nn_model.fit(X_all)
    
    print(f"  ✅ K-NN index built!")
    print("=" * 80)
    
    return nn_model, X_all, model_ids


def extract_features_from_batch(batch, device):
    """Extract features from a TensorFrame batch as flat numpy array"""
    batch = batch.to(device)
    
    feat_list = []
    for stype, feat_tensor in batch.feat_dict.items():
        if hasattr(feat_tensor, 'values'):
            feat_np = feat_tensor.values.cpu().numpy()
        else:
            feat_np = feat_tensor.cpu().numpy()
        
        if feat_np.ndim == 1:
            feat_np = feat_np.reshape(-1, 1)
        elif feat_np.ndim == 3:
            N, num_cols, emb_dim = feat_np.shape
            feat_np = feat_np.reshape(N, num_cols * emb_dim)
        
        feat_list.append(feat_np)
    
    if len(feat_list) == 0:
        return np.array([])
    
    return np.concatenate(feat_list, axis=1)


def load_model(model_path):
    """Load a trained model from checkpoint"""
    checkpoint = torch.load(model_path, map_location='cpu')
    # Assuming checkpoint contains model state_dict
    return checkpoint


def predict_with_knn_bayesian_ensemble(
    test_loader, 
    models,  # List of loaded models
    nn_model, 
    model_ids,
    val_metrics,  # List of validation metrics for each model
    device,
    is_regression,
    k_neighbors=20,
    time_decay=0.9
):
    """
    Predict using K-NN + Bayesian ensemble
    
    Returns:
        predictions, labels, inference_time
    """
    start_time = time.time()
    
    all_preds = []
    all_labels = []
    
    n_models = len(models)
    
    # Set priors (time decay)
    priors = np.array([time_decay ** (n_models - 1 - i) for i in range(n_models)])
    
    for batch in test_loader:
        # Extract features
        X_batch = extract_features_from_batch(batch, device)
        y_batch = batch.y.cpu().numpy()
        
        batch_preds = []
        
        for x in X_batch:
            # Step 1: K-NN retrieval
            distances, indices = nn_model.kneighbors([x], n_neighbors=k_neighbors)
            neighbor_model_ids = model_ids[indices[0]]
            
            # Step 2: Calculate sim_i (data similarity)
            sim = np.zeros(n_models)
            for model_id in range(n_models):
                count = np.sum(neighbor_model_ids == model_id)
                sim[model_id] = count / k_neighbors
            
            # Step 3: Get predictions and conf_i from all models
            model_preds = []
            conf = np.zeros(n_models)
            
            for model_id, model in enumerate(models):
                # Predict with single sample
                x_tensor = torch.from_numpy(x).float().unsqueeze(0)
                
                # Need to convert to proper TensorFrame format
                # For simplicity, we'll use the batch structure
                # This is a simplified version - you may need to adapt
                
                with torch.no_grad():
                    # Convert x back to batch format (simplified)
                    # In practice, you'd reconstruct the TensorFrame
                    pred = model(batch.to(device))
                    
                    if pred.dim() == 2 and pred.size(1) == 1:
                        pred = pred.squeeze(1)
                    elif pred.dim() > 1:
                        pred = pred.reshape(pred.size(0), -1).squeeze(1)
                    
                    if is_regression:
                        model_pred = pred[0].cpu().item()
                        # For regression, use validation MAE to compute conf
                        conf[model_id] = 1.0 / (1.0 + val_metrics[model_id])
                    else:
                        model_pred = torch.sigmoid(pred[0]).cpu().item()
                        # For classification, use prediction confidence
                        conf[model_id] = max(model_pred, 1 - model_pred)
                    
                    model_preds.append(model_pred)
            
            # Step 4: Calculate Bayesian weights
            weights = priors * sim * conf
            
            # Normalize
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                weights = weights / weight_sum
            else:
                # Fallback: uniform weights
                weights = np.ones(n_models) / n_models
            
            # Step 5: Weighted ensemble prediction
            ensemble_pred = np.dot(weights, model_preds)
            batch_preds.append(ensemble_pred)
        
        all_preds.extend(batch_preds)
        all_labels.extend(y_batch)
    
    inference_time = time.time() - start_time
    
    return np.array(all_preds), np.array(all_labels), inference_time


def main():
    parser = argparse.ArgumentParser(description="K-NN + Bayesian Ensemble")
    
    # Config
    parser.add_argument("--dataset_id", type=str, required=True)
    parser.add_argument("--config_file", type=str, default="./qzero_config.json")
    
    # Model checkpoints directory
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing trained model checkpoints")
    
    # K-NN parameters
    parser.add_argument("--k_neighbors", type=int, default=20,
                        help="Number of neighbors for K-NN")
    parser.add_argument("--time_decay", type=float, default=0.9,
                        help="Time decay factor for prior weights")
    
    # Device
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./result_raw_from_server/knn_ensemble")
    parser.add_argument("--model_name", type=str, default="KNN-Bayesian-Ensemble")
    
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
    
    print("=" * 80)
    print("K-NN + Bayesian Ensemble (Zero-Training)")
    print("=" * 80)
    print(f"📋 Dataset: {args.dataset_id}")
    print(f"🎯 Task: {task_type_str}")
    print(f"🎲 Seed: {args.seed}")
    print(f"🖥️  Device: {device}")
    print(f"📅 Groups: {len(all_groups)}")
    print(f"🔍 K-neighbors: {args.k_neighbors}")
    print(f"⏰ Time decay: {args.time_decay}")
    
    # Load data
    print(f"\n📊 Loading dataset...")
    table_data = TableData.load_from_dir(data_dir)
    table_data.dataset_dir = data_dir  # Store for later use
    
    if not table_data.is_materialize:
        from utils.resource import get_text_embedder_cfg
        text_cfg = get_text_embedder_cfg(device="cpu")
        table_data.materilize(col_to_text_embedder_cfg=text_cfg)
    
    # Task setup
    if table_data.task_type == TaskType.REGRESSION:
        evaluate_func = mean_absolute_error
        higher_is_better = False
        is_regression = True
        metric_name = "MAE"
    else:
        evaluate_func = safe_roc_auc
        higher_is_better = True
        is_regression = False
        metric_name = "AUC"
    
    # Build K-NN index (OFFLINE - not counted in inference time)
    nn_model, X_all, all_model_ids = build_knn_index(table_data, all_groups, device)
    
    # Load all trained models from checkpoints
    print(f"\n📦 Loading trained models from {args.checkpoint_dir}...")
    models = []
    val_metrics = []
    
    # TODO: Load models from checkpoint_dir
    # For now, this is a placeholder - you need to implement model loading
    print(f"  ⚠️  Model loading not implemented yet")
    print(f"  Please implement loading models from checkpoint directory")
    
    # Results storage
    results = []
    
    # Test on each group (starting from group 1)
    print(f"\n📊 Testing with K-NN + Bayesian Ensemble...")
    print("=" * 80)
    
    for test_group_idx in range(1, len(all_groups)):
        test_dates = all_groups[test_group_idx]
        test_indices = get_temporal_indices(data_dir, test_dates)
        
        if len(test_indices) == 0:
            continue
        
        # Create test loader
        temporal_test_tf = Subset(table_data.train_tf, test_indices)
        test_loader = torch_frame.data.DataLoader(
            temporal_test_tf, 
            batch_size=256, 
            shuffle=False, 
            pin_memory=True
        )
        
        print(f"\n  Group {test_group_idx} ({','.join(test_dates[:3])}{'...' if len(test_dates)>3 else ''}): {len(test_indices)} samples")
        
        # Ensemble prediction with timing
        test_pred, test_y, inference_time = predict_with_knn_bayesian_ensemble(
            test_loader,
            models,
            nn_model,
            all_model_ids,
            val_metrics,
            device,
            is_regression,
            args.k_neighbors,
            args.time_decay
        )
        
        # Calculate metric
        test_metric = evaluate_func(test_y, test_pred) if len(test_y) > 0 else 0.0
        
        print(f"    Test {metric_name}={test_metric:.4f}, Inference Time={inference_time:.2f}s")
        
        # Store results
        results.append({
            'test_group_index': test_group_idx,
            'test_dates': ','.join(test_dates),
            'n_test_samples': len(test_indices),
            'test_metric': test_metric,
            'inference_time_seconds': inference_time,
            'inference_time_per_sample_ms': (inference_time / len(test_indices)) * 1000,
            'k_neighbors': args.k_neighbors,
            'time_decay': args.time_decay,
            'n_models': len(models),
            'dataset': args.dataset_id,
            'model_name': args.model_name,
            'metric_name': metric_name,
            'seed': args.seed
        })
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_date_str = all_groups[0][0].replace('-', '')[:4]
    result_file = os.path.join(
        args.output_dir,
        f"{args.dataset_id}_knn_bayesian_results.csv"
    )
    
    result_df = pd.DataFrame(results)
    
    if os.path.exists(result_file):
        existing_df = pd.read_csv(result_file)
        result_df = pd.concat([existing_df, result_df], ignore_index=True)
    
    result_df.to_csv(result_file, index=False)
    
    print(f"\n💾 Results saved: {result_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()

