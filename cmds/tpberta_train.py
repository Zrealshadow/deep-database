"""
TP-BERTa Training Script (Unified)

This script supports both training modes:
1. Full fine-tuning: Train entire model (encoder + head) - default
2. Frozen encoder: Freeze encoder, only train head

Usage:
    # Full fine-tuning (original way)
    python cmds/tpberta_train.py --data_dir data/... --freeze_encoder False

    # Frozen encoder (only train head)
    python cmds/tpberta_train.py --data_dir data/... --freeze_encoder True
"""

import os
import sys
import argparse
import json
import copy
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm
import scipy.special
from sklearn.metrics import mean_absolute_error, roc_auc_score, mean_squared_error

# TP-BERTa imports
from bin import build_default_model
from lib import DataConfig, prepare_tpberta_loaders, magnitude_regloss, calculate_metrics, make_tpberta_optimizer
from bin.tpberta_modeling import TPBertaForClassification, TPBertaForMTLPretrain, RobertaConfig

# Local imports
from utils.logger import ModernLogger

logger = ModernLogger(name="tpberta_train", level="info")


# Args class for build_default_model (like original code)
class Args:
    def __init__(self, pretrain_dir, max_position_embeddings, max_feature_length, 
                 max_numerical_token, max_categorical_token, feature_map, batch_size):
        # base_model_dir is only needed for MTL pre-training, not for fine-tuning
        # For fine-tuning, we use pretrain_dir
        self.base_model_dir = None  # Not used in fine-tuning
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = 5
        self.max_seq_length = 512
        self.max_feature_length = max_feature_length
        self.max_numerical_token = max_numerical_token
        self.max_categorical_token = max_categorical_token
        self.feature_map = feature_map
        self.batch_size = batch_size
        self.pretrain_dir = str(pretrain_dir)  # pre-trained TPBerta dir (like original code)
        self.model_suffix = "pytorch_models/best"


def check_upper_num(x: str):
    """Check if string has multiple uppercase letters"""
    x1, x2 = list(x), list(x.lower())
    num_upper = sum([c1 != c2 for c1, c2 in zip(x1, x2)])
    if num_upper > 1:
        return True
    if num_upper == 1 and x[0].lower() == x[0]:
        return True
    return False


def fix_camel_case(x: str):
    """Convert camelCase to space-separated lowercase"""
    words = []
    cur_char = x[0].lower()
    min_chars_len = 10000
    for c in x[1:]:
        if c.isupper():
            min_chars_len = min(min_chars_len, len(cur_char))
            words.append(cur_char)
            cur_char = c.lower()
        else:
            cur_char += c
    min_chars_len = min(min_chars_len, len(cur_char))
    words.append(cur_char)
    return ' '.join(words), min_chars_len


def generate_feature_names(data_dir: str, output_file: str = None):
    """
    Generate feature_names.json from CSV files in data directory.
    
    Args:
        data_dir: Directory containing train.csv, val.csv, test.csv
        output_file: Output file path (default: data_dir/feature_names.json)
    """
    data_dir = Path(data_dir)
    
    if output_file is None:
        output_file = data_dir / "feature_names.json"
    else:
        output_file = Path(output_file)
    
    # Read all CSV files to collect feature names
    csv_files = []
    for split in ["train", "val", "test"]:
        csv_path = data_dir / f"{split}.csv"
        if csv_path.exists():
            csv_files.append(csv_path)
    
    if not csv_files:
        raise FileNotFoundError(f"No train.csv, val.csv or test.csv found in {data_dir}")
    
    feature_name_dict = {}
    
    # Collect feature names from all CSV files
    for csv_file in csv_files:
        logger.info(f"Processing file: {csv_file}")
        df = pd.read_csv(csv_file)
        
        # Last column is target, skip it
        for feature in df.columns[:-1]:
            if feature in feature_name_dict:
                continue
            
            temp = feature
            # Handle underscores
            if '_' in temp:
                temp = ' '.join(temp.lower().split('_'))
            # Handle dots
            if '.' in feature:
                temp = ' '.join(temp.lower().split('.'))
            # Handle hyphens
            if '-' in feature:
                temp = ' '.join(temp.lower().split('-'))
            
            # Handle camelCase
            if check_upper_num(temp):
                std_feature, min_char_len = fix_camel_case(feature)
                if min_char_len == 1:
                    # Special terms, keep as is
                    feature_name_dict[feature] = feature
                else:
                    feature_name_dict[feature] = std_feature
            else:
                feature_name_dict[feature] = temp.lower()
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(feature_name_dict, f, indent=4)
    
    logger.info(f"Generated feature_names.json: {output_file} ({len(feature_name_dict)} features)")
    
    return output_file


def convert_tabledata_to_tpberta_format(
    data_dir: str,
    output_dir: Optional[str] = None
) -> tuple[str, str, dict]:
    """
    Convert TableData format to TP-BERTa format.
    
    Returns:
        Tuple of (csv_path, task_type, split_info)
    """
    data_dir = Path(data_dir)
    if output_dir is None:
        output_dir = data_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load target column info
    target_col_file = data_dir / "target_col.txt"
    if not target_col_file.exists():
        raise FileNotFoundError(f"target_col.txt not found in {data_dir}")
    
    with open(target_col_file, 'r') as f:
        target_col = f.readline().strip()
        task_type_str = f.readline().strip()
    
    # Map TaskType to TP-BERTa task type
    task_type_map = {
        "BINARY_CLASSIFICATION": "binclass",
        "REGRESSION": "regression",
        "MULTICLASS_CLASSIFICATION": "multiclass"
    }
    tpberta_task_type = task_type_map.get(task_type_str, "binclass")
    
    # Load all splits
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    
    # Make sure target column is last
    all_columns = [col for col in train_df.columns if col != target_col] + [target_col]
    train_df = train_df[all_columns]
    val_df = val_df[all_columns]
    test_df = test_df[all_columns]
    
    # Store split sizes
    split_info = {
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df)
    }
    
    # Combine all splits
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # Save combined CSV
    dataset_name = data_dir.name
    output_csv = output_dir / f"{dataset_name}.csv"
    combined_df.to_csv(output_csv, index=False)
    
    # Generate feature_names.json if it doesn't exist
    feature_names_file = output_dir / "feature_names.json"
    if not feature_names_file.exists():
        logger.info("Generating feature_names.json...")
        generate_feature_names(str(data_dir), str(feature_names_file))
    
    # Save split info
    split_info_file = output_dir / "split_info.json"
    with open(split_info_file, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    return str(output_csv), tpberta_task_type, split_info


def train_tpberta(
    data_dir: str,
    pretrain_dir: str,
    result_dir: str = "./tpberta_outputs",
    freeze_encoder: bool = False,  # New parameter: freeze encoder or not
    max_epochs: int = 200,
    early_stop: int = 50,
    batch_size: int = 64,
    lr: Optional[float] = None,  # Auto-set based on freeze_encoder
    weight_decay: float = 0.01,
    lamb: float = 0.0,
    max_position_embeddings: int = 64,
    max_feature_length: int = 8,
    max_numerical_token: int = 256,
    max_categorical_token: int = 16,
    feature_map: str = "feature_names.json",
):
    """
    Train TP-BERTa on TableData.
    
    Args:
        data_dir: Directory containing TableData
        result_dir: Output directory for results
        pretrain_dir: Path to pre-trained TP-BERTa model
        freeze_encoder: If True, freeze encoder and only train head. If False, train entire model.
        max_epochs: Maximum training epochs
        early_stop: Early stopping patience
        batch_size: Batch size
        lr: Learning rate (auto-set if None: 1e-3 for frozen, 1e-5 for full)
        weight_decay: Weight decay
        lamb: Regularization weight
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Training mode: {'Frozen encoder (only head)' if freeze_encoder else 'Full fine-tuning (encoder + head)'}")
    
    # Auto-set learning rate
    if lr is None:
        lr = 1e-3 if freeze_encoder else 1e-5
        logger.info(f"Auto-set learning rate: {lr}")
    
    # Convert data format
    data_dir_path = Path(data_dir)
    dataset_name = data_dir_path.name
    
    # Create temp directory in current directory (relative path)
    temp_dir = Path(f"./{dataset_name}_tpberta_temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path, task_type, split_info = convert_tabledata_to_tpberta_format(data_dir, temp_dir)
    
    # pretrain_dir must be set (assumed to exist)
    pretrain_path = Path(pretrain_dir)
    
    # Calculate train_ratio
    train_val_size = split_info['train_size'] + split_info['val_size']
    train_ratio = split_info['train_size'] / train_val_size if train_val_size > 0 else 0.8
    
    # Create data config using from_pretrained (like original code)
    # This loads data_config.json and tokenizer from pretrain_dir (tp-joint)
    data_config = DataConfig.from_pretrained(
        pretrain_path,
        data_dir=temp_dir,
        batch_size=batch_size,
        train_ratio=train_ratio,
        preproc_type='lm',
        pre_train=False
    )
    
    # Prepare args for build_default_model (like original code)
    args = Args(
        pretrain_path,
        max_position_embeddings,
        max_feature_length,
        max_numerical_token,
        max_categorical_token,
        feature_map,
        batch_size
    )
    
    # Load dataset
    logger.info(f"Loading dataset: {dataset_name}")
    data_loaders, datasets = prepare_tpberta_loaders([dataset_name], data_config, tt=task_type)
    
    if len(data_loaders) == 0 or len(datasets) == 0:
        raise ValueError(f"Failed to load dataset {dataset_name}")
    
    # data_loaders is a list of tuples: [(dataloader_dict, task_type), ...]
    # Extract the dataloader dictionary from the first tuple
    data_loader, _ = data_loaders[0]
    dataset = datasets[0]
    
    logger.info(f"Dataset loaded: {dataset_name}")
    logger.info(f"  Task type: {dataset.task_type.value}")
    logger.info(f"  Num classes: {dataset.n_classes}")
    
    # Build model (like original code)
    logger.info("Building TP-BERTa model...")
    # args.pretrain_dir is already set above
    model_config, model = build_default_model(
        args, data_config, dataset.n_classes, device, pretrain=True
    )  # use pre-trained weights & configs (like original code)
    logger.info(f"Loaded pre-trained model from {pretrain_dir}")
    
    # Memory optimization: For large feature sets, DataParallel may cause OOM
    # If model is wrapped in DataParallel and we have memory issues, unwrap it
    if isinstance(model, torch.nn.DataParallel):
        n_features = dataset.n_num_features + (dataset.n_cat_features or 0) + (dataset.n_str_features or 0)
        if n_features > 200:  # Large feature set
            logger.warning(f"Large feature set ({n_features} features) detected. "
                         f"Consider using smaller batch_size (<= 4) to avoid OOM.")
            # Optionally unwrap DataParallel for single GPU (uncomment if needed):
            # logger.info("Unwrapping DataParallel to reduce memory usage...")
            # model = model.module
    
    # Get the actual model (unwrap DataParallel if needed)
    actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    # Freeze encoder if requested
    if freeze_encoder:
        logger.info("Freezing TP-BERTa encoder (only training head)...")
        for param in actual_model.tpberta.parameters():
            param.requires_grad = False
        # Setup optimizer (only head parameters)
        optimizer = torch.optim.AdamW(
            actual_model.classifier.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        logger.info(f"Optimizer: Only head parameters (lr={lr})")
    else:
        # Setup optimizer (all parameters)
        optimizer = make_tpberta_optimizer(actual_model, lr=lr, weight_decay=weight_decay)
        logger.info(f"Optimizer: All parameters (lr={lr})")
    
    # Training loop
    logger.info("Starting training...")
    best_metric = -np.inf
    final_test_metric = 0
    no_improvement = 0
    best_model_state = None  # Store best model state
    
    metric_key = {
        'regression': 'rmse',
        'binclass': 'roc_auc',
        'multiclass': 'accuracy'
    }.get(dataset.task_type.value, 'roc_auc')
    
    scale = 1 if not dataset.is_regression else -1
    
    tr_task_losses, tr_reg_losses = [], []
    ev_task_losses, ev_metrics = [], []
    test_metrics = []
    
    for epoch in tqdm(range(max_epochs), desc="Training"):
        # Training
        model.train()
        tr_loss = 0.0
        reg_loss = 0.0
        
        for batch in tqdm(data_loader['train'], desc=f"Epoch {epoch+1}", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop('labels')
            
            optimizer.zero_grad()
            logits, outputs = model(**batch)
            y = labels.float()
            
            # Task loss
            # Note: This code supports binclass and regression tasks
            if dataset.is_regression:
                # Regression: logits shape [batch_size, 1] -> squeeze to [batch_size]
                task_loss = torch.nn.functional.mse_loss(logits.squeeze(), y)
            elif dataset.task_type.value == 'binclass':
                # Binary classification: logits shape [batch_size, 1] -> squeeze to [batch_size]
                task_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits.squeeze(), y
                )
            else:  # multiclass (not used in your data, but kept for compatibility)
                task_loss = torch.nn.functional.cross_entropy(
                    logits, y.long()
                )
            
            # Regularization loss (if enabled)
            if lamb > 0 and len(outputs) > 1:
                reg_loss_batch = magnitude_regloss(outputs[0], batch['input_scales'])
                total_loss = task_loss + lamb * reg_loss_batch
            else:
                total_loss = task_loss
            
            total_loss.backward()
            optimizer.step()
            
            tr_loss += task_loss.item()
            if lamb > 0 and len(outputs) > 1:
                reg_loss += reg_loss_batch.item()
        
        tr_task_losses.append(tr_loss / len(data_loader['train']))
        if lamb > 0:
            tr_reg_losses.append(reg_loss / len(data_loader['train']))
        
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for batch in data_loader['val']:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop('labels')
                logits, _ = model(**batch)
                # For binclass and regression: logits shape [batch_size, 1] -> squeeze to [batch_size]
                # For multiclass: logits shape [batch_size, n_classes] -> keep as is
                if dataset.is_multiclass:
                    val_preds.append(logits.cpu())
                else:
                    val_preds.append(logits.squeeze().cpu())
                val_targets.append(labels.cpu())
        
        val_preds = torch.cat(val_preds).numpy()
        val_targets = torch.cat(val_targets).numpy()
        
        val_metrics = calculate_metrics(
            val_targets,
            val_preds,
            dataset.task_type.value,
            'logits' if not dataset.is_regression else None,
            dataset.y_info
        )
        val_metric = val_metrics[metric_key] * scale
        
        ev_task_losses.append(0.0)
        ev_metrics.append(val_metric)
        
        logger.info(f"Epoch {epoch+1}/{max_epochs}: "
                   f"Train Loss: {tr_task_losses[-1]:.4f}, "
                   f"Val {metric_key}: {val_metric:.4f}")
        
        # Check for improvement
        if val_metric * scale > best_metric * scale:
            best_metric = val_metric
            no_improvement = 0
            # Save best model state (deep copy)
            if isinstance(model, torch.nn.DataParallel):
                best_model_state = copy.deepcopy(model.module.state_dict())
            else:
                best_model_state = copy.deepcopy(model.state_dict())
        else:
            no_improvement += 1
        
        # Early stopping
        if no_improvement >= early_stop:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Test evaluation
    model.eval()
    test_preds, test_targets = [], []
    
    with torch.no_grad():
        for batch in data_loader['test']:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop('labels')
            logits, _ = model(**batch)
            # For binclass and regression: logits shape [batch_size, 1] -> squeeze to [batch_size]
            # For multiclass: logits shape [batch_size, n_classes] -> keep as is
            if dataset.is_multiclass:
                test_preds.append(logits.cpu())
            else:
                test_preds.append(logits.squeeze().cpu())
            test_targets.append(labels.cpu())
    
    test_preds = torch.cat(test_preds).numpy()
    test_targets = torch.cat(test_targets).numpy()
    
    test_metrics_dict = calculate_metrics(
        test_targets,
        test_preds,
        dataset.task_type.value,
        'logits' if not dataset.is_regression else None,
        dataset.y_info
    )
    final_test_metric = test_metrics_dict[metric_key] * scale
    
    # Save results
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Create tpberta subfolder, then dataset subfolder
    output_dir = result_dir / "tpberta" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results.json
    results = {
        'best_val_metric': best_metric,
        'final_test_metric': final_test_metric,
        'test_metrics': test_metrics_dict,
        'metric_key': metric_key,
        'train_losses': tr_task_losses,
        'val_metrics': ev_metrics,
        'freeze_encoder': freeze_encoder,
        'lr': lr,
        'num_epochs_trained': epoch + 1,
        'batch_size': batch_size,
        'max_epochs': max_epochs,
        'early_stop': early_stop,
        'weight_decay': weight_decay,
        'lamb': lamb,
        'task_type': task_type,
        'dataset_name': dataset_name,
    }
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save predictions (test set)
    np.save(output_dir / "test_predictions.npy", test_preds)
    np.save(output_dir / "test_targets.npy", test_targets)
    
    # Save predictions as CSV for easy viewing
    if dataset.is_regression:
        pred_df = pd.DataFrame({
            'target': test_targets,
            'prediction': test_preds
        })
    elif dataset.task_type.value == 'binclass':
        pred_probs = scipy.special.expit(test_preds)  # Convert logits to probabilities
        pred_df = pd.DataFrame({
            'target': test_targets,
            'prediction': pred_probs
        })
    else:  # multiclass
        pred_probs = scipy.special.softmax(test_preds, axis=1)
        pred_df = pd.DataFrame({
            'target': test_targets,
            **{f'class_{i}_prob': pred_probs[:, i] for i in range(pred_probs.shape[1])},
            'predicted_class': pred_probs.argmax(axis=1)
        })
    pred_df.to_csv(output_dir / "test_predictions.csv", index=False)
    
    # Save training history
    history = {
        'train_losses': tr_task_losses,
        'val_metrics': ev_metrics,
        'tr_reg_losses': tr_reg_losses if lamb > 0 else [],
    }
    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save best model weights (restore best model state first)
    if best_model_state is not None:
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(best_model_state)
        else:
            model.load_state_dict(best_model_state)
        logger.info("Restored best model state for saving")
    
    best_model_path = output_dir / "best_model.pth"
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), best_model_path)
    else:
        torch.save(model.state_dict(), best_model_path)
    
    # Save model config
    config_info = {
        'model_config': model_config.to_dict() if hasattr(model_config, 'to_dict') else str(model_config),
        'data_config': {
            'num_cont_token': data_config.num_cont_token,
            'num_cat_token': data_config.num_cat_token,
            'max_feature_length': data_config.max_feature_length,
            'max_seq_length': data_config.max_seq_length,
        },
        'pretrain_dir': str(pretrain_dir),
    }
    with open(output_dir / "model_config.json", 'w') as f:
        json.dump(config_info, f, indent=2, default=str)
    
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"  - results.json: Training results and metrics")
    logger.info(f"  - test_predictions.npy: Test set predictions (numpy)")
    logger.info(f"  - test_predictions.csv: Test set predictions (CSV)")
    logger.info(f"  - test_targets.npy: Test set ground truth")
    logger.info(f"  - training_history.json: Training history")
    logger.info(f"  - best_model.pth: Best model weights")
    logger.info(f"  - model_config.json: Model configuration")
    logger.info(f"Best Val {metric_key}: {best_metric:.4f}")
    logger.info(f"Final Test {metric_key}: {final_test_metric:.4f}")
    
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="TP-BERTa Training (Unified)")
    
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to TableData directory")
    parser.add_argument("--result_dir", type=str, default="./tpberta_outputs",
                       help="Output directory for results")
    parser.add_argument("--pretrain_dir", type=str, default=None,
                       help="Path to pre-trained TP-BERTa model")
    parser.add_argument("--freeze_encoder", action="store_true",
                       help="Freeze encoder, only train head (default: False, train entire model)")
    parser.add_argument("--max_epochs", type=int, default=200,
                       help="Maximum training epochs")
    parser.add_argument("--early_stop", type=int, default=50,
                       help="Early stopping patience")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=None,
                       help="Learning rate (auto: 1e-3 if freeze_encoder, 1e-5 otherwise)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--lamb", type=float, default=0.0,
                       help="Regularization weight")
    
    args = parser.parse_args()
    
    # Get pretrain_dir from args or environment variable
    pretrain_dir = args.pretrain_dir or os.environ.get("TPBERTA_PRETRAIN_DIR")
    
    train_tpberta(
        data_dir=args.data_dir,
        result_dir=args.result_dir,
        pretrain_dir=pretrain_dir,
        freeze_encoder=args.freeze_encoder,
        max_epochs=args.max_epochs,
        early_stop=args.early_stop,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lamb=args.lamb,
    )


if __name__ == "__main__":
    main()

