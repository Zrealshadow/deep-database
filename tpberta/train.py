"""
TP-BERTa Training Module

Trains prediction head on preprocessed embedding data.
"""

import sys
from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import base64
import json
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error


class EmbeddingDataset(Dataset):
    """Dataset for loading embedding strings and labels."""
    
    def __init__(self, embedding_strings: List[str], labels: np.ndarray, 
                 embedding_dim: int, format: str = "base64"):
        """
        Args:
            embedding_strings: List of embedding strings
            labels: Array of labels
            embedding_dim: Dimension of embeddings
            format: Format of embedding strings ("base64" or "comma_separated")
        """
        self.embedding_strings = embedding_strings
        self.labels = labels
        self.embedding_dim = embedding_dim
        self.format = format
    
    def __len__(self):
        return len(self.embedding_strings)
    
    def __getitem__(self, idx):
        # Decode embedding string
        if self.format == "base64":
            emb_bytes = base64.b64decode(self.embedding_strings[idx])
            embedding = np.frombuffer(emb_bytes, dtype=np.float32).reshape(-1)
        elif self.format == "comma_separated":
            embedding = np.array([float(x) for x in self.embedding_strings[idx].split(",")])
        else:
            raise ValueError(f"Unknown format: {self.format}")
        
        # Ensure correct dimension
        if len(embedding) != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {len(embedding)}")
        
        return {
            'embedding': torch.FloatTensor(embedding),
            'label': torch.FloatTensor([self.labels[idx]])
        }


class PredictionHead(nn.Module):
    """Simple MLP prediction head."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128], 
                 output_dim: int = 1, dropout: float = 0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def train_prediction_head(
    embedding_csv_path: str,
    output_dir: str,
    embedding_dim: int,
    embedding_format: str = "base64",  # "base64" or "comma_separated"
    task_type: str = "binclass",  # "binclass" or "regression"
    hidden_dims: List[int] = [256, 128],
    dropout: float = 0.2,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    num_epochs: int = 200,
    early_stop: int = 50,
    device: Optional[str] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict:
    """
    Train prediction head on preprocessed embedding data.
    
    Args:
        embedding_csv_path: Path to CSV file with columns: ['embedding', 'label']
                           or ['embedding_string', 'label'] where embedding_string is base64/comma-separated
        output_dir: Directory to save model and results
        embedding_dim: Dimension of embeddings
        embedding_format: Format of embedding strings ("base64" or "comma_separated")
        task_type: Task type ("binclass" or "regression")
        hidden_dims: Hidden layer dimensions for prediction head
        dropout: Dropout rate
        batch_size: Batch size
        learning_rate: Learning rate
        num_epochs: Maximum number of epochs
        early_stop: Early stopping patience
        device: Device to use (default: "cuda" if available)
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
    
    Returns:
        Dictionary with training results and metrics
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading embedding data from {embedding_csv_path}...")
    df = pd.read_csv(embedding_csv_path)
    
    # Find embedding and label columns
    embedding_col = None
    for col in ['embedding', 'embedding_string', 'embedding_str']:
        if col in df.columns:
            embedding_col = col
            break
    
    if embedding_col is None:
        raise ValueError(f"Could not find embedding column in CSV. Available columns: {df.columns.tolist()}")
    
    label_col = 'label'
    if label_col not in df.columns:
        raise ValueError(f"Could not find label column in CSV. Available columns: {df.columns.tolist()}")
    
    embedding_strings = df[embedding_col].tolist()
    labels = df[label_col].values
    
    # Split data
    n_total = len(embedding_strings)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    
    train_strings = embedding_strings[:n_train]
    train_labels = labels[:n_train]
    
    val_strings = embedding_strings[n_train:n_train+n_val]
    val_labels = labels[n_train:n_train+n_val]
    
    test_strings = embedding_strings[n_train+n_val:]
    test_labels = labels[n_train+n_val:]
    
    print(f"Data split: Train={n_train}, Val={n_val}, Test={n_test}")
    
    # Create datasets
    train_dataset = EmbeddingDataset(train_strings, train_labels, embedding_dim, embedding_format)
    val_dataset = EmbeddingDataset(val_strings, val_labels, embedding_dim, embedding_format)
    test_dataset = EmbeddingDataset(test_strings, test_labels, embedding_dim, embedding_format)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = PredictionHead(
        input_dim=embedding_dim,
        hidden_dims=hidden_dims,
        output_dim=1,
        dropout=dropout
    ).to(device)
    
    # Setup loss and optimizer
    if task_type == "binclass":
        criterion = nn.BCEWithLogitsLoss()
        metric_fn = roc_auc_score
        higher_is_better = True
    else:  # regression
        criterion = nn.MSELoss()
        metric_fn = mean_squared_error
        higher_is_better = False
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_metric = -np.inf if higher_is_better else np.inf
    best_model_state = None
    no_improvement = 0
    train_losses = []
    val_metrics = []
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Task: {task_type}, Device: {device}")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            embeddings = batch['embedding'].to(device)
            labels = batch['label'].squeeze().to(device)
            
            optimizer.zero_grad()
            logits = model(embeddings).squeeze()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch['embedding'].to(device)
                labels = batch['label'].squeeze().cpu().numpy()
                
                logits = model(embeddings).squeeze().cpu().numpy()
                
                if task_type == "binclass":
                    probs = 1 / (1 + np.exp(-logits))  # sigmoid
                    val_preds.extend(probs)
                else:
                    val_preds.extend(logits)
                
                val_targets.extend(labels)
        
        val_metric = metric_fn(val_targets, val_preds)
        val_metrics.append(val_metric)
        
        # Check for improvement
        is_better = (higher_is_better and val_metric > best_val_metric) or \
                   (not higher_is_better and val_metric < best_val_metric)
        
        if is_better:
            best_val_metric = val_metric
            best_model_state = model.state_dict().copy()
            no_improvement = 0
            print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.6f} | Val {metric_fn.__name__}: {val_metric:.6f} *")
        else:
            no_improvement += 1
            print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.6f} | Val {metric_fn.__name__}: {val_metric:.6f}")
        
        # Early stopping
        if no_improvement >= early_stop:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model and evaluate on test set
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    model.eval()
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            embeddings = batch['embedding'].to(device)
            labels = batch['label'].squeeze().cpu().numpy()
            
            logits = model(embeddings).squeeze().cpu().numpy()
            
            if task_type == "binclass":
                probs = 1 / (1 + np.exp(-logits))  # sigmoid
                test_preds.extend(probs)
            else:
                test_preds.extend(logits)
            
            test_targets.extend(labels)
    
    test_metric = metric_fn(test_targets, test_preds)
    
    # Save results
    results = {
        'best_val_metric': float(best_val_metric),
        'test_metric': float(test_metric),
        'task_type': task_type,
        'embedding_dim': embedding_dim,
        'num_epochs_trained': epoch + 1,
        'train_losses': train_losses,
        'val_metrics': val_metrics,
    }
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save model
    torch.save({
        'model_state_dict': best_model_state,
        'embedding_dim': embedding_dim,
        'hidden_dims': hidden_dims,
        'task_type': task_type,
    }, output_dir / "prediction_head.pth")
    
    # Save predictions
    np.save(output_dir / "test_predictions.npy", np.array(test_preds))
    np.save(output_dir / "test_targets.npy", np.array(test_targets))
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Val {metric_fn.__name__}: {best_val_metric:.6f}")
    print(f"Test {metric_fn.__name__}: {test_metric:.6f}")
    print(f"Model saved to: {output_dir / 'prediction_head.pth'}")
    
    return results

