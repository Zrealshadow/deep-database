"""
TP-BERTa Preprocessing Module

Converts CSV rows (semicolon-separated strings) to embedding strings.
"""

import sys
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import torch
import base64
import json

# Add tp-berta to path (assuming it's in the parent directory of sharing-embedding-table)
# This matches the structure: relational_data/tp-berta and relational_data/sharing-embedding-table
tpberta_root = Path(__file__).resolve().parent.parent.parent.parent / "tp-berta"
if tpberta_root.exists():
    sys.path.insert(0, str(tpberta_root))

# TP-BERTa imports (will work if tp-berta is in sys.path)
try:
    from bin import build_default_model
    from lib import DataConfig, prepare_tpberta_loaders
    from bin.tpberta_modeling import TPBertaForClassification, RobertaConfig
except ImportError as e:
    raise ImportError(
        f"Failed to import TP-BERTa modules. Make sure tp-berta directory is in the correct location. "
        f"Error: {e}. Tried path: {tpberta_root}"
    )


def process_csv_rows_to_embeddings(
    csv_rows: List[str],
    pretrain_dir: str,
    feature_names_file: Optional[str] = None,
    delimiter: str = ";",
    output_format: str = "base64",  # "base64" or "comma_separated"
    device: Optional[str] = None,
) -> List[str]:
    """
    Process CSV rows (semicolon-separated strings) to embedding strings.
    
    Args:
        csv_rows: List of CSV row strings, each row is semicolon-separated values.
                  Format: "value1;value2;value3;...;label" (label is last)
        pretrain_dir: Path to pre-trained TP-BERTa model directory
        feature_names_file: Path to feature_names.json (optional, will generate if not provided)
        delimiter: Delimiter used in CSV rows (default: ";")
        output_format: Output format for embeddings ("base64" or "comma_separated")
        device: Device to use (default: "cuda" if available, else "cpu")
    
    Returns:
        List of embedding strings (one per input row)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    # Parse CSV rows into DataFrame
    rows_data = []
    for row in csv_rows:
        values = row.strip().split(delimiter)
        rows_data.append(values)
    
    # Create DataFrame (assume last column is label)
    if not rows_data:
        return []
    
    num_cols = len(rows_data[0])
    col_names = [f"feature_{i}" for i in range(num_cols - 1)] + ["label"]
    df = pd.DataFrame(rows_data, columns=col_names)
    
    # Convert to TP-BERTa format and get embeddings
    embeddings = _get_tpberta_embeddings(
        df=df,
        pretrain_dir=pretrain_dir,
        feature_names_file=feature_names_file,
        device=device,
    )
    
    # Convert embeddings to string format
    embedding_strings = []
    for emb in embeddings:
        if output_format == "base64":
            # Encode as base64 string
            emb_bytes = emb.tobytes()
            emb_b64 = base64.b64encode(emb_bytes).decode('utf-8')
            embedding_strings.append(emb_b64)
        elif output_format == "comma_separated":
            # Convert to comma-separated string
            emb_str = ",".join([str(x) for x in emb.flatten()])
            embedding_strings.append(emb_str)
        else:
            raise ValueError(f"Unknown output_format: {output_format}")
    
    return embedding_strings


def _get_tpberta_embeddings(
    df: pd.DataFrame,
    pretrain_dir: str,
    feature_names_file: Optional[str] = None,
    device: torch.device = None,
) -> np.ndarray:
    """
    Get TP-BERTa embeddings for a DataFrame.
    
    Args:
        df: DataFrame with features and label (label is last column)
        pretrain_dir: Path to pre-trained TP-BERTa model
        feature_names_file: Path to feature_names.json
        device: Device to use
    
    Returns:
        numpy array of embeddings [N, hidden_size]
    """
    from pathlib import Path
    import tempfile
    import shutil
    
    # Create temporary directory for TP-BERTa processing
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Save DataFrame as CSV
        dataset_name = "temp_dataset"
        csv_path = temp_dir / f"{dataset_name}.csv"
        df.to_csv(csv_path, index=False)
        
        # Generate feature_names.json if not provided
        if feature_names_file is None or not Path(feature_names_file).exists():
            feature_names_file = temp_dir / "feature_names.json"
            _generate_feature_names(df, feature_names_file)
        else:
            # Copy to temp dir
            import shutil
            shutil.copy(feature_names_file, temp_dir / "feature_names.json")
        
        # Load pre-trained model
        pretrain_path = Path(pretrain_dir)
        data_config = DataConfig.from_pretrained(
            pretrain_path,
            data_dir=temp_dir,
            batch_size=32,
            train_ratio=1.0,  # Use all data
            preproc_type='lm',
            pre_train=False
        )
        
        # Determine task type from label
        label_col = df.columns[-1]
        label_values = df[label_col].dropna().unique()
        if len(label_values) == 2:
            task_type = "binclass"
        elif df[label_col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            task_type = "regression"
        else:
            task_type = "binclass"  # default
        
        # Prepare data loaders
        from lib.data_utils import prepare_tpberta_loaders
        data_loaders, datasets = prepare_tpberta_loaders(
            [dataset_name], 
            data_config, 
            tt=task_type
        )
        
        if len(data_loaders) == 0:
            raise ValueError("Failed to prepare data loaders")
        
        data_loader, _ = data_loaders[0]
        dataset = datasets[0]
        
        # Build model (encoder only, no head needed for embeddings)
        class Args:
            def __init__(self, pretrain_dir, max_position_embeddings, max_feature_length,
                         max_numerical_token, max_categorical_token, feature_map, batch_size):
                self.base_model_dir = None
                self.max_position_embeddings = max_position_embeddings
                self.type_vocab_size = 5
                self.max_seq_length = 512
                self.max_feature_length = max_feature_length
                self.max_numerical_token = max_numerical_token
                self.max_categorical_token = max_categorical_token
                self.feature_map = feature_map
                self.batch_size = batch_size
                self.pretrain_dir = str(pretrain_dir)
                self.model_suffix = "pytorch_models/best"
        
        args = Args(
            pretrain_path,
            max_position_embeddings=64,
            max_feature_length=8,
            max_numerical_token=256,
            max_categorical_token=16,
            feature_map="feature_names.json",
            batch_size=32
        )
        
        model_config, model = build_default_model(
            args, data_config, dataset.n_classes, device, pretrain=True
        )
        
        # Extract embeddings (use CLS token from encoder output)
        model.eval()
        all_embeddings = []
        
        with torch.no_grad():
            for batch in data_loader['train']:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop('labels', None)
                
                # Get encoder output
                outputs = model.tpberta(**batch)
                # Use pooled output (CLS token)
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    embeddings = outputs.pooler_output
                else:
                    # Fallback: use first token (CLS)
                    embeddings = outputs.last_hidden_state[:, 0, :]
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        # Concatenate all embeddings
        final_embeddings = np.vstack(all_embeddings)
        
        return final_embeddings
        
    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def _generate_feature_names(df: pd.DataFrame, output_file: Path):
    """Generate feature_names.json from DataFrame."""
    feature_name_dict = {}
    
    for col in df.columns[:-1]:  # Skip label column
        temp = col
        # Handle underscores
        if '_' in temp:
            temp = ' '.join(temp.lower().split('_'))
        # Handle dots
        if '.' in temp:
            temp = ' '.join(temp.lower().split('.'))
        # Handle hyphens
        if '-' in temp:
            temp = ' '.join(temp.lower().split('-'))
        
        feature_name_dict[col] = temp.lower()
    
    with open(output_file, 'w') as f:
        json.dump(feature_name_dict, f, indent=4)

