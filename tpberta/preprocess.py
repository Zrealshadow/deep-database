"""
TP-BERTa Preprocessing Module

Converts CSV rows (semicolon-separated strings) to embedding strings.
"""

import tempfile
import shutil
import argparse
import traceback
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import torch
import json

from bin import build_default_model
from lib import DataConfig
from lib.data_utils import prepare_tpberta_loaders


class ModelArgs:
    """Arguments class for building TP-BERTa model."""

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

#
# def process_csv_rows_to_embeddings(
#         csv_rows: List[str],
#         pretrain_dir: str,
#         feature_names_file: Optional[str] = None,
#         delimiter: str = ";",
#         device: Optional[str] = None,
# ) -> List[str]:
#     """
#     Process CSV rows (semicolon-separated strings) to embedding strings.
#
#     Args:
#         csv_rows: List of CSV row strings, each row is semicolon-separated values.
#                   Format: "value1;value2;value3;...;label" (label is last)
#         pretrain_dir: Path to pre-trained TP-BERTa model directory
#         feature_names_file: Path to feature_names.json (optional, will generate if not provided)
#         delimiter: Delimiter used in CSV rows (default: ";")
#         device: Device to use (default: "cuda" if available, else "cpu")
#
#     Returns:
#         List of comma-separated embedding strings (one per input row)
#     """
#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#     device = torch.device(device)
#
#     # Parse CSV rows into DataFrame
#     rows_data = []
#     for row in csv_rows:
#         values = row.strip().split(delimiter)
#         rows_data.append(values)
#
#     # Create DataFrame (assume last column is label)
#     if not rows_data:
#         return []
#
#     num_cols = len(rows_data[0])
#     col_names = [f"feature_{i}" for i in range(num_cols - 1)] + ["label"]
#     df = pd.DataFrame(rows_data, columns=col_names)
#
#     # Convert to TP-BERTa format and get embeddings
#     # Check if DataFrame has label column (assume last column is label if it exists)
#     has_label = "label" in df.columns
#     embeddings = _get_tpberta_embeddings(
#         df=df,
#         pretrain_dir=pretrain_dir,
#         feature_names_file=feature_names_file,
#         device=device,
#         has_label=has_label,
#     )
#
#     # Convert embeddings to comma-separated strings
#     embedding_strings = []
#     for emb in embeddings:
#         emb_str = ",".join([str(x) for x in emb.flatten()])
#         embedding_strings.append(emb_str)
#
#     return embedding_strings


def _get_tpberta_embeddings(
        df: pd.DataFrame,
        pretrain_dir: str,
        feature_names_file: Optional[str] = None,
        device: torch.device = None,
        has_label: bool = True,
        dataset_name: str = "temp_dataset",
) -> np.ndarray:
    """
    Get TP-BERTa embeddings for a DataFrame.
    
    Args:
        df: DataFrame with features. If has_label=True, label should be the last column.
        pretrain_dir: Path to pre-trained TP-BERTa model
        feature_names_file: Path to feature_names.json
        device: Device to use (can be torch.device or str like "cuda"/"cpu")
        has_label: Whether the DataFrame has a label column (default: True)
    
    Returns:
        numpy array of embeddings [N, hidden_size]
    
    Note:
        temp_dir is used because TP-BERTa's data loaders require reading CSV and 
        feature_names.json from the filesystem. We create a temporary directory,
        save the DataFrame as CSV there, process it, then clean up.
        
        If has_label=False, a dummy label column will be added (TP-BERTa requires it).
    """
    # Convert device string to torch.device if needed
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    
    # Create temporary directory for TP-BERTa processing
    # TP-BERTa data loaders need to read from filesystem, so we save DataFrame to temp CSV
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Prepare DataFrame: TP-BERTa requires label column as last column
        df_to_save = df.copy()
        if not has_label:
            # Add dummy label column (TP-BERTa data loader requires it)
            df_to_save['dummy_label'] = 0
        
        # Save DataFrame as CSV
        csv_path = temp_dir / f"{dataset_name}.csv"
        df_to_save.to_csv(csv_path, index=False)

        # Handle feature_names.json: TP-BERTa MUST read from filesystem
        # Auto-generate if not provided (will be cleaned up with temp_dir)
        temp_feature_names_file = temp_dir / "feature_names.json"
        
        if feature_names_file is None or not Path(feature_names_file).exists():
            # Auto-generate from DataFrame (temporary, will be cleaned up)
            _generate_feature_names(df, temp_feature_names_file, has_label=has_label)
        else:
            # File path provided: copy to temp dir
            shutil.copy(feature_names_file, temp_feature_names_file)

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

        # For embedding extraction, task_type doesn't matter (we only use encoder, not head)
        # TP-BERTa data loader requires a valid task_type, but it won't affect embeddings
        # Prepare data loaders
        data_loaders, datasets = prepare_tpberta_loaders(
            [dataset_name],
            data_config,
            tt="binclass"  # Default task_type (required by TP-BERTa, but doesn't affect embeddings)
        )

        if len(data_loaders) == 0:
            raise ValueError("Failed to prepare data loaders")

        data_loader, _ = data_loaders[0]
        dataset = datasets[0]

        # Build model (encoder only, no head needed for embeddings)
        args = ModelArgs(
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

        # Handle DataParallel: if model is wrapped, access via .module
        actual_model = model.module if isinstance(model, torch.nn.DataParallel) else model

        # Extract embeddings (use CLS token from encoder output)
        actual_model.eval()
        all_embeddings = []

        with torch.no_grad():
            for batch in data_loader['train']:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop('labels', None)

                # Get encoder output
                outputs = actual_model.tpberta(**batch)
                # Extract CLS token (first token) from sequence output
                # Note: TPBertaForClassification and TPBertaForMTLPretrain both set
                # add_pooling_layer=False, so pooler_output is always None.
                # We directly use the CLS token from last_hidden_state.
                embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

                all_embeddings.append(embeddings.cpu().numpy())

        # Concatenate all embeddings
        final_embeddings = np.vstack(all_embeddings)

        return final_embeddings

    finally:
        # Cleanup temporary directory and all files (including feature_names.json)
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def _generate_feature_names(df: pd.DataFrame, output_file: Path, has_label: bool = True):
    """Generate feature_names.json from DataFrame."""
    feature_name_dict = {}

    # Determine which columns are features (skip label if present)
    feature_cols = df.columns[:-1] if has_label else df.columns
    
    for col in feature_cols:
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


def preprocess(
        input_dir: str,
        output_dir: str,
        target_col: Optional[str] = None,
        task_type: Optional[str] = None,
        device: Optional[str] = None,
) -> str:
    """
    Offline preprocessing: Convert TableData format to TP-BERTa format with embeddings.
    
    This function performs offline batch processing:
    1. Reads train/val/test CSV files
    2. Generates TP-BERTa embeddings for each row
    3. Saves feature_names.json (persistent, not cleaned up)
    4. Outputs train.csv, val.csv, test.csv with embeddings (2-column: embedding, target)
    
    Args:
        input_dir: Directory containing train.csv, val.csv, test.csv, and target_col.txt
        output_dir: Output directory for TP-BERTa format files
        target_col: Target column name (if None, read from target_col.txt)
        task_type: Task type (if None, read from target_col.txt)
        device: Device to use (default: "cuda" if available, else "cpu")
    
    Returns:
        Path to output directory
    
    Note:
        This is for offline batch processing. For runtime embedding extraction,
        use get_embeddings() instead.
    """
    import os
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get pretrain_dir from environment variable
    pretrain_dir = os.environ.get("TPBERTA_PRETRAIN_DIR")
    if pretrain_dir is None:
        raise ValueError(
            "TPBERTA_PRETRAIN_DIR environment variable not set. "
            "Please set TPBERTA_PRETRAIN_DIR to the path of pre-trained TP-BERTa model."
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load target column info from target_col.txt
    target_col_file = input_dir / "target_col.txt"
    if target_col is None or task_type is None:
        if not target_col_file.exists():
            raise FileNotFoundError(
                f"target_col.txt not found in {input_dir}. "
                f"Please provide target_col and task_type, or create target_col.txt"
            )

        with open(target_col_file, 'r') as f:
            lines = f.readlines()
            if target_col is None:
                target_col = lines[0].strip()
            if task_type is None:
                task_type = lines[1].strip() if len(lines) > 1 else "BINARY_CLASSIFICATION"

    # Map TaskType to TP-BERTa task type
    task_type_map = {
        "BINARY_CLASSIFICATION": "binclass",
        "REGRESSION": "regression",
        "MULTICLASS_CLASSIFICATION": "multiclass"
    }
    tpberta_task_type = task_type_map.get(task_type, "binclass")

    # Load all splits
    train_df = pd.read_csv(input_dir / "train.csv")
    val_df = pd.read_csv(input_dir / "val.csv")
    test_df = pd.read_csv(input_dir / "test.csv")

    # Verify target column exists
    if target_col not in train_df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in CSV. "
            f"Available columns: {list(train_df.columns)}"
        )

    # Make sure target column is last
    all_columns = [col for col in train_df.columns if col != target_col] + [target_col]
    train_df = train_df[all_columns]
    val_df = val_df[all_columns]
    test_df = test_df[all_columns]

    # Generate feature_names.json (needed for embedding generation)
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    feature_names_file = output_dir / "feature_names.json"
    _generate_feature_names(combined_df, feature_names_file)

    print(f"🔄 Generating TP-BERTa embeddings...")
    print(f"   Device: {device}")
    print(f"   Pretrain dir: {pretrain_dir}")

    # Generate embeddings for each split
    def process_split(df, split_name):
        """Process a single split and return embeddings + targets."""
        print(f"   Processing {split_name} split ({len(df)} rows)...")

        # Get embeddings
        embeddings = _get_tpberta_embeddings(
            df=df,
            pretrain_dir=pretrain_dir,
            feature_names_file=str(feature_names_file),
            device=device,
        )

        # Convert embeddings to comma-separated strings
        embedding_strings = []
        for emb in embeddings:
            emb_str = ",".join([str(x) for x in emb.flatten()])
            embedding_strings.append(emb_str)

        # Get targets
        targets = df[target_col].values

        return embedding_strings, targets

    # Process all splits separately
    train_embeddings, train_targets = process_split(train_df, "train")
    val_embeddings, val_targets = process_split(val_df, "val")
    test_embeddings, test_targets = process_split(test_df, "test")

    # Save each split as separate CSV file
    dataset_name = input_dir.name

    train_output_df = pd.DataFrame({
        'embedding': train_embeddings,
        'target': train_targets
    })
    train_csv = output_dir / "train.csv"
    train_output_df.to_csv(train_csv, index=False)

    val_output_df = pd.DataFrame({
        'embedding': val_embeddings,
        'target': val_targets
    })
    val_csv = output_dir / "val.csv"
    val_output_df.to_csv(val_csv, index=False)

    test_output_df = pd.DataFrame({
        'embedding': test_embeddings,
        'target': test_targets
    })
    test_csv = output_dir / "test.csv"
    test_output_df.to_csv(test_csv, index=False)

    print(f"✅ Converted to TP-BERTa format with embeddings:")
    print(f"   Input: {input_dir}")
    print(f"   Output: {output_dir}")
    print(f"   Train CSV: {train_csv} ({len(train_output_df)} rows, 2 columns: embedding, target)")
    print(f"   Val CSV:   {val_csv} ({len(val_output_df)} rows, 2 columns: embedding, target)")
    print(f"   Test CSV:  {test_csv} ({len(test_output_df)} rows, 2 columns: embedding, target)")
    print(f"   Feature names: {feature_names_file}")
    print(f"   Target column: {target_col}")
    print(f"   Task type: {tpberta_task_type}")
    print(f"   Total rows: {len(train_output_df) + len(val_output_df) + len(test_output_df)}")

    return str(train_csv)


def main():
    """Main function to convert TableData format to TP-BERTa format with embeddings."""

    parser = argparse.ArgumentParser(
        description="Convert TableData format to TP-BERTa format with embeddings"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing train.csv, val.csv, test.csv, and target_col.txt"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for TP-BERTa format files"
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default=None,
        help="Target column name (if not provided, read from target_col.txt)"
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default=None,
        help="Task type: BINARY_CLASSIFICATION, REGRESSION, or MULTICLASS_CLASSIFICATION "
             "(if not provided, read from target_col.txt)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto-detect)"
    )

    args = parser.parse_args()

    try:
        output_csv = preprocess(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            target_col=args.target_col,
            task_type=args.task_type,
            device=args.device,
        )
        print(f"\n✅ Success! Output CSV: {output_csv}")
        print(f"   Format: 2 columns (embedding, target)")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
