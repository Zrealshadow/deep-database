"""
Unified Text Embedding Module

Supports multiple embedding models:
- TP-BERTa (table-specific)
- nomic-embed-text-v1.5
- bge-base-en-v1.5
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import Optional, Union
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Import TP-BERTa function
from tpberta.preprocess import _get_tpberta_embeddings


def _dataframe_to_texts(
    df: pd.DataFrame,
    format: str = "key_value",
    prefix: Optional[str] = None,
    feature_names_map: Optional[dict] = None,
) -> list:
    """
    Convert DataFrame rows to text strings.
    
    Args:
        df: Input DataFrame
        format: Text format style
            - "key_value": "column1: value1, column2: value2, ..."
            - "simple": "value1 value2 value3 ..."
            - "json_like": "{'column1': 'value1', 'column2': 'value2', ...}"
        prefix: Optional prefix to add to each text (e.g., "search_document:")
        feature_names_map: Optional dict mapping original column names to standardized names
                          (e.g., from feature_names.json: {"BloodPressure": "blood pressure"})
                          If provided, uses standardized names as keys instead of original column names.
    
    Returns:
        List of text strings, one per row
    """
    texts = []
    
    for _, row in df.iterrows():
        if format == "key_value":
            # Format: "column1: value1, column2: value2, ..."
            parts = []
            for col, val in row.items():
                if pd.notna(val):
                    # Use standardized feature name if available, otherwise use original column name
                    key = feature_names_map.get(col, col) if feature_names_map else col
                    parts.append(f"{key}: {val}")
            text = ", ".join(parts)
        elif format == "simple":
            # Format: "value1 value2 value3 ..."
            parts = [str(val) for val in row.values if pd.notna(val)]
            text = " ".join(parts)
        elif format == "json_like":
            # Format: "{'column1': 'value1', 'column2': 'value2', ...}"
            parts = []
            for col, val in row.items():
                if pd.notna(val):
                    # Use standardized feature name if available
                    key = feature_names_map.get(col, col) if feature_names_map else col
                    parts.append(f"'{key}': '{val}'")
            text = "{" + ", ".join(parts) + "}"
        else:
            raise ValueError(f"Unknown format: {format}")
        
        # Add prefix if specified
        if prefix:
            text = f"{prefix} {text}"
        
        texts.append(text)
    
    return texts


def _get_nomic_embeddings(
    df: pd.DataFrame,
    task_prefix: str = "search_document",
    text_format: str = "key_value",
    batch_size: int = 32,
    device: Optional[Union[str, torch.device]] = None,
    normalize: bool = True,
    matryoshka_dim: Optional[int] = None,
    feature_names_file: Optional[str] = None,
) -> np.ndarray:
    """
    Get embeddings using nomic-embed-text-v1.5 model.
    
    Args:
        df: Input DataFrame
        task_prefix: Task instruction prefix for nomic model
            Options: "search_document", "search_query", "clustering", "classification"
        text_format: Format for converting DataFrame to text (see _dataframe_to_texts)
        batch_size: Batch size for encoding
        device: Device to use ("cuda", "cpu", or torch.device)
        normalize: Whether to normalize embeddings
        matryoshka_dim: Optional dimension for Matryoshka representation (768, 512, 256, 128, 64)
        feature_names_file: Optional path to feature_names.json. If provided, uses standardized
                           feature names as keys instead of original column names.
    
    Returns:
        numpy array of embeddings [N, embedding_dim]
    """
    # Convert device string to torch.device if needed
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    
    # Load feature names map if provided
    feature_names_map = None
    if feature_names_file and Path(feature_names_file).exists():
        with open(feature_names_file, 'r') as f:
            feature_names_map = json.load(f)
    
    # Load model
    model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
        device=str(device)
    )
    
    # Convert DataFrame to texts with prefix
    prefix = f"{task_prefix}:"
    texts = _dataframe_to_texts(df, format=text_format, prefix=prefix, feature_names_map=feature_names_map)
    
    # Encode in batches
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = model.encode(
            batch_texts,
            convert_to_tensor=True,
            batch_size=batch_size,
            show_progress_bar=False
        )
        all_embeddings.append(batch_embeddings)
    
    # Concatenate all batches
    embeddings = torch.cat(all_embeddings, dim=0)
    
    # Apply Matryoshka representation learning if specified
    if matryoshka_dim is not None:
        # Layer normalization
        embeddings = F.layer_norm(
            embeddings,
            normalized_shape=(embeddings.shape[1],)
        )
        # Truncate to specified dimension
        embeddings = embeddings[:, :matryoshka_dim]
    
    # Normalize if requested
    if normalize:
        embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Convert to numpy
    embeddings = embeddings.cpu().numpy()
    
    return embeddings


def _get_bge_embeddings(
    df: pd.DataFrame,
    text_format: str = "key_value",
    batch_size: int = 32,
    device: Optional[Union[str, torch.device]] = None,
    normalize: bool = True,
    feature_names_file: Optional[str] = None,
) -> np.ndarray:
    """
    Get embeddings using bge-base-en-v1.5 model.
    
    Args:
        df: Input DataFrame
        text_format: Format for converting DataFrame to text (see _dataframe_to_texts)
        batch_size: Batch size for encoding
        device: Device to use ("cuda", "cpu", or torch.device)
        normalize: Whether to normalize embeddings
        feature_names_file: Optional path to feature_names.json. If provided, uses standardized
                           feature names as keys instead of original column names.
    
    Returns:
        numpy array of embeddings [N, embedding_dim]
    """
    # Convert device string to torch.device if needed
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    
    # Load feature names map if provided
    feature_names_map = None
    if feature_names_file and Path(feature_names_file).exists():
        with open(feature_names_file, 'r') as f:
            feature_names_map = json.load(f)
    
    # Load model
    model = SentenceTransformer(
        "BAAI/bge-base-en-v1.5",
        device=str(device)
    )
    
    # Convert DataFrame to texts (no prefix needed for BGE)
    texts = _dataframe_to_texts(df, format=text_format, prefix=None, feature_names_map=feature_names_map)
    
    # Encode in batches
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = model.encode(
            batch_texts,
            convert_to_tensor=True,
            batch_size=batch_size,
            show_progress_bar=False
        )
        all_embeddings.append(batch_embeddings)
    
    # Concatenate all batches
    embeddings = torch.cat(all_embeddings, dim=0)
    
    # Normalize if requested
    if normalize:
        embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Convert to numpy
    embeddings = embeddings.cpu().numpy()
    
    return embeddings


def get_embeddings(
    df: pd.DataFrame,
    model: str = "tpberta",
    dataset_name: Optional[str] = None,
    pretrain_dir: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    has_label: bool = False,
    # TP-BERTa specific
    feature_names_file: Optional[str] = None,
    # Text model specific
    text_format: str = "key_value",
    batch_size: int = 32,
    # Nomic specific
    task_prefix: str = "search_document",
    matryoshka_dim: Optional[int] = None,
    # BGE specific (no additional params needed)
) -> np.ndarray:
    """
    Get embeddings from DataFrame (runtime function, returns numpy array).
    
    This is the main interface for runtime embedding extraction. It automatically handles
    feature_names.json generation and cleanup for TP-BERTa. For nomic/bge, it can optionally
    use standardized feature names.
    
    Args:
        df: Input DataFrame
        model: Model to use ("tpberta", "nomic", or "bge")
        dataset_name: Optional dataset name for feature_names.json generation (auto-generated if None)
        pretrain_dir: Path to TP-BERTa pretrained model (required for "tpberta")
        device: Device to use ("cuda", "cpu", or torch.device)
        has_label: Whether DataFrame has a label column (for TP-BERTa)
        feature_names_file: Optional path to feature_names.json. If None, will auto-generate
                           for TP-BERTa (temporary file, cleaned up after use).
                           For nomic/bge, if provided, uses standardized feature names as keys.
        text_format: Text format for text models ("key_value", "simple", "json_like")
        batch_size: Batch size for text models
        task_prefix: Task prefix for nomic model ("search_document", "search_query", etc.)
        matryoshka_dim: Optional dimension for nomic Matryoshka representation
    
    Returns:
        numpy array of embeddings [N, embedding_dim]
    
    Examples:
        # TP-BERTa (auto-generates and cleans up feature_names.json)
        embeddings = get_embeddings(df, model="tpberta", pretrain_dir="...")
        
        # Nomic
        embeddings = get_embeddings(df, model="nomic", task_prefix="search_document")
        
        # BGE
        embeddings = get_embeddings(df, model="bge")
    """
    import tempfile
    import hashlib
    
    # Auto-generate dataset_name if not provided
    if dataset_name is None:
        # Generate a unique name based on DataFrame columns and shape
        df_hash = hashlib.md5(str(df.columns.tolist()).encode()).hexdigest()[:8]
        dataset_name = f"dataset_{df_hash}"
    
    if model.lower() == "tpberta":
        if pretrain_dir is None:
            raise ValueError("pretrain_dir is required for TP-BERTa model")
        
        # For TP-BERTa, auto-generate feature_names.json if not provided
        # It will be created in temp dir and cleaned up automatically
        return _get_tpberta_embeddings(
            df=df,
            pretrain_dir=pretrain_dir,
            feature_names_file=feature_names_file,
            device=device,
            has_label=has_label,
            dataset_name=dataset_name,
        )
    elif model.lower() == "nomic":
        return _get_nomic_embeddings(
            df=df,
            task_prefix=task_prefix,
            text_format=text_format,
            batch_size=batch_size,
            device=device,
            normalize=True,
            matryoshka_dim=matryoshka_dim,
            feature_names_file=feature_names_file,
        )
    elif model.lower() == "bge":
        return _get_bge_embeddings(
            df=df,
            text_format=text_format,
            batch_size=batch_size,
            device=device,
            normalize=True,
            feature_names_file=feature_names_file,
        )
    else:
        raise ValueError(
            f"Unknown model: {model}. "
            "Supported models: 'tpberta', 'nomic', 'bge'"
        )

