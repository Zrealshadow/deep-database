"""
TP-BERTa Package

This package provides:
1. Preprocessing: Convert CSV rows to embedding strings
2. Training: Train prediction head on preprocessed embedding data
"""

from .preprocess import process_csv_rows_to_embeddings
from .train import train_prediction_head

__all__ = ['process_csv_rows_to_embeddings', 'train_prediction_head']

