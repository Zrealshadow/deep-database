#!/usr/bin/env python3
"""
Metrics Calculation Module for Basic Data Profiling Evaluation

Provides unified functions for calculating evaluation metrics (precision, recall, F1)
at both column and table levels. This eliminates code duplication between run_eval.py
and run_test.py.

Usage:
    from aida.eval.basic_data_parse.metrics import calculate_column_metrics

    metrics = calculate_column_metrics(
        predicted_schema=filtered_schema,
        ground_truth_schema=clean_schema,
        noised_schema=noised_schema  # optional
    )

    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1: {metrics['f1']:.3f}")
"""

from typing import Dict, List, Optional
from aida.db.profile import DatabaseSchema


def calculate_column_metrics(
    predicted_schema: DatabaseSchema,
    ground_truth_schema: DatabaseSchema,
    noised_schema: Optional[DatabaseSchema] = None
) -> Dict:
    """
    Calculate column-level metrics (precision, recall, F1).

    Compares the predicted schema against ground truth at the column level.
    If noised_schema is provided, also calculates true negatives (noise correctly filtered).

    Args:
        predicted_schema: Schema after operator filtering
        ground_truth_schema: Original clean schema (ground truth)
        noised_schema: Schema with noise added (optional, for TN calculation)

    Returns:
        Dictionary with:
        - precision: float (0-1)
        - recall: float (0-1)
        - f1: float (0-1)
        - true_positives: int
        - false_positives: int
        - false_negatives: int
        - true_negatives: int (only if noised_schema provided)
        - predicted_count: int
        - ground_truth_count: int
        - true_positive_columns: Set[str] (format: "table.column")
        - false_positive_columns: Set[str]
        - false_negative_columns: Set[str]
        - true_negative_columns: Set[str] (only if noised_schema provided)
    """
    # Extract ground truth columns
    gt_columns = set()
    for table_name, table in ground_truth_schema.tables.items():
        for col in table.columns:
            gt_columns.add(f"{table_name}.{col}")

    # Extract noise columns (if noised schema provided)
    noise_columns = set()
    if noised_schema:
        for table_name, table in noised_schema.tables.items():
            if table_name in ground_truth_schema.tables:
                # Noise columns in original tables
                clean_cols = set(ground_truth_schema.tables[table_name].columns)
                for col in table.columns:
                    if col not in clean_cols:
                        noise_columns.add(f"{table_name}.{col}")
            else:
                # All columns in noise tables
                for col in table.columns:
                    noise_columns.add(f"{table_name}.{col}")

    # Extract selected columns from predicted schema
    selected_columns = set()
    for table_name, table in predicted_schema.tables.items():
        for col in table.columns:
            selected_columns.add(f"{table_name}.{col}")

    # Calculate confusion matrix
    true_positives = gt_columns & selected_columns  # Ground truth columns kept
    false_negatives = gt_columns - selected_columns  # Ground truth columns lost
    false_positives = noise_columns & selected_columns if noise_columns else (selected_columns - gt_columns)
    true_negatives = noise_columns - selected_columns if noise_columns else set()

    # Calculate metrics
    precision = len(true_positives) / len(selected_columns) if selected_columns else 0.0
    recall = len(true_positives) / len(gt_columns) if gt_columns else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    result = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': len(true_positives),
        'false_positives': len(false_positives),
        'false_negatives': len(false_negatives),
        'predicted_count': len(selected_columns),
        'ground_truth_count': len(gt_columns),
        'true_positive_columns': true_positives,
        'false_positive_columns': false_positives,
        'false_negative_columns': false_negatives,
    }

    if noise_columns:
        result['true_negatives'] = len(true_negatives)
        result['true_negative_columns'] = true_negatives

    return result


def calculate_table_metrics(
    predicted_schema: DatabaseSchema,
    ground_truth_schema: DatabaseSchema
) -> Dict:
    """
    Calculate table-level metrics (precision, recall, F1).

    Compares the predicted schema against ground truth at the table level.

    Args:
        predicted_schema: Schema after operator filtering
        ground_truth_schema: Original clean schema (ground truth)

    Returns:
        Dictionary with:
        - precision: float (0-1)
        - recall: float (0-1)
        - f1: float (0-1)
        - true_positives: int
        - false_positives: int
        - false_negatives: int
        - predicted_count: int
        - ground_truth_count: int
        - true_positive_tables: Set[str]
        - false_positive_tables: Set[str]
        - false_negative_tables: Set[str]
    """
    predicted_tables = set(predicted_schema.tables.keys())
    ground_truth_tables = set(ground_truth_schema.tables.keys())

    true_positives = predicted_tables & ground_truth_tables
    false_positives = predicted_tables - ground_truth_tables
    false_negatives = ground_truth_tables - predicted_tables

    precision = len(true_positives) / len(predicted_tables) if predicted_tables else 0.0
    recall = len(true_positives) / len(ground_truth_tables) if ground_truth_tables else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': len(true_positives),
        'false_positives': len(false_positives),
        'false_negatives': len(false_negatives),
        'predicted_count': len(predicted_tables),
        'ground_truth_count': len(ground_truth_tables),
        'true_positive_tables': true_positives,
        'false_positive_tables': false_positives,
        'false_negative_tables': false_negatives,
    }


def aggregate_metrics(results_list: List[Dict]) -> Dict:
    """
    Aggregate metrics across multiple evaluations.

    Computes mean, std, min, max for precision, recall, and F1 scores.
    Used for multi-database evaluation summaries.

    Args:
        results_list: List of metric dictionaries (from calculate_column_metrics or calculate_table_metrics)

    Returns:
        Dictionary with aggregate statistics:
        - precision: {'mean': float, 'std': float, 'min': float, 'max': float}
        - recall: {'mean': float, 'std': float, 'min': float, 'max': float}
        - f1: {'mean': float, 'std': float, 'min': float, 'max': float}
        - num_evaluations: int
    """
    if not results_list:
        return {
            'precision': {'mean': 0, 'std': 0, 'min': 0, 'max': 0},
            'recall': {'mean': 0, 'std': 0, 'min': 0, 'max': 0},
            'f1': {'mean': 0, 'std': 0, 'min': 0, 'max': 0},
            'num_evaluations': 0
        }

    # Extract metric values
    precisions = [r['precision'] for r in results_list]
    recalls = [r['recall'] for r in results_list]
    f1s = [r['f1'] for r in results_list]

    def compute_stats(values: List[float]) -> Dict[str, float]:
        """Compute statistics for a list of values."""
        if not values:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}

        mean_val = sum(values) / len(values)

        # Calculate standard deviation
        if len(values) > 1:
            variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
            std_val = variance ** 0.5
        else:
            std_val = 0.0

        return {
            'mean': mean_val,
            'std': std_val,
            'min': min(values),
            'max': max(values)
        }

    return {
        'precision': compute_stats(precisions),
        'recall': compute_stats(recalls),
        'f1': compute_stats(f1s),
        'num_evaluations': len(results_list)
    }
