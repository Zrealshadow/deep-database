#!/usr/bin/env python3
"""
Evaluation Dataset Generator

Generates evaluation datasets for basic data profiling by:
1. Auto-discovering all available (database, task) pairs from DatabaseFactory
2. Generating multiple noised versions per database with different noise configurations
3. Each evaluation instance: (db_name, task_name, clean_schema, noised_schema, noise_config)

The dataset generation is deterministic (controlled by random seeds) and can be
regenerated on-the-fly without serialization.

Usage:
    # Generate and iterate over dataset
    from aida.eval.basic_data_parse.dataset import generate_evaluation_dataset

    dataset = generate_evaluation_dataset(
        num_noise_variants=3,  # 3 different noise configs per database
        base_seed=42
    )

    for instance in dataset:
        clean = instance['clean_schema']
        noised = instance['noised_schema']
        # Run evaluation...
"""

import random
from typing import List, Tuple
from .noise_generator import NoiseConfig


def get_all_registered_database_task_pairs() -> List[Tuple[str, str]]:
    """
    Get all registered (database, task) pairs from DatabaseFactory.

    Returns:
        List of (db_name, task_name) tuples
    """
    from utils.data.database_factory import DatabaseFactory

    pairs = []
    for db_name, tasks in DatabaseFactory._task_registry.items():
        for task_name in tasks.keys():
            pairs.append((db_name, task_name))

    return pairs


# Benchmark configuration - all registered (database, task) pairs
# EVALUATION_BENCHMARK: List[Tuple[str, str]] = get_all_registered_database_task_pairs()
EVALUATION_BENCHMARK: List[Tuple[str, str]] = [
    ("avito", "user-clicks"),
    ("avito", "ad-ctr"),
    ("event", "user-ignore"),
    ("event", "user-repeat"),
    ("event", "user-attendance"),
    ("hm", "user-churn"),
    ("hm", "item-sales"),
    ("amazon", "user-churn"),
    ("amazon", "item-churn"),
    ("f1", "driver-dnf"),
    ("f1", "driver-top3"),
    ("trial", "study-outcome"),
    ("trial", "site-success"),
    ("olist", "order-delay")
]

# Difficulty level definitions
DIFFICULTY_LEVELS = {
    'A': {
        'name': 'Simple',
        'num_noise_tables': (1, 3),  # Sample from [1, 2, 3]
        'noise_columns_range': (0, 2),  # Sample from [0, 1, 2]
        'linking_strategy': 'none',  # No linking
        'description': '1-3 unlinked noise tables, 0-2 noise columns per existing table'
    },
    'B': {
        'name': 'Medium',
        'num_noise_tables': (3, 5),  # Sample from [3, 4, 5]
        'noise_columns_range': (0, 2),  # Sample from [0, 1, 2]
        'linking_strategy': 'random',  # Random linking
        'description': '3-5 noise tables with random linking, 0-2 noise columns'
    },
    'C': {
        'name': 'Challenge',
        'num_noise_tables': (4, 6),  # Sample from [4, 5, 6]
        'noise_columns_range': (2, 5),  # Sample from [2, 3, 4, 5]
        'linking_strategy': 'all',  # All linked
        'description': '4-6 fully-linked noise tables, 2-5 noise columns per table'
    }
}


def get_noise_config(difficulty_level: str, random_seed: int = 42) -> NoiseConfig:
    """
    Get noise configuration for a specific difficulty level.

    Args:
        difficulty_level: Difficulty level ("A", "B", or "C")
        random_seed: Random seed for reproducibility

    Returns:
        NoiseConfig object configured for the specified difficulty

    Raises:
        ValueError: If difficulty_level is not A, B, or C
    """
    if difficulty_level not in DIFFICULTY_LEVELS:
        raise ValueError(f"Unknown difficulty: {difficulty_level}. Choose from A, B, C")

    config = DIFFICULTY_LEVELS[difficulty_level]

    # Sample num_noise_tables from the defined range
    rng = random.Random(random_seed)
    num_tables_range = config['num_noise_tables']
    num_tables = rng.randint(num_tables_range[0], num_tables_range[1])

    return NoiseConfig(
        num_noise_tables=num_tables,
        noise_columns_range=config['noise_columns_range'],
        linking_strategy=config['linking_strategy'],
        random_seed=random_seed,
        difficulty_level=difficulty_level
    )




