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
from typing import List, Dict
from dataclasses import dataclass

from utils.data.database_factory import DatabaseFactory
from aida.db.profile import DatabaseSchema
from .noise_generator import NoiseGenerator, NoiseConfig


@dataclass
class EvaluationInstance:
    """A single evaluation instance."""
    db_name: str
    task_name: str
    clean_schema: DatabaseSchema
    noised_schema: DatabaseSchema
    noise_config: NoiseConfig
    instance_id: str  # Format: "dbname_taskname_variant{i}"


def get_all_database_task_pairs() -> List[tuple]:
    """
    Get all available (database, task) pairs from DatabaseFactory.

    Returns:
        List of (db_name, task_name) tuples
    """
    # Get all available databases
    available_databases = DatabaseFactory.get_available_databases()

    pairs = []
    for db_name in available_databases:
        try:
            # Get default task for this database
            # You can extend this to get all tasks if DatabaseFactory supports it
            task_name = DatabaseFactory.get_default_task_name(db_name)
            pairs.append((db_name, task_name))
        except Exception as e:
            print(f"Warning: Could not get task for {db_name}: {e}")
            continue

    return pairs


# Difficulty level definitions
DIFFICULTY_LEVELS = {
    'A': {
        'name': 'Simple',
        'num_noise_tables': 2,
        'noise_columns_range': (0, 2),  # Sample from [0, 1, 2]
        'linking_strategy': 'none',  # No linking
        'description': '2 unlinked noise tables, 0-2 noise columns per existing table'
    },
    'B': {
        'name': 'Medium',
        'num_noise_tables': 3,  # Will be randomized to 2-4
        'noise_columns_range': (0, 2),  # Sample from [0, 1, 2]
        'linking_strategy': 'random',  # Random linking
        'description': '2-4 noise tables with random linking, 0-2 noise columns'
    },
    'C': {
        'name': 'Challenge',
        'num_noise_tables': 5,
        'noise_columns_range': (2, 5),  # Sample from [2, 3, 4, 5]
        'linking_strategy': 'all',  # All linked
        'description': '5 fully-linked noise tables, 2-5 noise columns per table'
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

    # For level B, randomly choose 2-4 tables
    num_tables = config['num_noise_tables']
    if difficulty_level == 'B':
        rng = random.Random(random_seed)
        num_tables = rng.randint(2, 4)

    return NoiseConfig(
        num_noise_tables=num_tables,
        noise_columns_range=config['noise_columns_range'],
        linking_strategy=config['linking_strategy'],
        random_seed=random_seed,
        difficulty_level=difficulty_level
    )


def get_all_difficulty_configs(base_seed: int = 42) -> List[NoiseConfig]:
    """
    Get configs for all difficulty levels A, B, C.

    Args:
        base_seed: Base random seed (each level gets base_seed + offset)

    Returns:
        List of 3 NoiseConfig objects (one for each difficulty level)
    """
    return [
        get_noise_config('A', base_seed),
        get_noise_config('B', base_seed + 1),
        get_noise_config('C', base_seed + 2)
    ]


def generate_evaluation_dataset(
    database_task_pairs: List[tuple] = None,
    num_noise_variants: int = 3,
    base_seed: int = 42,
    verbose: bool = False
) -> List[EvaluationInstance]:
    """
    Generate complete evaluation dataset.

    Args:
        database_task_pairs: List of (db_name, task_name) tuples.
                           If None, auto-discovers all available databases.
        num_noise_variants: Number of noise variants per database
        base_seed: Base random seed for reproducibility
        verbose: Print progress

    Returns:
        List of EvaluationInstance objects
    """
    # Auto-discover databases if not provided
    if database_task_pairs is None:
        if verbose:
            print("Auto-discovering databases...")
        database_task_pairs = get_all_database_task_pairs()
        if verbose:
            print(f"  Found {len(database_task_pairs)} database-task pairs")

    # Generate noise configurations (one per difficulty level, capped at num_variants)
    noise_configs = get_all_difficulty_configs(base_seed)[:num_noise_variants]

    if verbose:
        print(f"\nGenerating evaluation dataset...")
        print(f"  Databases: {len(database_task_pairs)}")
        print(f"  Noise variants per database: {num_noise_variants}")
        print(f"  Total instances: {len(database_task_pairs) * num_noise_variants}")

    dataset = []

    for db_idx, (db_name, task_name) in enumerate(database_task_pairs, 1):
        if verbose:
            print(f"\n[{db_idx}/{len(database_task_pairs)}] Processing {db_name} ({task_name})...")

        try:
            # Load clean schema (ground truth)
            db = DatabaseFactory.get_db(db_name)
            clean_schema = DatabaseSchema.from_relbench_database(db, db_name)

            if verbose:
                clean_tables = len(clean_schema.tables)
                clean_cols = sum(len(t.columns) for t in clean_schema.tables.values())
                print(f"  Clean schema: {clean_tables} tables, {clean_cols} columns")

            # Generate multiple noised versions
            for variant_idx, noise_config in enumerate(noise_configs, 1):
                # Generate noised schema
                generator = NoiseGenerator(random_seed=noise_config.random_seed)
                noised_schema = generator.add_noise(clean_schema, noise_config)

                # Create instance
                instance = EvaluationInstance(
                    db_name=db_name,
                    task_name=task_name,
                    clean_schema=clean_schema,
                    noised_schema=noised_schema,
                    noise_config=noise_config,
                    instance_id=f"{db_name}_{task_name}_v{variant_idx}"
                )

                dataset.append(instance)

                if verbose:
                    noised_tables = len(noised_schema.tables)
                    noised_cols = sum(len(t.columns) for t in noised_schema.tables.values())
                    print(f"    Variant {variant_idx}: {noised_tables} tables, {noised_cols} columns "
                          f"(+{noised_tables - clean_tables} tables, +{noised_cols - clean_cols} cols)")

        except Exception as e:
            print(f"  ❌ Error processing {db_name}: {e}")
            continue

    if verbose:
        print(f"\n✅ Dataset generated: {len(dataset)} instances")

    return dataset


def get_benchmark_dataset(
    benchmark_name: str = "default",
    base_seed: int = 42,
    verbose: bool = False
) -> List[EvaluationInstance]:
    """
    Get a predefined benchmark dataset configuration.

    Args:
        benchmark_name: Name of benchmark configuration
                       "default" - Standard benchmark with medium noise
                       "small" - Small quick benchmark for testing
                       "full" - All databases with multiple noise levels
        base_seed: Base random seed
        verbose: Print progress

    Returns:
        List of EvaluationInstance objects
    """
    if benchmark_name == "small":
        # Quick benchmark for testing
        pairs = [
            ('avito', 'user-clicks'),
            ('hm', 'customer-churn'),
        ]
        num_variants = 1

    elif benchmark_name == "default":
        # Standard benchmark
        pairs = [
            ('avito', 'user-clicks'),
            ('hm', 'customer-churn'),
            ('donor', 'donor-return'),
        ]
        num_variants = 3

    elif benchmark_name == "full":
        # All available databases
        pairs = None  # Auto-discover
        num_variants = 5

    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

    return generate_evaluation_dataset(
        database_task_pairs=pairs,
        num_noise_variants=num_variants,
        base_seed=base_seed,
        verbose=verbose
    )


def print_dataset_summary(dataset: List[EvaluationInstance]):
    """Print summary statistics of the dataset."""
    print(f"\n{'='*80}")
    print(f"Evaluation Dataset Summary")
    print(f"{'='*80}")

    print(f"\nTotal instances: {len(dataset)}")

    # Group by database
    db_counts = {}
    for instance in dataset:
        db_counts[instance.db_name] = db_counts.get(instance.db_name, 0) + 1

    print(f"\nInstances per database:")
    for db_name, count in sorted(db_counts.items()):
        print(f"  {db_name}: {count} variants")

    # Noise statistics
    print(f"\nNoise configurations:")
    noise_tables = [inst.noise_config.num_noise_tables for inst in dataset]
    print(f"  Noise tables: {min(noise_tables)}-{max(noise_tables)} (avg: {sum(noise_tables)/len(noise_tables):.1f})")

    # Show column noise info per unique difficulty level
    seen_levels = set()
    for inst in dataset:
        config = inst.noise_config
        level = config.difficulty_level or 'custom'
        if level not in seen_levels:
            seen_levels.add(level)
            if config.noise_columns_range is not None:
                print(f"  Noise columns range (level {level}): {config.noise_columns_range}")
            else:
                print(f"  Noise columns/table (level {level}): {config.num_noise_columns_per_table}")

    # Schema size statistics
    total_clean_cols = sum(
        sum(len(t.columns) for t in inst.clean_schema.tables.values())
        for inst in dataset
    ) / len(dataset)

    total_noised_cols = sum(
        sum(len(t.columns) for t in inst.noised_schema.tables.values())
        for inst in dataset
    ) / len(dataset)

    print(f"\nAverage schema size:")
    print(f"  Clean: {total_clean_cols:.0f} columns")
    print(f"  Noised: {total_noised_cols:.0f} columns (+{total_noised_cols - total_clean_cols:.0f})")

    print(f"{'='*80}\n")


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate and inspect evaluation dataset")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="default",
        choices=["small", "default", "full"],
        help="Benchmark configuration"
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base random seed"
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    # Generate dataset
    dataset = get_benchmark_dataset(
        benchmark_name=args.benchmark,
        base_seed=args.base_seed,
        verbose=args.verbose
    )

    # Print summary
    print_dataset_summary(dataset)

    # Show first few instances
    print("Sample instances:")
    for inst in dataset[:3]:
        print(f"  {inst.instance_id}: "
              f"{len(inst.clean_schema.tables)} → {len(inst.noised_schema.tables)} tables")
