#!/usr/bin/env python3
"""
Test Runner for Basic Data Profile Parsing with Noise

Test script for running the complete data profiling pipeline on noised schemas.
This script adds noise to a database schema and tests if the pipeline can filter it out.

Unlike run_eval.py, this script does NOT calculate metrics - it just demonstrates
the filtering process by showing before/after schemas.

Usage:
    python -m aida.eval.basic_data_parse.run_test <db_name> <task_name> \\
        --provider <provider> [--noise-tables N] [--noise-columns N] [--verbose]

Examples:
    # Test with predefined task
    python -m aida.eval.basic_data_parse.run_test avito user-clicks \\
        --provider deepseek --verbose

    # Custom noise configuration
    python -m aida.eval.basic_data_parse.run_test hm customer-churn \\
        --provider openai --noise-tables 10 --noise-columns 5
"""

import argparse
import sys
import json
from typing import Optional

from utils.data.database_factory import DatabaseFactory
from utils.task.task_factory import TaskFactory
from aida.db.profile import DatabaseSchema, PredictionTaskProfile
from aida.query_analyzer import TableSelector, FeatureSelector
from aida.llm import LLMClientFactory
from .noise_generator import NoiseGenerator, NoiseConfig


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def print_section(title: str):
    """Print a section header."""
    print(f"\n{title}")
    print_separator("-", len(title))


def run_test(
    db_name: str,
    task_name: str,
    provider: str,
    model: Optional[str] = None,
    noise_tables: int = 5,
    noise_columns: int = 3,
    max_tables: int = 10,
    random_seed: int = 42,
    verbose: bool = False
) -> bool:
    """
    Run basic data parsing test with noised schema.

    Args:
        db_name: Database name
        task_name: Predefined task name
        provider: LLM provider
        model: Model name (optional)
        noise_tables: Number of noise tables to add
        noise_columns: Number of noise columns per table
        max_tables: Maximum number of tables to select
        random_seed: Random seed for noise generation
        verbose: Verbose output

    Returns:
        True if test completed successfully, False otherwise
    """
    # ============================================================
    # STEP 1: Load Database and Task
    # ============================================================
    print_section("STEP 1: LOAD DATABASE AND TASK")

    # Load database
    print(f"Loading database: {db_name}...")
    try:
        db = DatabaseFactory.get_db(db_name)
        clean_schema = DatabaseSchema.from_relbench_database(db, db_name)
        print(f"✓ Database loaded: {len(clean_schema.tables)} tables")
    except Exception as e:
        print(f"❌ Error loading database: {e}")
        return False

    # Load task
    print(f"Loading task: {task_name}...")
    try:
        task = TaskFactory.get_task(db_name, task_name)
        task_profile = PredictionTaskProfile.from_relbench_task(task, task_name)
        print(f"✓ Task loaded")
        print(f"  Task Type: {task_profile.task_type}")
        print(f"  Entity Table: {task_profile.entity_table}")
        print(f"  Target Column: {task_profile.target_column}")

        if verbose:
            print(f"\n  Description: {task_profile.description}")
    except Exception as e:
        print(f"❌ Error loading task: {e}")
        return False

    # ============================================================
    # STEP 2: Generate Noised Schema
    # ============================================================
    print_section("STEP 2: ADD NOISE TO SCHEMA")

    config = NoiseConfig(
        num_noise_tables=noise_tables,
        num_noise_columns_per_table=noise_columns,
        random_seed=random_seed
    )
    generator = NoiseGenerator(random_seed=random_seed)

    try:
        noised_schema = generator.add_noise(clean_schema, config)

        print(f"✓ Noise added successfully")
        print(f"\n  Before: {len(clean_schema.tables)} tables")
        print(f"  After:  {len(noised_schema.tables)} tables")

        # Calculate total columns
        clean_cols = sum(len(t.columns) for t in clean_schema.tables.values())
        noised_cols = sum(len(t.columns) for t in noised_schema.tables.values())
        print(f"\n  Before: {clean_cols} columns")
        print(f"  After:  {noised_cols} columns")

        if verbose:
            # Show noise tables
            noise_table_names = set(noised_schema.tables.keys()) - set(clean_schema.tables.keys())
            print(f"\n  Noise tables added ({len(noise_table_names)}):")
            for table in sorted(noise_table_names):
                print(f"    - {table}")

            # Show noise columns in original tables
            print(f"\n  Noise columns added to original tables:")
            for table_name in clean_schema.tables.keys():
                clean_cols = set(clean_schema.tables[table_name].columns)
                noised_cols = set(noised_schema.tables[table_name].columns)
                added_cols = noised_cols - clean_cols
                if added_cols:
                    print(f"    {table_name}: {sorted(added_cols)}")
    except Exception as e:
        print(f"❌ Error generating noise: {e}")
        return False

    # ============================================================
    # STEP 3: Initialize LLM Client
    # ============================================================
    print_section("STEP 3: INITIALIZE LLM CLIENT")

    print(f"Provider: {provider}")
    if model:
        print(f"Model: {model}")
    else:
        print(f"Model: [using provider default]")

    try:
        llm_client = LLMClientFactory.create(provider=provider, model=model)
        print("✓ LLM client initialized")
    except Exception as e:
        print(f"❌ Error creating LLM client: {e}")
        return False

    # ============================================================
    # STEP 4: Table Selection (Filter Noised Schema)
    # ============================================================
    print_section("STEP 4: TABLE SELECTION")

    # Convert to TaskProfile for operators
    task_profile_for_operators = task_profile.to_task_profile()

    table_selector = TableSelector(
        max_tables=max_tables,
        include_examples=True,
        focus_on_connectivity=True
    )

    try:
        table_result = table_selector(
            llm_client=llm_client,
            task_profile=task_profile_for_operators,
            db_schema=noised_schema
        )

        if not table_result.success:
            print("❌ Table selection failed!")
            for error in table_result.errors:
                print(f"  ❌ {error}")
            return False

        table_filtered_schema = table_result.db_schema
        selected_tables = list(table_filtered_schema.tables.keys())

        print(f"✓ Table selection completed")
        print(f"\n  Selected {len(selected_tables)} tables:")
        for table in selected_tables:
            is_noise = table.startswith("noise_")
            marker = "❌" if is_noise else "✅"
            print(f"    {marker} {table}")

        # Check if any noise tables leaked through
        noise_leaked = [t for t in selected_tables if t.startswith("noise_")]
        if noise_leaked:
            print(f"\n  ⚠️  WARNING: {len(noise_leaked)} noise table(s) leaked through!")
        else:
            print(f"\n  ✅ All noise tables filtered out!")

    except Exception as e:
        print(f"❌ Error during table selection: {e}")
        return False

    # ============================================================
    # STEP 5: Feature Selection (Filter Columns)
    # ============================================================
    print_section("STEP 5: FEATURE SELECTION")

    feature_selector = FeatureSelector(include_examples=True)

    try:
        feature_result = feature_selector(
            llm_client=llm_client,
            db_schema=table_filtered_schema,
            entity_table=task_profile.entity_table
        )

        if not feature_result.success:
            print("❌ Feature selection failed!")
            for error in feature_result.errors:
                print(f"  ❌ {error}")
            return False

        final_schema = feature_result.db_schema
        total_selected_cols = sum(len(t.columns) for t in final_schema.tables.values())

        print(f"✓ Feature selection completed")
        print(f"\n  Selected {total_selected_cols} columns across {len(final_schema.tables)} tables")

        if verbose:
            print(f"\n  Detailed breakdown:")
            for table_name, table in final_schema.tables.items():
                print(f"\n    {table_name} ({len(table.columns)} columns):")

                # Check if this table is in clean schema
                if table_name in clean_schema.tables:
                    clean_cols = set(clean_schema.tables[table_name].columns)
                    selected_cols = set(table.columns)

                    # Show which columns were kept
                    for col in table.columns:
                        is_original = col in clean_cols
                        marker = "✅" if is_original else "❌"
                        print(f"      {marker} {col}")
                else:
                    # Noise table that leaked through
                    for col in table.columns:
                        print(f"      ❌ {col}")

    except Exception as e:
        print(f"❌ Error during feature selection: {e}")
        return False

    # ============================================================
    # STEP 6: Summary
    # ============================================================
    print_section("SUMMARY")

    print(f"\n📊 Schema Evolution:")
    print(f"  Original:        {len(clean_schema.tables):2d} tables, {sum(len(t.columns) for t in clean_schema.tables.values()):3d} columns")
    print(f"  After Noise:     {len(noised_schema.tables):2d} tables, {sum(len(t.columns) for t in noised_schema.tables.values()):3d} columns")
    print(f"  After Filtering: {len(final_schema.tables):2d} tables, {sum(len(t.columns) for t in final_schema.tables.values()):3d} columns")

    # Calculate what was filtered
    tables_removed = len(noised_schema.tables) - len(final_schema.tables)
    cols_removed = sum(len(t.columns) for t in noised_schema.tables.values()) - sum(len(t.columns) for t in final_schema.tables.values())

    print(f"\n🔍 Filtering Results:")
    print(f"  Tables filtered: {tables_removed}")
    print(f"  Columns filtered: {cols_removed}")

    return True


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Test basic data parsing with noised schema",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "db_name",
        type=str,
        help="Database name (e.g., 'avito', 'hm', 'donor')"
    )

    parser.add_argument(
        "task_name",
        type=str,
        help="Predefined task name (e.g., 'user-clicks', 'customer-churn')"
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="deepseek",
        choices=["openai", "anthropic", "ollama", "deepseek"],
        help="LLM provider (default: deepseek)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (optional, uses provider default if not specified)"
    )

    parser.add_argument(
        "--noise-tables",
        type=int,
        default=5,
        help="Number of noise tables to add (default: 5)"
    )

    parser.add_argument(
        "--noise-columns",
        type=int,
        default=3,
        help="Number of noise columns per table (default: 3)"
    )

    parser.add_argument(
        "--max-tables",
        type=int,
        default=10,
        help="Maximum number of tables to select (default: 10)"
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for noise generation (default: 42)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose mode: show detailed output"
    )

    args = parser.parse_args()

    # Print header
    print_separator("=", 80)
    print("Basic Data Profile Parsing - Test Runner".center(80))
    print_separator("=", 80)

    # Run test
    success = run_test(
        db_name=args.db_name,
        task_name=args.task_name,
        provider=args.provider,
        model=args.model,
        noise_tables=args.noise_tables,
        noise_columns=args.noise_columns,
        max_tables=args.max_tables,
        random_seed=args.random_seed,
        verbose=args.verbose
    )

    # Exit with appropriate code
    if success:
        print_separator("=", 80)
        print("✅ TEST COMPLETED SUCCESSFULLY".center(80))
        print_separator("=", 80)
        sys.exit(0)
    else:
        print_separator("=", 80)
        print("❌ TEST FAILED".center(80))
        print_separator("=", 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
