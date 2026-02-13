#!/usr/bin/env python3
"""
Benchmark Evaluation for Basic Data Profile Parsing

Evaluates the complete data profiling pipeline across multiple databases by:
1. Loading ground truth schema (clean database schema)
2. Adding noise (irrelevant tables and columns) - deterministic with fixed seed
3. Running the full parsing pipeline (task parsing → table selection → feature selection)
4. Comparing results with ground truth at column level
5. Aggregating metrics across multiple databases

Usage:
    # Run benchmark with specific difficulty level
    python -m aida.eval.basic_data_parse.run_eval --provider deepseek --difficulty B

    # Run all difficulty levels
    python -m aida.eval.basic_data_parse.run_eval --provider deepseek --difficulty all

Examples:
    # Run benchmark with Level C (challenge)
    python -m aida.eval.basic_data_parse.run_eval --provider deepseek --difficulty C --verbose

    # Run all difficulty levels
    python -m aida.eval.basic_data_parse.run_eval --provider openai --difficulty all
"""

import argparse
import sys
from typing import Dict, List, Optional

from utils.data.database_factory import DatabaseFactory
from aida.db.profile import DatabaseSchema, PredictionTaskProfile
from aida.query_analyzer import TableSelector, FeatureSelector
from aida.llm import LLMClientFactory
from .noise_generator import NoiseGenerator
from .dataset import get_noise_config, DIFFICULTY_LEVELS
from .metrics import calculate_column_metrics, aggregate_metrics


# Benchmark configuration
EVALUATION_BENCHMARK = {
    'databases': [
        ('avito', 'user-clicks'),
        ('hm', 'customer-churn'),
        ('donor', 'donor-return'),
    ],
    'difficulty_level': 'B',  # Default to medium difficulty
    'random_seed': 42
}


def print_separator(char="=", length=100):
    """Print a separator line."""
    print(char * length)


def print_section(title: str):
    """Print a section header."""
    print(f"\n{title}")
    print_separator("-", len(title))


def run_single_eval(
    db_name: str,
    task_name: str,
    llm_client,
    noise_config,
    max_tables: int = 10,
    verbose: bool = False
) -> Optional[Dict]:
    """
    Run evaluation on a single database.

    Args:
        db_name: Database name
        task_name: Task name
        llm_client: LLM client instance
        noise_config: Noise configuration
        max_tables: Maximum number of tables to select
        verbose: Verbose output

    Returns:
        Dictionary with evaluation results, or None if failed
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Evaluating: {db_name} ({task_name})")
        print(f"{'='*80}")

    try:
        # ============================================================
        # Step 1: Load Ground Truth Schema
        # ============================================================
        if verbose:
            print("\n[1/5] Loading ground truth schema...")

        db = DatabaseFactory.get_db(db_name)
        ground_truth_schema = DatabaseSchema.from_relbench_database(db, db_name)

        if verbose:
            print(f"  ✓ Loaded: {len(ground_truth_schema.tables)} tables, "
                  f"{sum(len(t.columns) for t in ground_truth_schema.tables.values())} columns")

        # ============================================================
        # Step 2: Generate Noised Schema
        # ============================================================
        if verbose:
            print("\n[2/5] Generating noised schema...")

        generator = NoiseGenerator(random_seed=noise_config.random_seed)
        noised_schema = generator.add_noise(ground_truth_schema, noise_config)

        if verbose:
            print(f"  ✓ Added noise: {len(noised_schema.tables)} tables, "
                  f"{sum(len(t.columns) for t in noised_schema.tables.values())} columns")

        # ============================================================
        # Step 3: Load Task
        # ============================================================
        if verbose:
            print("\n[3/5] Loading task...")

        dataset = DatabaseFactory.get_dataset()
        task = DatabaseFactory.get_task(db_name, task_name)
        task_profile = PredictionTaskProfile.from_relbench_task(task, task_name).to_task_profile()

        if verbose:
            print(f"  ✓ Task: {task_profile.task_type}, Entity: {task_profile.entity_table}")

        # ============================================================
        # Step 4: Run Pipeline (Table Selection → Feature Selection)
        # ============================================================
        if verbose:
            print("\n[4/5] Running data profiling pipeline...")

        # Table selection
        table_selector = TableSelector(
            max_tables=max_tables,
            include_examples=True,
            focus_on_connectivity=True
        )

        table_result = table_selector(
            llm_client=llm_client,
            task_profile=task_profile,
            db_schema=noised_schema
        )

        if not table_result.success:
            if verbose:
                print("  ❌ Table selection failed")
            return None

        if verbose:
            print(f"  ✓ Table selection: {len(table_result.db_schema.tables)} tables selected")

        # Feature selection
        feature_selector = FeatureSelector(include_examples=True)

        feature_result = feature_selector(
            llm_client=llm_client,
            db_schema=table_result.db_schema,
            entity_table=task_profile.entity_table
        )

        if not feature_result.success:
            if verbose:
                print("  ❌ Feature selection failed")
            return None

        final_schema = feature_result.db_schema

        if verbose:
            total_cols = sum(len(t.columns) for t in final_schema.tables.values())
            print(f"  ✓ Feature selection: {total_cols} columns selected")

        # ============================================================
        # Step 5: Calculate Column-Level Metrics
        # ============================================================
        if verbose:
            print("\n[5/5] Calculating metrics...")

        metrics = calculate_column_metrics(
            predicted_schema=final_schema,
            ground_truth_schema=ground_truth_schema,
            noised_schema=noised_schema
        )

        if verbose:
            print(f"  ✓ Precision: {metrics['precision']:.3f}")
            print(f"  ✓ Recall:    {metrics['recall']:.3f}")
            print(f"  ✓ F1 Score:  {metrics['f1']:.3f}")

        return {
            'db_name': db_name,
            'task_name': task_name,
            'metrics': metrics,
            'success': True
        }

    except Exception as e:
        if verbose:
            print(f"\n  ❌ Error: {e}")
        return None


def print_results_table(results: List[Dict], aggregate: Dict):
    """Print evaluation results in a formatted table."""
    print_section("EVALUATION RESULTS")

    # Check if we have multiple difficulty levels
    difficulty_levels_in_results = set(r.get('difficulty_level', 'N/A') for r in results)
    has_multiple_difficulties = len(difficulty_levels_in_results) > 1

    # Header
    if has_multiple_difficulties:
        print(f"\n{'Difficulty':<10} | {'Database':<15} | {'Task':<20} | {'Precision':>10} | {'Recall':>10} | {'F1':>10} | "
              f"{'TP':>6} | {'FP':>6} | {'FN':>6}")
        print_separator("-", 105)
    else:
        print(f"\n{'Database':<20} | {'Task':<25} | {'Precision':>10} | {'Recall':>10} | {'F1':>10} | "
              f"{'TP':>6} | {'FP':>6} | {'FN':>6}")
        print_separator("-", 100)

    # Per-database results
    for r in results:
        m = r['metrics']
        if has_multiple_difficulties:
            print(f"{r.get('difficulty_level', 'N/A'):<10} | {r['db_name']:<15} | {r['task_name']:<20} | "
                  f"{m['precision']:>10.3f} | {m['recall']:>10.3f} | {m['f1']:>10.3f} | "
                  f"{m['true_positives']:>6} | {m['false_positives']:>6} | {m['false_negatives']:>6}")
        else:
            print(f"{r['db_name']:<20} | {r['task_name']:<25} | "
                  f"{m['precision']:>10.3f} | {m['recall']:>10.3f} | {m['f1']:>10.3f} | "
                  f"{m['true_positives']:>6} | {m['false_positives']:>6} | {m['false_negatives']:>6}")

    if len(results) > 1:
        # Separator
        print_separator("-", 105 if has_multiple_difficulties else 100)

        # If multiple difficulties, show aggregate per difficulty
        if has_multiple_difficulties:
            for level in sorted(difficulty_levels_in_results):
                level_results = [r for r in results if r.get('difficulty_level') == level]
                if level_results:
                    agg = aggregate_metrics([r['metrics'] for r in level_results])
                    print(f"{level:<10} | {'MEAN':<15} | {'':<20} | "
                          f"{agg['precision']['mean']:>10.3f} | {agg['recall']['mean']:>10.3f} | {agg['f1']['mean']:>10.3f} |")

        # Overall aggregate statistics
        agg = aggregate
        label = 'OVERALL MEAN' if has_multiple_difficulties else 'MEAN'
        print(f"{label:<20} | {'':<25} | "
              f"{agg['precision']['mean']:>10.3f} | {agg['recall']['mean']:>10.3f} | {agg['f1']['mean']:>10.3f} |")

        print(f"{'STD':<20} | {'':<25} | "
              f"{agg['precision']['std']:>10.3f} | {agg['recall']['std']:>10.3f} | {agg['f1']['std']:>10.3f} |")

        print(f"{'MIN':<20} | {'':<25} | "
              f"{agg['precision']['min']:>10.3f} | {agg['recall']['min']:>10.3f} | {agg['f1']['min']:>10.3f} |")

        print(f"{'MAX':<20} | {'':<25} | "
              f"{agg['precision']['max']:>10.3f} | {agg['recall']['max']:>10.3f} | {agg['f1']['max']:>10.3f} |")

    print_separator("=", 105 if has_multiple_difficulties else 100)

    # Summary
    if len(results) > 1:
        print(f"\n📊 SUMMARY (Column-Level Metrics)")
        print(f"   Precision: {aggregate['precision']['mean']:.3f} ± {aggregate['precision']['std']:.3f}")
        print(f"   Recall:    {aggregate['recall']['mean']:.3f} ± {aggregate['recall']['std']:.3f}")
        print(f"   F1 Score:  {aggregate['f1']['mean']:.3f} ± {aggregate['f1']['std']:.3f}")
        print(f"   Evaluations: {len(results)}")
        if has_multiple_difficulties:
            print(f"   Difficulty levels: {', '.join(sorted(difficulty_levels_in_results))}")


def run_benchmark_eval(
    provider: str,
    model: Optional[str] = None,
    difficulty: str = "B",
    max_tables: int = 10,
    verbose: bool = False
) -> List[Dict]:
    """
    Run evaluation on the full benchmark.

    Args:
        provider: LLM provider
        model: Model name (optional)
        difficulty: Difficulty level ("A", "B", "C", or "all")
        max_tables: Maximum tables to select
        verbose: Verbose output

    Returns:
        List of evaluation results
    """
    print_separator("=", 100)
    print("BENCHMARK EVALUATION - Basic Data Profile Parsing".center(100))
    print_separator("=", 100)

    # Initialize LLM client
    print_section("INITIALIZING LLM CLIENT")
    try:
        llm_client = LLMClientFactory.create(provider=provider, model=model)
        print("✓ LLM client initialized")
        print(f"  Provider: {provider}")
        if model:
            print(f"  Model: {model}")
    except Exception as e:
        print(f"❌ Error creating LLM client: {e}")
        return []

    # Determine which difficulty levels to run
    if difficulty == "all":
        difficulty_levels = ['A', 'B', 'C']
    else:
        difficulty_levels = [difficulty]

    all_results = []

    for level in difficulty_levels:
        # Get noise config for this difficulty level
        noise_config = get_noise_config(level, EVALUATION_BENCHMARK['random_seed'])

        print_section(f"DIFFICULTY LEVEL {level}: {noise_config.difficulty_level}")

        level_info = DIFFICULTY_LEVELS[level]
        print(f"  Description: {level_info['description']}")
        print(f"  Noise tables: {noise_config.num_noise_tables}")
        print(f"  Noise columns range: {noise_config.noise_columns_range}")
        print(f"  Linking strategy: {noise_config.linking_strategy}")
        print(f"  Random seed: {noise_config.random_seed}")

        # Run evaluations for this difficulty level
        print(f"\n  Running evaluations on {len(EVALUATION_BENCHMARK['databases'])} databases...")

        for i, (db_name, task_name) in enumerate(EVALUATION_BENCHMARK['databases'], 1):
            print(f"  [{i}/{len(EVALUATION_BENCHMARK['databases'])}] {db_name} ({task_name})...", end=" ")

            result = run_single_eval(
                db_name=db_name,
                task_name=task_name,
                llm_client=llm_client,
                noise_config=noise_config,
                max_tables=max_tables,
                verbose=verbose
            )

            if result:
                # Add difficulty level to result
                result['difficulty_level'] = level
                all_results.append(result)
                print(f"✅ F1: {result['metrics']['f1']:.3f}")
            else:
                print("❌ Failed")

    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark evaluation for basic data profile parsing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
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
        help="Model name (optional)"
    )

    parser.add_argument(
        "--difficulty",
        type=str,
        default="all",
        choices=["A", "B", "C", "all"],
        help="Difficulty level: A (simple), B (medium), C (challenge), all (run all levels) (default: all)"
    )

    parser.add_argument(
        "--max-tables",
        type=int,
        default=10,
        help="Maximum tables to select (default: 10)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose mode"
    )

    args = parser.parse_args()

    # Run benchmark evaluation
    results = run_benchmark_eval(
        provider=args.provider,
        model=args.model,
        difficulty=args.difficulty,
        max_tables=args.max_tables,
        verbose=args.verbose
    )

    if not results:
        print("\n❌ No successful evaluations!")
        sys.exit(1)

    # Aggregate and display
    aggregate = aggregate_metrics([r['metrics'] for r in results])
    print_results_table(results, aggregate)

    # Calculate expected total
    num_difficulties = 3 if args.difficulty == 'all' else 1
    expected_total = len(EVALUATION_BENCHMARK['databases']) * num_difficulties

    print(f"\n✅ Benchmark completed: {len(results)}/{expected_total} evaluations")
    sys.exit(0)


if __name__ == "__main__":
    main()
