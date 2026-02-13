#!/usr/bin/env python3
"""
Unified Evaluation for Basic Data Profile Parsing

Evaluates the complete data profiling pipeline by:
1. Loading ground truth schema (clean database schema)
2. Adding noise (irrelevant tables and columns) - deterministic with fixed seed
3. Running the full parsing pipeline (task parsing → table selection → feature selection)
4. Comparing results with ground truth at column level
5. Aggregating metrics across multiple databases

Usage:
    # Evaluate on single database
    python -m aida.eval.basic_data_parse.run_eval \\
        --database avito --task user-clicks \\
        --provider deepseek --verbose

    # Batch evaluation on benchmark
    python -m aida.eval.basic_data_parse.run_eval \\
        --benchmark --provider deepseek

Examples:
    # Single database with custom noise
    python -m aida.eval.basic_data_parse.run_eval \\
        --database hm --task customer-churn \\
        --provider openai --noise-tables 10 --noise-columns 5

    # Run full benchmark evaluation
    python -m aida.eval.basic_data_parse.run_eval \\
        --benchmark --provider deepseek --verbose
"""

import argparse
import sys
from typing import Optional, Dict, List
from statistics import mean, stdev

from utils.data.database_factory import DatabaseFactory
from aida.db.profile import DatabaseSchema, PredictionTaskProfile
from aida.query_analyzer import TableSelector, FeatureSelector
from aida.llm import LLMClientFactory
from .noise_generator import NoiseGenerator, NoiseConfig
from .dataset import generate_evaluation_dataset, get_benchmark_dataset, EvaluationInstance


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
    noise_config: NoiseConfig,
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

        task = TaskFactory.get_task(db_name, task_name)
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

        metrics = generator.calculate_metrics(
            predicted_schema=final_schema,
            ground_truth_schema=ground_truth_schema,
            level="column"
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


def aggregate_metrics(results: List[Dict]) -> Dict:
    """
    Aggregate metrics across multiple evaluations.

    Args:
        results: List of evaluation results

    Returns:
        Dictionary with aggregate statistics
    """
    if not results:
        return {}

    precisions = [r['metrics']['precision'] for r in results]
    recalls = [r['metrics']['recall'] for r in results]
    f1s = [r['metrics']['f1'] for r in results]

    def stats(values):
        if len(values) == 0:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        return {
            'mean': mean(values),
            'std': stdev(values) if len(values) > 1 else 0,
            'min': min(values),
            'max': max(values)
        }

    return {
        'precision': stats(precisions),
        'recall': stats(recalls),
        'f1': stats(f1s),
        'num_databases': len(results)
    }


def print_results_table(results: List[Dict], aggregate: Dict):
    """Print evaluation results in a formatted table."""
    print_section("EVALUATION RESULTS")

    # Header
    print(f"\n{'Database':<20} | {'Task':<25} | {'Precision':>10} | {'Recall':>10} | {'F1':>10} | "
          f"{'TP':>6} | {'FP':>6} | {'FN':>6}")
    print_separator("-", 100)

    # Per-database results
    for r in results:
        m = r['metrics']
        print(f"{r['db_name']:<20} | {r['task_name']:<25} | "
              f"{m['precision']:>10.3f} | {m['recall']:>10.3f} | {m['f1']:>10.3f} | "
              f"{m['true_positives']:>6} | {m['false_positives']:>6} | {m['false_negatives']:>6}")

    if len(results) > 1:
        # Separator
        print_separator("-", 100)

        # Aggregate statistics
        agg = aggregate
        print(f"{'MEAN':<20} | {'':<25} | "
              f"{agg['precision']['mean']:>10.3f} | {agg['recall']['mean']:>10.3f} | {agg['f1']['mean']:>10.3f} |")

        print(f"{'STD':<20} | {'':<25} | "
              f"{agg['precision']['std']:>10.3f} | {agg['recall']['std']:>10.3f} | {agg['f1']['std']:>10.3f} |")

        print(f"{'MIN':<20} | {'':<25} | "
              f"{agg['precision']['min']:>10.3f} | {agg['recall']['min']:>10.3f} | {agg['f1']['min']:>10.3f} |")

        print(f"{'MAX':<20} | {'':<25} | "
              f"{agg['precision']['max']:>10.3f} | {agg['recall']['max']:>10.3f} | {agg['f1']['max']:>10.3f} |")

    print_separator("=", 100)

    # Summary
    if len(results) > 1:
        print(f"\n📊 SUMMARY (Column-Level Metrics)")
        print(f"   Precision: {aggregate['precision']['mean']:.3f} ± {aggregate['precision']['std']:.3f}")
        print(f"   Recall:    {aggregate['recall']['mean']:.3f} ± {aggregate['recall']['std']:.3f}")
        print(f"   F1 Score:  {aggregate['f1']['mean']:.3f} ± {aggregate['f1']['std']:.3f}")
        print(f"   Databases: {len(results)}")


def run_benchmark_eval(
    provider: str,
    model: Optional[str] = None,
    max_tables: int = 10,
    verbose: bool = False
) -> List[Dict]:
    """
    Run evaluation on the full benchmark.

    Args:
        provider: LLM provider
        model: Model name (optional)
        max_tables: Maximum tables to select
        verbose: Verbose output

    Returns:
        List of evaluation results
    """
    print_separator("=", 100)
    print("BENCHMARK EVALUATION - Basic Data Profile Parsing".center(100))
    print_separator("=", 100)

    print(f"\nBenchmark Configuration:")
    print(f"  Databases: {len(EVALUATION_BENCHMARK['databases'])}")
    print(f"  Noise: {EVALUATION_BENCHMARK['noise_tables']} tables, "
          f"{EVALUATION_BENCHMARK['noise_columns']} columns/table")
    print(f"  Random Seed: {EVALUATION_BENCHMARK['random_seed']}")
    print(f"  Provider: {provider}")
    if model:
        print(f"  Model: {model}")

    # Initialize LLM client
    print_section("INITIALIZING LLM CLIENT")
    try:
        llm_client = LLMClientFactory.create(provider=provider, model=model)
        print("✓ LLM client initialized")
    except Exception as e:
        print(f"❌ Error creating LLM client: {e}")
        return []

    # Setup noise configuration
    noise_config = NoiseConfig(
        num_noise_tables=EVALUATION_BENCHMARK['noise_tables'],
        num_noise_columns_per_table=EVALUATION_BENCHMARK['noise_columns'],
        random_seed=EVALUATION_BENCHMARK['random_seed']
    )

    # Run evaluations
    print_section("RUNNING EVALUATIONS")
    results = []

    for i, (db_name, task_name) in enumerate(EVALUATION_BENCHMARK['databases'], 1):
        print(f"\n[{i}/{len(EVALUATION_BENCHMARK['databases'])}] {db_name} ({task_name})...", end=" ")

        result = run_single_eval(
            db_name=db_name,
            task_name=task_name,
            llm_client=llm_client,
            noise_config=noise_config,
            max_tables=max_tables,
            verbose=verbose
        )

        if result:
            results.append(result)
            print(f"✅ F1: {result['metrics']['f1']:.3f}")
        else:
            print("❌ Failed")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Unified evaluation for basic data profile parsing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run full benchmark evaluation"
    )

    parser.add_argument(
        "--database",
        type=str,
        help="Single database name (e.g., 'avito')"
    )

    parser.add_argument(
        "--task",
        type=str,
        help="Task name for single database"
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
        "--noise-tables",
        type=int,
        default=None,
        help="Number of noise tables (default: use benchmark config)"
    )

    parser.add_argument(
        "--noise-columns",
        type=int,
        default=None,
        help="Number of noise columns per table (default: use benchmark config)"
    )

    parser.add_argument(
        "--max-tables",
        type=int,
        default=10,
        help="Maximum tables to select (default: 10)"
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed (default: use benchmark config)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose mode"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.benchmark and args.database:
        print("Error: Cannot specify both --benchmark and --database")
        sys.exit(1)

    if not args.benchmark and not args.database:
        print("Error: Must specify either --benchmark or --database")
        parser.print_help()
        sys.exit(1)

    # Run evaluation
    if args.benchmark:
        # Benchmark evaluation
        results = run_benchmark_eval(
            provider=args.provider,
            model=args.model,
            max_tables=args.max_tables,
            verbose=args.verbose
        )

        if not results:
            print("\n❌ No successful evaluations!")
            sys.exit(1)

        # Aggregate and display
        aggregate = aggregate_metrics(results)
        print_results_table(results, aggregate)

        print(f"\n✅ Benchmark completed: {len(results)}/{len(EVALUATION_BENCHMARK['databases'])} databases")
        sys.exit(0)

    else:
        # Single database evaluation
        if not args.task:
            args.task = TaskFactory.get_default_task_name(args.database)

        # Setup noise config
        noise_config = NoiseConfig(
            num_noise_tables=args.noise_tables if args.noise_tables else EVALUATION_BENCHMARK['noise_tables'],
            num_noise_columns_per_table=args.noise_columns if args.noise_columns else EVALUATION_BENCHMARK['noise_columns'],
            random_seed=args.random_seed if args.random_seed else EVALUATION_BENCHMARK['random_seed']
        )

        # Initialize LLM
        try:
            llm_client = LLMClientFactory.create(provider=args.provider, model=args.model)
        except Exception as e:
            print(f"❌ Error creating LLM client: {e}")
            sys.exit(1)

        # Run evaluation
        result = run_single_eval(
            db_name=args.database,
            task_name=args.task,
            llm_client=llm_client,
            noise_config=noise_config,
            max_tables=args.max_tables,
            verbose=True  # Always verbose for single eval
        )

        if result:
            print_results_table([result], {})
            print("\n✅ Evaluation completed successfully")
            sys.exit(0)
        else:
            print("\n❌ Evaluation failed")
            sys.exit(1)


if __name__ == "__main__":
    main()
