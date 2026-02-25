#!/usr/bin/env python3
"""
BasicTaskParser Evaluation Script

Evaluates BasicTaskParser on all datasets defined in dataset.py.

Usage:
    # Dry-run with 10 random samples (quick validation)
    python -m aida.eval.basic_task_parse.run_eval --mode random

    # Full evaluation on all datasets (default)
    python -m aida.eval.basic_task_parse.run_eval
    python -m aida.eval.basic_task_parse.run_eval --mode all

    # Change provider
    python -m aida.eval.basic_task_parse.run_eval --provider anthropic

    # Specify model
    python -m aida.eval.basic_task_parse.run_eval --provider openai --model gpt-4o-mini

    # Quick dry-run with different provider
    python -m aida.eval.basic_task_parse.run_eval --provider openai --mode random

    # Verbose mode to debug failures
    python -m aida.eval.basic_task_parse.run_eval --verbose
    python -m aida.eval.basic_task_parse.run_eval -v --mode random
"""

import argparse

from utils.data import DatabaseFactory
from aida.db.profile import DatabaseSchema
from aida.query_analyzer import BasicTaskParser
from aida.llm import LLMClientFactory

from .dataset import collect_eval_dataset


def run_eval(provider="deepseek", model=None, mode="all", verbose=False):
    """Run evaluation on all datasets and report performance."""

    # Setup
    print("=" * 80)
    print("BasicTaskParser Evaluation".center(80))
    print("=" * 80)
    print(f"\nProvider: {provider}")


    llm_client = LLMClientFactory.create(provider, model=model)
    parser = BasicTaskParser()
    
    model = llm_client.default_model if model is None else model
    print(f"Model: {model}")
    print(f"Mode: {mode} ({'dry-run with 10 samples' if mode == 'random' else 'full evaluation'})")
    if verbose:
        print("Verbose: ON (will show details for failed cases)")
    print()
    
    # Collect dataset
    print("Collecting evaluation dataset...")
    dataset = collect_eval_dataset(mode=mode)
    print(f"Total: {len(dataset)} cases\n")

    # Cache db schemas
    db_schemas = {}

    # Run evaluation
    results = []
    for i, item in enumerate(dataset, 1):
        db_name = item["db_name"]
        expected_success = item["expected_success"]

        # Get schema
        if db_name not in db_schemas:
            db = DatabaseFactory.get_db(db_name, upto_test_timestamp=False)
            db_schemas[db_name] = DatabaseSchema.from_relbench_database(db, db_name)

        # Parse
        result = parser(
            llm_client=llm_client,
            nl_query=item["nl_query"],
            db_schema=db_schemas[db_name]
        )

        # Evaluate based on expected outcome
        if expected_success:
            # Positive case: all three keys must match correctly
            if result.success and result.profile:
                p = result.profile

                # Check task_type
                type_match = p.task_type == item["task_type"]

                # Check entity_table
                table_match = p.entity_table.lower() == item["entity_table"].lower()

                # Check time_duration (including None cases)
                gt_time = item["time_duration"]
                pred_time = p.time_duration
                if gt_time is None and pred_time is None:
                    time_match = True
                elif gt_time is None or pred_time is None:
                    time_match = False
                else:
                    # Allow 20% tolerance or 7 days, whichever is larger
                    time_match = abs(pred_time - gt_time) <= max(7, gt_time * 0.2)

                # Correct only if ALL keys match
                correct = type_match and table_match and time_match
            else:
                correct = False

            results.append({"case_type": "positive", "correct": correct})
            status = "✓" if correct else "✗"
            print(f"[{i:3d}/{len(dataset)}] {status} {db_name:12s} {item['task_name']:20s}")

            # Show details for failed cases in verbose mode
            if not correct and verbose:
                print("\n" + "-" * 80)
                print(f"FAILED CASE: {db_name}/{item['task_name']}")
                print("-" * 80)
                print(f"Query: {item['nl_query']}")
                print(f"\nDatabase Schema:")
                print(db_schemas[db_name].description)
                print(f"\nExpected:")
                print(f"  task_type: {item['task_type']}")
                print(f"  entity_table: {item['entity_table']}")
                print(f"  time_duration: {item['time_duration']}")
                if result.success and result.profile:
                    print(f"\nPredicted:")
                    print(f"  task_type: {result.profile.task_type} {'✓' if type_match else '✗'}")
                    print(f"  entity_table: {result.profile.entity_table} {'✓' if table_match else '✗'}")
                    print(f"  time_duration: {result.profile.time_duration} {'✓' if time_match else '✗'}")
                else:
                    print(f"\nPredicted: PARSING FAILED")
                    if result.errors:
                        print(f"Errors: {result.errors}")
                print("-" * 80 + "\n")

        else:
            # Negative case: should be rejected
            correct = not result.success
            results.append({"case_type": "negative", "correct": correct})

            status = "✓" if correct else "✗"
            print(f"[{i:3d}/{len(dataset)}] {status} [NEG] {item['task_name']:30s} "
                  f"{'rejected' if not result.success else 'wrongly accepted'}")

            # Show details for failed negative cases in verbose mode
            if not correct and verbose:
                print("\n" + "-" * 80)
                print(f"FAILED NEGATIVE CASE: {item['task_name']}")
                print("-" * 80)
                print(f"Query: {item['nl_query']}")
                print(f"Failure reason: {item.get('failure_reason', 'N/A')}")
                print(f"\nDatabase Schema:")
                print(db_schemas[db_name].description)
                print(f"\nExpected: Should be REJECTED")
                if result.success and result.profile:
                    print(f"\nPredicted: WRONGLY ACCEPTED")
                    print(f"  task_type: {result.profile.task_type}")
                    print(f"  entity_table: {result.profile.entity_table}")
                    print(f"  time_duration: {result.profile.time_duration}")
                print("-" * 80 + "\n")

    # Calculate metrics
    positive_results = [r for r in results if r["case_type"] == "positive"]
    negative_results = [r for r in results if r["case_type"] == "negative"]

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    n_total = len(results)
    n_correct = sum(r['correct'] for r in results)
    total_acc = n_correct / n_total if n_total > 0 else 0

    print(f"\nTotal Cases: {n_total}")
    print(f"  Positive: {len(positive_results)}")
    print(f"  Negative: {len(negative_results)}")
    print(f"\nCorrect: {n_correct}")
    print(f"Wrong: {n_total - n_correct}")
    print(f"\nOverall Accuracy: {total_acc:.1%}")
    print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate BasicTaskParser on all datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="deepseek",
        choices=LLMClientFactory.available_providers(),
        help="LLM provider (default: deepseek)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (optional, uses provider default)"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["random", "all"],
        help="Evaluation mode: 'random' (10 random samples for dry-run) or 'all' (full evaluation) (default: all)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose mode: show database schema and query for failed cases"
    )

    args = parser.parse_args()
    run_eval(provider=args.provider, model=args.model, mode=args.mode, verbose=args.verbose)


if __name__ == "__main__":
    main()
