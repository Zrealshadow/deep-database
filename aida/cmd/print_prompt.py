#!/usr/bin/env python3
"""
Command-line tool to generate and print AIDA prompts for LLM-based database analysis.

Usage:
    python -m aida.cmd.print_prompt <database> <task> <prompt_type>

Examples:
    python -m aida.cmd.print_prompt event user-attendance table_selection
    python -m aida.cmd.print_prompt stack user-badge table_selection
    python -m aida.cmd.print_prompt ratebeer user-active table_selection
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from utils.data import DatabaseFactory
from aida.db.profile import DatabaseSchema, PredictionTaskProfile
from aida.prompt.table_selection import TableSelectionPrompt


def get_available_databases():
    """Get list of available databases from DatabaseFactory"""
    return DatabaseFactory.DBList


def get_available_tasks(db_name: str):
    """Get available tasks for a given database"""
    task_map = {
        'event': ['user-attendance', 'user-ignore', 'user-repeat'],
        'stack': ['post-votes', 'user-badge', 'user-engagement'],
        'avito': ['ad-ctr', 'user-clicks'],
        'trial': ['site-success', 'study-adverse', 'study-outcome'],
        'ratebeer': ['user-active', 'place-positive', 'beer-positive'],
        'f1': ['driver-dnf', 'driver-top3'],
        # 'amazon': ['user-churn', 'product-rating']
    }
    return task_map.get(db_name, [])


def get_available_prompt_types():
    """Get list of available prompt types"""
    return ['table_selection']




def load_database_and_task(db_name: str, task_name: str):
    """Load database and task using DatabaseFactory"""
    try:
        # Load database
        db = DatabaseFactory.get_db(db_name)

        # Load dataset
        dataset = DatabaseFactory.get_dataset(db_name)

        # Load task
        task = DatabaseFactory.get_task(db_name, task_name, dataset)

        return db, task
    except Exception as e:
        print(f"Error loading database '{db_name}' or task '{task_name}': {e}")
        print(f"Available databases: {get_available_databases()}")
        print(f"Available tasks for {db_name}: {get_available_tasks(db_name)}")
        sys.exit(1)


def generate_prompt(db_name: str, task_name: str, prompt_type: str):
    """Generate the specified prompt type"""

    # Load database and task
    db, task = load_database_and_task(db_name, task_name)

    # Convert to AIDA schema objects
    db_schema = DatabaseSchema.from_relbench_database(db, f"{db_name}_database")
    task_profile = PredictionTaskProfile.from_relbench_task(task, f"{task_name}_prediction")

    # Generate prompt based on type
    if prompt_type == 'table_selection':
        prompt = TableSelectionPrompt.generate_table_selection_prompt(
            prediction_task=task_profile,
            database_schema=db_schema,
            max_tables=10,
            include_examples=True,
            focus_on_connectivity=True
        )
    else:
        print(f"Error: Unsupported prompt type '{prompt_type}'")
        print(f"Available prompt types: {get_available_prompt_types()}")
        sys.exit(1)

    return prompt


def main():
    """Main command-line interface"""
    parser = argparse.ArgumentParser(
        description="Generate AIDA prompts for LLM-based database analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        'database',
        help=f"Database name. Available: {', '.join(get_available_databases())}"
    )

    parser.add_argument(
        'task',
        help="Task name (e.g., user-attendance, user-badge, etc.)"
    )

    parser.add_argument(
        'prompt_type',
        help=f"Prompt type. Available: {', '.join(get_available_prompt_types())}"
    )

    parser.add_argument(
        '--output', '-o',
        help="Output file path (default: print to stdout)"
    )

    parser.add_argument(
        '--list-tasks',
        action='store_true',
        help="List available tasks for the specified database"
    )

    args = parser.parse_args()

    # Handle --list-tasks option
    if args.list_tasks:
        tasks = get_available_tasks(args.database)
        if tasks:
            print(f"Available tasks for {args.database}: {', '.join(tasks)}")
        else:
            print(f"No predefined tasks found for {args.database}")
        return

    # Validate inputs
    if args.database not in get_available_databases():
        print(f"Error: Unknown database '{args.database}'")
        print(f"Available databases: {get_available_databases()}")
        sys.exit(1)

    if args.prompt_type not in get_available_prompt_types():
        print(f"Error: Unknown prompt type '{args.prompt_type}'")
        print(f"Available prompt types: {get_available_prompt_types()}")
        sys.exit(1)

    # Generate prompt
    print(f"Generating {args.prompt_type} prompt for {args.database}/{args.task}...")
    prompt = generate_prompt(args.database, args.task, args.prompt_type)

    # Output prompt
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(prompt)
            print(f"Prompt saved to: {args.output}")
        except Exception as e:
            print(f"Error writing to file '{args.output}': {e}")
            sys.exit(1)
    else:
        print("\n" + "="*80)
        print("GENERATED PROMPT:")
        print("="*80)
        print(prompt)
        print("="*80)


if __name__ == "__main__":
    main()