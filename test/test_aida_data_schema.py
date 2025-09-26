#!/usr/bin/env python3
"""
Test script for TableSchema and DatabaseSchema classes using real RelBench data.

This script loads real data from the RelBench stack dataset and tests the
dynamic description generation functionality.
"""

from utils.data import DatabaseFactory
from aida.db.profile import TableSchema, DatabaseSchema, PredictionTaskProfile


def test_with_real_data():
    """Test TableSchema and DatabaseSchema with real RelBench data"""
    print("=== Testing with Real RelBench Data ===")

    # Load the stack (ratebeer) dataset
    db_name = "event"
    # cache_dir = "/home/lingze/.cache/relbench/stack"

    print(f"Loading database: {db_name}")
    db = DatabaseFactory.get_db(
        db_name,
        # cache_dir=cache_dir,
        upto_test_timestamp=False,
        with_text_compress=False
    )

    print(f"Loading dataset: {db_name}")
    dataset = DatabaseFactory.get_dataset(
        db_name,
        # cache_dir=cache_dir,
    )

    # Create DatabaseSchema from real data
    print("\nCreating DatabaseSchema...")
    db_schema = DatabaseSchema.from_relbench_database(db, db_name)
    description = db_schema.description  # Trigger description generation
    print(f"Database name: {db_schema.name}")
    print(f"Number of tables: {len(db_schema.tables)}")
    print(f"Database description: {description}")

    return db_schema, db, dataset


def test_with_task():
    """Test with a specific prediction task"""
    print("\n=== Testing with Prediction Task ===")

    db_name = "event"
    task_name = "user-repeat"  # Using post-votes task

    # Load data
    db = DatabaseFactory.get_db(
        db_name,
        upto_test_timestamp=False,
        with_text_compress=False
    )

    dataset = DatabaseFactory.get_dataset(
        db_name,
    )

    # Load task
    task = DatabaseFactory.get_task(db_name, task_name, dataset)

    # Create task profile
    task_profile = PredictionTaskProfile.from_relbench_task(task, task_name)

    print(task_profile.description)

    return task_profile, task


def main():
    """Run all tests with real data"""
    print("Testing TableSchema and DatabaseSchema with Real RelBench Data")
    print("=" * 70)

    try:
        # Test basic functionality
        # db_schema, db, dataset = test_with_real_data()
        db_schema = None
        # Test with specific task
        task_profile, task = test_with_task()


        print("\n" + "=" * 70)
        print("All tests completed successfully!")

        return db_schema, task_profile

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    db_schema, task_profile = main()