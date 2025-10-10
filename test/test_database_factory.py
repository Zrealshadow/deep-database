"""
Tests for DatabaseFactory registration pattern using unittest.

This test file verifies:
1. Dataset registration and retrieval
2. Task registration and retrieval
3. Self-registration of custom datasets
4. Error handling for unknown datasets/tasks

Usage:
    # Run fast tests only (skip integration tests)
    python -m unittest test.test_database_factory

    # Run all tests including integration tests
    XXX_RUN_INTEGRATION_TESTS=1 python -m unittest test.test_database_factory

    # Or set the environment variable
    export XXX_RUN_INTEGRATION_TESTS=1
    python -m unittest test.test_database_factory
"""

import os
import unittest
from utils.data.database_factory import DatabaseFactory
from relbench.base import Database, Dataset, BaseTask

# Flag to control integration tests, XXX is the name of this system.
XXX_RUN_INTEGRATION_TESTS = os.environ.get(
    "XXX_RUN_INTEGRATION_TESTS", "0") == "1"


@unittest.skipIf(XXX_RUN_INTEGRATION_TESTS, "Skipping unit tests when integration tests are enabled")
class TestDatabaseFactoryRegistration(unittest.TestCase):
    """Test the registration pattern of DatabaseFactory."""

    def test_1_print_registration_summary(self):
        """Print a summary of all registered datasets and tasks."""
        print("\n" + "="*70)
        print("DATABASE FACTORY REGISTRATION SUMMARY")
        print("="*70)

        databases = DatabaseFactory.get_registered_databases()
        print(f"\n📊 Total Registered Databases: {len(databases)}")
        print(f"   {databases}\n")

        for db_name in sorted(databases):
            tasks = DatabaseFactory.get_registered_tasks(db_name)
            print(f"🗂️  {db_name}:")
            print(f"   Tasks ({len(tasks)}): {tasks}")

        print("\n" + "="*70)
        print("✅ All registrations working correctly!")
        print("="*70 + "\n")

    def test_2_registered_databases_exist(self):
        """Test that default databases are registered."""
        registered_dbs = DatabaseFactory.get_registered_databases()

        # Check default relbench datasets
        self.assertIn("event", registered_dbs)
        self.assertIn("avito", registered_dbs)
        self.assertIn("trial", registered_dbs)
        self.assertIn("f1", registered_dbs)
        self.assertIn("amazon", registered_dbs)

        # Check custom self-registered datasets
        self.assertIn("stack", registered_dbs)
        self.assertIn("ratebeer", registered_dbs)

        print(f"✅ Registered databases: {registered_dbs}")

    def test_3_event_tasks_registered(self):
        """Test that event tasks are registered."""
        event_tasks = DatabaseFactory.get_registered_tasks("event")
        self.assertIn("user-repeat", event_tasks)
        self.assertIn("user-ignore", event_tasks)
        self.assertIn("user-attendance", event_tasks)
        print(f"✅ Event tasks: {event_tasks}")

    def test_3_avito_tasks_registered(self):
        """Test that avito tasks are registered."""
        avito_tasks = DatabaseFactory.get_registered_tasks("avito")
        self.assertIn("user-clicks", avito_tasks)
        self.assertIn("ad-ctr", avito_tasks)
        self.assertIn("user-visits", avito_tasks)
        print(f"✅ Avito tasks: {avito_tasks}")

    def test_3_stack_tasks_registered(self):
        """Test that stack tasks are registered."""
        stack_tasks = DatabaseFactory.get_registered_tasks("stack")
        self.assertIn("user-engagement", stack_tasks)
        self.assertIn("user-badge", stack_tasks)
        self.assertIn("post-vote", stack_tasks)
        print(f"✅ Stack tasks: {stack_tasks}")

    def test_3_ratebeer_tasks_registered(self):
        """Test that ratebeer tasks are registered."""
        ratebeer_tasks = DatabaseFactory.get_registered_tasks("ratebeer")
        self.assertIn("user-active", ratebeer_tasks)
        self.assertIn("beer-positive", ratebeer_tasks)
        self.assertIn("place-positive", ratebeer_tasks)
        print(f"✅ RateBeer tasks: {ratebeer_tasks}")

    def test_3_trial_tasks_registered(self):
        """Test that trial tasks are registered."""
        trial_tasks = DatabaseFactory.get_registered_tasks("trial")
        self.assertIn("study-outcome", trial_tasks)
        self.assertIn("site-success", trial_tasks)
        self.assertIn("study-adverse", trial_tasks)
        print(f"✅ Trial tasks: {trial_tasks}")

    def test_3_f1_tasks_registered(self):
        """Test that f1 tasks are registered."""
        f1_tasks = DatabaseFactory.get_registered_tasks("f1")
        self.assertIn("driver-dnf", f1_tasks)
        self.assertIn("driver-top3", f1_tasks)
        print(f"✅ F1 tasks: {f1_tasks}")

    def test_4_get_dataset_unknown_database_raises_error(self):
        """Test error handling for unknown database."""
        with self.assertRaises(ValueError) as context:
            DatabaseFactory.get_dataset("nonexistent_db")

        self.assertIn("Unknown database name", str(context.exception))
        self.assertIn("nonexistent_db", str(context.exception))
        self.assertIn("Available databases", str(context.exception))
        print(f"✅ Error message for unknown database: {context.exception}")

    def test_4_get_db_unknown_database_raises_error(self):
        """Test error handling for unknown database in get_db."""
        with self.assertRaises(ValueError) as context:
            DatabaseFactory.get_db("nonexistent_db")

        self.assertIn("not found", str(context.exception))
        print(f"✅ Error message for unknown db: {context.exception}")

    def test_4_get_task_unknown_database_raises_error(self):
        """Test error handling for unknown database in get_task."""

        with self.assertRaises(ValueError) as context:
            DatabaseFactory.get_task("nonexistent_db", "some-task", None)

        self.assertIn("Unknown database name", str(context.exception))
        self.assertIn("nonexistent_db", str(context.exception))
        print(
            f"✅ Error message for unknown database in get_task: {context.exception}")

    def test_4_get_task_unknown_task_raises_error(self):
        """Test error handling for unknown task name."""
        with self.assertRaises(ValueError) as context:
            DatabaseFactory.get_task("event", "nonexistent-task", None)

        self.assertIn("Unknown task name", str(context.exception))
        self.assertIn("nonexistent-task", str(context.exception))
        self.assertIn("Available tasks", str(context.exception))
        print(f"✅ Error message for unknown task: {context.exception}")

    def test_5_dataset_registry_structure(self):
        """Test that dataset registry has correct structure."""
        registry = DatabaseFactory._dataset_registry

        self.assertIsInstance(registry, dict)
        self.assertGreater(len(registry), 0)

        # Check structure of a registered dataset
        self.assertIn("event", registry)
        self.assertIn("loader", registry["event"])
        self.assertIn("preprocessor", registry["event"])
        self.assertTrue(callable(registry["event"]["loader"]))
        print(f"✅ Dataset registry structure is correct")

    def test_5_task_registry_structure(self):
        """Test that task registry has correct structure."""
        registry = DatabaseFactory._task_registry

        self.assertIsInstance(registry, dict)
        self.assertGreater(len(registry), 0)

        # Check structure
        self.assertIn("event", registry)
        self.assertIsInstance(registry["event"], dict)
        self.assertIn("user-repeat", registry["event"])
        print(f"✅ Task registry structure is correct")


@unittest.skipIf(not XXX_RUN_INTEGRATION_TESTS, "Integration tests disabled. Set XXX_RUN_INTEGRATION_TESTS=1 to enable.")
class TestDatabaseFactoryIntegration(unittest.TestCase):
    """Integration tests - controlled by XXX_RUN_INTEGRATION_TESTS environment variable."""

    # Test configuration: (db_name, task_name)
    DATASET_TEST_CASES = [
        ("event", "user-repeat"),
        ("avito", "user-clicks"),
        ("trial", "study-outcome"),
        ("f1", "driver-dnf"),
        ("stack", "user-engagement"),  # Uncomment if you want to test stack
        ("ratebeer", "user-active"), # Uncomment if you have local ratebeer data
    ]

    def test_get_db_all(self):
        """Integration test: Load database for all configured databases."""
        for db_name, task_name in self.DATASET_TEST_CASES:
            with self.subTest(database=db_name):
                db = DatabaseFactory.get_db(db_name, upto_test_timestamp=True)
                self.assertIsInstance(db, Database)
                self.assertGreater(len(db.table_dict), 0)

                dataset = DatabaseFactory.get_dataset(db_name)
                self.assertIsInstance(dataset, Dataset)

                task = DatabaseFactory.get_task(db_name, task_name, dataset)
                self.assertIsInstance(task, BaseTask)
                train_table = task.get_table(
                    split="train", mask_input_cols=False)
                self.assertGreater(len(train_table.df), 0)

                print(
                    f"✅ Successfully loaded {db_name} database with {len(db.table_dict)} tables")


if __name__ == "__main__":
    # Run tests
    print("Running unit tests...\n")
    unittest.main(verbosity=2)
