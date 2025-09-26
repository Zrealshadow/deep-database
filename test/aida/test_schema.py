#!/usr/bin/env python3
"""
Test cases for TableSchema and DatabaseSchema classes using real RelBench data.

This module converts the existing test functions into proper unittest test cases.
"""

import unittest
from utils.data import DatabaseFactory
from aida.db.profile import TableSchema, DatabaseSchema, PredictionTaskProfile


class TestDatabaseSchema(unittest.TestCase):
    """Test cases for DatabaseSchema with real RelBench data"""

    @classmethod
    def setUpClass(cls):
        """Set up test data once for all test methods"""
        cls.db_name = "event"
        print(f"Loading database: {cls.db_name}")
        cls.db = DatabaseFactory.get_db(
            cls.db_name,
            upto_test_timestamp=False,
            with_text_compress=False
        )
        cls.dataset = DatabaseFactory.get_dataset(cls.db_name)
        cls.db_schema = DatabaseSchema.from_relbench_database(cls.db, cls.db_name)


    def test_database_description(self):
        """Test database description generation"""
        print("\n=== Test Database Description ===")

        description = self.db_schema.description
        self.assertIsInstance(description, str)
        self.assertIn(self.db_name, description)
        self.assertIn("tables", description)

        print(f"Database description:\n{description}")

    def test_table_schemas(self):
        """Test individual table schemas"""
        print("\n=== Test Individual Table Schemas ===")

        for table_name, table_schema in self.db_schema.tables.items():
            with self.subTest(table=table_name):
                self.assertIsInstance(table_schema, TableSchema)
                self.assertEqual(table_schema.name, table_name)
                self.assertIsInstance(table_schema.columns, list)
                self.assertGreater(len(table_schema.columns), 0)

                print(f"\n**Table: {table_name}**")
                print(f"Columns ({len(table_schema.columns)}): {', '.join(table_schema.columns[:5])}{'...' if len(table_schema.columns) > 5 else ''}")
                print(f"Primary key: {table_schema.primary_key}")
                print(f"Foreign keys: {table_schema.foreign_keys}")
                print(f"Time column: {table_schema.time_column}")
                print(f"Sample size: {table_schema.sample_size}")
                print(f"Description: {table_schema.description}")

    def test_relationships(self):
        """Test relationship extraction"""
        print("\n=== Test Relationship Analysis ===")

        relationships = self.db_schema.extract_relationships()
        self.assertIsInstance(relationships, list)

        for rel in relationships:
            self.assertIn("from", rel)
            self.assertIn("to", rel)
            self.assertIn("fk", rel)
            self.assertIn("pk", rel)

        print(f"Found {len(relationships)} relationships:")
        for rel in relationships:
            print(rel)



class TestPredictionTaskProfile(unittest.TestCase):
    """Test cases for PredictionTaskProfile"""

    def test_task_profile_creation(self):
        """Test with a specific prediction task"""
        print("\n=== Test Prediction Task Profile ===")

        db_name = "event"
        task_name = "user-repeat"

        # Load data
        db = DatabaseFactory.get_db(
            db_name,
            upto_test_timestamp=False,
            with_text_compress=False
        )

        dataset = DatabaseFactory.get_dataset(
            db_name,
            # cache_dir=cache_dir,
        )
        task = DatabaseFactory.get_task(db_name, task_name, dataset)

        # Create task profile
        task_profile = PredictionTaskProfile.from_relbench_task(task, task_name)

        # Assertions
        self.assertIsInstance(task_profile, PredictionTaskProfile)
        self.assertEqual(task_profile.name, task_name)
        self.assertIsInstance(task_profile.description, str)
        self.assertIsNotNone(task_profile.task_type)

        # Print results
        print(f"Task name: {task_profile.name}")
        print(f"Task description: {task_profile.description}")




if __name__ == "__main__":
    # Run tests with detailed output
    unittest.main(verbosity=2)