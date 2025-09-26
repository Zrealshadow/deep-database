#!/usr/bin/env python3
"""
Test table selection prompt generation using AIDA prompt classes.
"""

import unittest
from utils.data import DatabaseFactory
from aida.db.profile import DatabaseSchema, PredictionTaskProfile
from aida.prompt.table_selection import TableSelectionPrompt


class TestTableSelectionPrompt(unittest.TestCase):
    """Test table selection prompt generation"""

    def setUp(self):
        """Set up test fixtures"""
        self.db_name = "avito"
        self.task_name = "user-clicks"
        self.dataset = DatabaseFactory.get_dataset(self.db_name)
        print(f"\nLoaded database: {self.db_name}, task: {self.task_name}")
        self.task = DatabaseFactory.get_task(self.db_name, self.task_name, self.dataset)
        self.db = DatabaseFactory.get_db(
            self.db_name,
            upto_test_timestamp=False,
            with_text_compress=False
        )
        
    def test_table_selection_prompt(self):
        """Generate and print table selection prompt"""
        print(f"\n{'='*100}")
        print("TABLE SELECTION PROMPT")
        print(f"{'='*100}")


        # Create schema and task profile
        db_schema = DatabaseSchema.from_relbench_database(self.db, self.db_name)
        task_profile = PredictionTaskProfile.from_relbench_task(self.task, self.task_name)

        # Generate prompt using TableSelectionPrompt class
        prompt = TableSelectionPrompt.generate_table_selection_prompt(
            prediction_task=task_profile,
            database_schema=db_schema,
            max_tables=8,
            include_examples=True,
            focus_on_connectivity=True
        )

        print(prompt)
        print(f"{'='*100}")


if __name__ == "__main__":
    # Run the table selection prompt test by default
    suite = unittest.TestSuite()
    suite.addTest(TestTableSelectionPrompt('test_table_selection_prompt'))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)