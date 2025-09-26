"""
AIDA Prompt Generation Module

This module provides modular prompt generators for LLM-based database analysis
and feature engineering for prediction tasks.

Components:
- schema: Data structures for representing databases and tasks with factory methods
- table_selection: Focused prompts for table selection
- relationship_selection: Prompts for relationship and join analysis
- feature_engineering: Prompts for feature engineering recommendations
"""

from relbench.base import TaskType

from .table_selection import TableSelectionPrompt

__all__ = [
    # Schema types
    'TableSchema',
    'DatabaseSchema',
    'PredictionTaskProfile',

    # Prompt generators
    'TableSelectionPrompt',
]