"""
Data Profile Parsers

Operators for selecting relevant tables and features from database schema.

These operators modify DatabaseSchema in-place by:
- TableSelector: Removing unselected tables
- FeatureSelector: Removing unselected columns
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import copy

try:
    from typing import override  # Python 3.12+
except ImportError:
    from typing_extensions import override  # Python < 3.12

from aida.db.profile import DatabaseSchema
from aida.prompt.table_selection import TableSelectionPrompt
from aida.prompt.feature_selection import FeatureSelectionPrompt
from .base import BaseOperator, BaseResult
from .nl2task import TaskProfile


@dataclass
class TableSelectionResult(BaseResult):
    """Result from TableSelector."""
    db_schema: Optional[DatabaseSchema] = None

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        if self.db_schema:
            result["selected_tables"] = list(self.db_schema.tables.keys())
            result["num_tables"] = len(self.db_schema.tables)
        return result


@dataclass
class FeatureSelectionResult(BaseResult):
    """Result from FeatureSelector."""
    db_schema: Optional[DatabaseSchema] = None

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        if self.db_schema:
            result["selected_features"] = {
                table_name: list(table.columns)
                for table_name, table in self.db_schema.tables.items()
            }
            result["num_tables"] = len(self.db_schema.tables)
            result["total_columns"] = sum(len(t.columns) for t in self.db_schema.tables.values())
        return result


class TableSelector(BaseOperator):
    """
    Table Selection Operator.

    Uses LLM to heuristically select a slice of database schema
    based on the task profile.

    Example:
        >>> selector = TableSelector()
        >>> result = selector(
        ...     llm_client=llm_client,
        ...     task_profile=task_profile,
        ...     db_schema=db_schema
        ... )
        >>> if result.success:
        ...     print(result.profile.selected_tables)
    """

    def __init__(
        self,
        max_tables: int = 10,
        include_examples: bool = True,
        focus_on_connectivity: bool = True
    ):
        self.max_tables = max_tables
        self.include_examples = include_examples
        self.focus_on_connectivity = focus_on_connectivity

    @override
    def __build_prompt__(
        self,
        task_profile: TaskProfile,
        db_schema: DatabaseSchema,
        **kwargs
    ) -> str:
        return TableSelectionPrompt.generate_table_selection_prompt(
            task_profile=task_profile,
            database_schema=db_schema,
            max_tables=self.max_tables,
            include_examples=self.include_examples,
            focus_on_connectivity=self.focus_on_connectivity
        )

    @override
    def __get_system_prompt__(self) -> str:
        return TableSelectionPrompt.get_system_prompt()

    @override
    def __validate__(
        self,
        extraction: Dict[str, Any],
        db_schema: DatabaseSchema,
        task_profile: TaskProfile,
        **kwargs
    ) -> tuple[List[str], List[str]]:
        errors = []
        warnings = []

        # Check if extraction is a dict (should be {table_name: {}, ...})
        if not isinstance(extraction, dict):
            errors.append("Expected a dictionary mapping table names to empty objects")
            return errors, warnings

        # Validate that all selected tables exist in schema
        for table_name in extraction.keys():
            if table_name not in db_schema.tables:
                errors.append(f"Selected table '{table_name}' does not exist in database schema")

        # Check if entity table is included
        if task_profile.entity_table not in extraction:
            warnings.append(
                f"Entity table '{task_profile.entity_table}' was not selected. "
                "This may impact prediction quality."
            )

        # Check table count
        if len(extraction) > self.max_tables:
            warnings.append(
                f"Selected {len(extraction)} tables, but max_tables is {self.max_tables}. "
                "Consider reducing the number of tables."
            )

        return errors, warnings

    @override
    def __build_result__(
        self,
        extraction: Dict[str, Any],
        warnings: List[str],
        db_schema: DatabaseSchema,
        **kwargs
    ) -> TableSelectionResult:
        selected_tables = list(extraction.keys())

        # Create a deep copy and keep only selected tables
        filtered_schema = copy.deepcopy(db_schema)
        tables_to_remove = [
            table_name for table_name in filtered_schema.tables.keys()
            if table_name not in selected_tables
        ]

        for table_name in tables_to_remove:
            del filtered_schema.tables[table_name]

        # Clear cached description since schema changed
        filtered_schema._cached_description = None

        return TableSelectionResult(
            success=True,
            db_schema=filtered_schema,
            warnings=warnings,
            raw_extraction=extraction
        )


class FeatureSelector(BaseOperator):
    """
    Feature Selection Operator.

    Filters out unnecessary modeling or database built-in columns
    (like created_by, updated_at, etc.) from tables.

    Example:
        >>> selector = FeatureSelector()
        >>> result = selector(
        ...     llm_client=llm_client,
        ...     db_schema=filtered_db_schema,  # from TableSelector
        ...     entity_table="users"
        ... )
        >>> if result.success:
        ...     print(result.db_schema.tables)  # Only selected columns
    """

    def __init__(
        self,
        include_examples: bool = True
    ):
        self.include_examples = include_examples

    @override
    def __build_prompt__(
        self,
        db_schema: DatabaseSchema,
        entity_table: str,
        **kwargs
    ) -> str:
        selected_tables = list(db_schema.tables.keys())

        return FeatureSelectionPrompt.generate_feature_selection_prompt(
            selected_tables=selected_tables,
            database_schema=db_schema,
            entity_table=entity_table,
            include_examples=self.include_examples
        )

    @override
    def __get_system_prompt__(self) -> str:
        return FeatureSelectionPrompt.get_system_prompt()

    @override
    def __validate__(
        self,
        extraction: Dict[str, Any],
        db_schema: DatabaseSchema,
        **kwargs
    ) -> tuple[List[str], List[str]]:
        errors = []
        warnings = []

        # Check if extraction is a dict mapping table -> list of columns
        if not isinstance(extraction, dict):
            errors.append("Expected a dictionary mapping table names to column lists")
            return errors, warnings

        # Validate each table's selected features
        for table_name, columns in extraction.items():
            # Check table exists in schema
            if table_name not in db_schema.tables:
                errors.append(f"Table '{table_name}' does not exist in database schema")
                continue

            # Check columns is a list
            if not isinstance(columns, list):
                errors.append(f"Columns for table '{table_name}' must be a list")
                continue

            # Validate all columns exist in the table
            table_schema = db_schema.tables[table_name]
            for col in columns:
                if col not in table_schema.columns:
                    errors.append(
                        f"Column '{col}' does not exist in table '{table_name}'"
                    )

            # Warn if primary key is missing
            if table_schema.primary_key and table_schema.primary_key not in columns:
                warnings.append(
                    f"Primary key '{table_schema.primary_key}' not selected for table '{table_name}'. "
                    "This may cause issues with entity identification."
                )

            # Warn if foreign keys are missing
            for fk_col in table_schema.foreign_keys.keys():
                if fk_col not in columns:
                    warnings.append(
                        f"Foreign key '{fk_col}' not selected for table '{table_name}'. "
                        "This may limit relationship-based features."
                    )

            # Warn if time column is missing
            if table_schema.time_column and table_schema.time_column not in columns:
                warnings.append(
                    f"Time column '{table_schema.time_column}' not selected for table '{table_name}'. "
                    "This may impact temporal feature engineering."
                )

        # Check if all tables in schema have features selected
        for table_name in db_schema.tables.keys():
            if table_name not in extraction:
                warnings.append(
                    f"No features selected for table '{table_name}'. "
                    "Consider removing this table or selecting relevant columns."
                )

        return errors, warnings

    @override
    def __build_result__(
        self,
        extraction: Dict[str, Any],
        warnings: List[str],
        db_schema: DatabaseSchema,
        **kwargs
    ) -> FeatureSelectionResult:
        # Create a deep copy and keep only selected columns
        filtered_schema = copy.deepcopy(db_schema)

        for table_name, selected_columns in extraction.items():
            if table_name in filtered_schema.tables:
                table = filtered_schema.tables[table_name]
                # Keep only selected columns
                table.columns = [col for col in table.columns if col in selected_columns]

        # Clear cached description since schema changed
        filtered_schema._cached_description = None

        return FeatureSelectionResult(
            success=True,
            db_schema=filtered_schema,
            warnings=warnings,
            raw_extraction=extraction
        )
