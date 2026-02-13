"""
Database and Task Schema Definitions for LLM-based Table Selection

This module provides data structures for representing database schemas and prediction tasks
in a format suitable for LLM prompts.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from relbench.base import TaskType, Database, BaseTask, Table


@dataclass
class TableSchema:
    """Represents a database table schema"""
    name: str
    columns: List[str]
    primary_key: Optional[str]
    foreign_keys: Dict[str, str]  # column_name -> referenced_table
    time_column: Optional[str]
    sample_size: Optional[int] = None

    @classmethod
    def from_relbench_table(cls, table: Table, table_name: str) -> 'TableSchema':
        """Create TableSchema from a RBench Table object"""
        # Extract column names
        columns = list(table.df.columns)

        # Get primary key
        primary_key = table.pkey_col

        # Get foreign keys mapping
        foreign_keys = table.fkey_col_to_pkey_table.copy()

        # Get time column
        time_column = table.time_col

        # Get sample size
        sample_size = len(table.df)

        return cls(
            name=table_name,
            columns=columns,
            primary_key=primary_key,
            foreign_keys=foreign_keys,
            time_column=time_column,
            sample_size=sample_size,
        )

    @property
    def description(self) -> str:
        """Generate a dynamic description based on table characteristics"""
        return self._generate_table_description()

    def _generate_table_description(self) -> str:
        """Generate a clear, simple description for LLM input"""
        parts = []

        # Column information
        column_list = ', '.join(self.columns)
        parts.append(f"columns: [{column_list}]")

        # Basic stats
        parts.append(f"{self.sample_size or 0} rows")

        # Key information
        if self.primary_key:
            parts.append(f"primary key: {self.primary_key}")

        if self.foreign_keys:
            fk_list = ', '.join(
                [f"{fk}->{ref}" for fk, ref in self.foreign_keys.items()])
            parts.append(f"foreign keys: {fk_list}")

        if self.time_column:
            parts.append(f"time column: {self.time_column}")

        return f"{self.name} table - {'; '.join(parts)}"

    def add_column(
        self,
        column_name: str,
        is_foreign_key: bool = False,
        references: Optional[tuple] = None  # (table, pk)
    ) -> None:
        """
        Add a column to the table.

        Args:
            column_name: Name of column to add
            is_foreign_key: Whether this is a foreign key
            references: If FK, tuple of (referenced_table, referenced_pk)
        """
        if column_name in self.columns:
            raise ValueError(f"Column {column_name} already exists in table {self.name}")

        self.columns.append(column_name)

        if is_foreign_key and references:
            ref_table, ref_pk = references
            self.foreign_keys[column_name] = ref_table

    def remove_column(self, column_name: str) -> None:
        """
        Remove a column from the table.

        Args:
            column_name: Name of column to remove
        """
        if column_name not in self.columns:
            raise ValueError(f"Column {column_name} not found in table {self.name}")

        # Remove from columns list
        self.columns.remove(column_name)

        # Remove from foreign keys if it's a FK
        if column_name in self.foreign_keys:
            del self.foreign_keys[column_name]

        # Clear primary key if removing the PK
        if self.primary_key == column_name:
            self.primary_key = None

        # Clear time column if removing it
        if self.time_column == column_name:
            self.time_column = None


@dataclass
class DatabaseSchema:
    """Represents a complete database schema"""
    name: str
    tables: Dict[str, TableSchema]
    _cached_description: Optional[str] = None

    @classmethod
    def from_relbench_database(cls, db: Database, db_name: str = "database") -> 'DatabaseSchema':
        """Create DatabaseSchema from a RBench Database object"""
        tables = {}

        for table_name, table in db.table_dict.items():
            table_schema = TableSchema.from_relbench_table(table, table_name)
            tables[table_name] = table_schema

        return cls(
            name=db_name,
            tables=tables
        )

    @property
    def description(self) -> str:
        """Generate a cached dynamic description of the database schema"""
        if self._cached_description is None:
            self._cached_description = self._generate_description()
        return self._cached_description

    def _generate_description(self) -> str:
        """Generate a clear, simple database description for LLM input"""
        parts = []

        # Basic stats
        total_tables = len(self.tables)
        total_columns = sum(len(t.columns) for t in self.tables.values())
        parts.append(f"{total_tables} tables, {total_columns} columns")

        # Relationships
        relationships = self.extract_relationships()
        if relationships:
            parts.append(f"{len(relationships)} relationships")

        # Temporal tables
        temporal_count = sum(1 for t in self.tables.values() if t.time_column)
        if temporal_count > 0:
            parts.append(f"{temporal_count} temporal tables")

        # Data size
        if any(t.sample_size for t in self.tables.values()):
            total_rows = sum(
                t.sample_size for t in self.tables.values() if t.sample_size)
            parts.append(f"{total_rows:,} rows")

        overview = f"{self.name} database - {', '.join(parts)}"

        # Add each table's description
        table_descriptions = []
        for table_schema in self.tables.values():
            table_descriptions.append(table_schema.description)

        relationships = self.extract_relationships()
        # materialize relationships
        rel_descriptions = [f"{rel['from']} -> {rel['to']} (FK: {rel['fk']}, PK: {rel['pk']})"
                            for rel in relationships]
        return (f"{overview}\n\n"+ 
            "======================Tables:=====================\n" + '\n'.join(table_descriptions) + "\n\n ===============Relationships===============\n" + '\n'.join(rel_descriptions)
        )
    def extract_relationships(self) -> List[Dict[str, str]]:
        """Extract relationships between tables in the schema"""
        relationships = []

        for table_name, table in self.tables.items():
            for fkey_col, referenced_table in table.foreign_keys.items():
                if referenced_table in self.tables:
                    ref_table = self.tables[referenced_table]
                    relationship = {
                        "from": table_name,
                        "to": referenced_table,
                        "fk": fkey_col,
                        "pk": ref_table.primary_key
                    }
                    relationships.append(relationship)

        return relationships

    def add_table(self, table: 'TableSchema') -> None:
        """
        Add a table to the schema.

        Args:
            table: TableSchema to add

        Raises:
            ValueError: If table name already exists or FK references are invalid
        """
        if table.name in self.tables:
            raise ValueError(f"Table {table.name} already exists")

        # Validate foreign key references
        for fk_col, ref_table_name in table.foreign_keys.items():
            # Check if referenced table exists
            if ref_table_name not in self.tables:
                raise ValueError(
                    f"Foreign key {fk_col} in table {table.name} references "
                    f"non-existent table {ref_table_name}"
                )

            # Check if referenced table has a primary key
            ref_table = self.tables[ref_table_name]
            if not ref_table.primary_key:
                raise ValueError(
                    f"Foreign key {fk_col} in table {table.name} references "
                    f"table {ref_table_name} which has no primary key"
                )

        self.tables[table.name] = table
        self._cached_description = None

    def remove_table(self, table_name: str, cascade: bool = True) -> None:
        """
        Remove a table from the schema.

        Args:
            table_name: Name of table to remove
            cascade: If True, remove all FKs referencing this table

        Raises:
            ValueError: If table doesn't exist
        """
        if table_name not in self.tables:
            raise ValueError(f"Table {table_name} not found")

        if cascade:
            # Remove all foreign keys in other tables that reference this table
            for other_table in self.tables.values():
                if other_table.name == table_name:
                    continue

                # Find FKs pointing to this table
                fks_to_remove = [
                    fk_col for fk_col, ref_table in other_table.foreign_keys.items()
                    if ref_table == table_name
                ]

                # Remove FK columns and FK references
                for fk_col in fks_to_remove:
                    other_table.remove_column(fk_col)

        # Remove the table
        del self.tables[table_name]
        self._cached_description = None

    def add_column_to_table(
        self,
        table_name: str,
        column_name: str,
        is_foreign_key: bool = False,
        references: Optional[tuple] = None  # (table_name, pk_column)
    ) -> None:
        """
        Add a column to a table.

        Args:
            table_name: Target table name
            column_name: Column name to add
            is_foreign_key: Whether this is a foreign key
            references: If FK, tuple of (referenced_table, referenced_column)

        Raises:
            ValueError: If table not found or FK reference is invalid
        """
        if table_name not in self.tables:
            raise ValueError(f"Table {table_name} not found")

        # Validate foreign key reference if provided
        if is_foreign_key and references:
            ref_table_name, ref_pk = references

            # Check if referenced table exists
            if ref_table_name not in self.tables:
                raise ValueError(
                    f"Foreign key {column_name} references non-existent table {ref_table_name}"
                )

            # Check if referenced table has the specified primary key
            ref_table = self.tables[ref_table_name]
            if not ref_table.primary_key:
                raise ValueError(
                    f"Foreign key {column_name} references table {ref_table_name} which has no primary key"
                )

            if ref_table.primary_key != ref_pk:
                raise ValueError(
                    f"Foreign key {column_name} references {ref_table_name}.{ref_pk}, "
                    f"but primary key is {ref_table.primary_key}"
                )

        table = self.tables[table_name]
        table.add_column(column_name, is_foreign_key, references)
        self._cached_description = None

    def remove_column_from_table(
        self,
        table_name: str,
        column_name: str,
        cascade: bool = True
    ) -> None:
        """
        Remove a column from a table.

        Args:
            table_name: Target table name
            column_name: Column to remove
            cascade: If True and column is a PK, remove FKs referencing it
        """
        if table_name not in self.tables:
            raise ValueError(f"Table {table_name} not found")

        table = self.tables[table_name]

        # If removing a PK, cascade delete FKs in other tables
        if cascade and table.primary_key == column_name:
            for other_table in self.tables.values():
                if other_table.name == table_name:
                    continue

                # Find FKs referencing this PK
                fks_to_remove = [
                    fk_col for fk_col, ref_table in other_table.foreign_keys.items()
                    if ref_table == table_name
                ]

                for fk_col in fks_to_remove:
                    other_table.remove_column(fk_col)

        table.remove_column(column_name)
        self._cached_description = None


@dataclass
class PredictionTaskProfile:
    """Represents a prediction task profile"""
    name: str
    task_type: TaskType
    target_entity: str
    target_column: str
    entity_table: str
    docs: Optional[str] = None
    time_column: Optional[str] = None
    timedelta: Optional[str] = None

    @classmethod
    def from_relbench_task(cls, task: BaseTask, task_name: str) -> 'PredictionTaskProfile':
        """Create PredictionTaskProfile from a RBench BaseTask object"""
        # Get task type directly from relbench
        task_type = task.task_type

        # Extract task attributes
        name = task_name
        docs = getattr(task, '__doc__', None)
        # replace '\n' with space in docs
        if docs:       
            docs = docs.replace('\n', ' ')
            
        target_column = getattr(task, 'target_col', None)
        time_column = getattr(task, 'time_col', None)
        entity_table = getattr(task, 'entity_table', None)
        target_entity = getattr(task, 'entity_col', None)

        # Extract timedelta if available
        timedelta_str = None
        if hasattr(task, 'timedelta'):
            timedelta_str = str(task.timedelta)

        return cls(
            name=name,
            task_type=task_type,
            target_entity=target_entity or "unknown",
            target_column=target_column or "target",
            entity_table=entity_table or "main",
            docs=docs,
            time_column=time_column,
            timedelta=timedelta_str
        )

    @property
    def description(self) -> str:
        """Return the task description"""
        return self._generate_task_description()

    def _generate_task_description(self) -> str:
        """Generate a descriptive text for a prediction task"""
        descriptions = []

        descriptions.append(f"task type:{self.task_type}")
        if self.docs:
            descriptions.append(self.docs.strip())


        # Add target info
        target_info = []
        if self.target_column:
            target_info.append(f"predicting <{self.target_column}>")

        if self.entity_table:
            target_info.append(f"for entities in <{self.entity_table}> table")

        # Add temporal info if available
        if self.timedelta:
            target_info.append(f"with prediction horizon of <{self.timedelta}>")

        target_info = ', '.join(target_info)
        descriptions.append(target_info)
        
        return "\n".join(descriptions).capitalize()

    def to_task_profile(self):
        """
        Convert PredictionTaskProfile to TaskProfile for operator compatibility.

        Allows using predefined benchmark tasks with operators that expect TaskProfile.

        Returns:
            TaskProfile instance with fields mapped from PredictionTaskProfile
        """
        # Import here to avoid circular dependency
        from aida.query_analyzer.nl2task import TaskProfile

        # Use description or docs as natural language query
        nl_query = self.description if self.description else self.docs

        # Parse timedelta to days if it's a string representation
        time_duration = None
        if self.timedelta:
            try:
                if isinstance(self.timedelta, (int, float)):
                    time_duration = int(self.timedelta)
                elif isinstance(self.timedelta, str):
                    # Handle pandas Timedelta string format like "30 days" or "30 days 00:00:00"
                    import pandas as pd
                    td = pd.Timedelta(self.timedelta)
                    time_duration = int(td.days)
            except (ValueError, TypeError):
                # If parsing fails, leave as None
                time_duration = None

        return TaskProfile(
            nl_query=nl_query,
            task_type=self.task_type,
            entity_table=self.entity_table,
            time_duration=time_duration,
            target_col=self.target_column,
            entity_col=self.target_entity,
        )
