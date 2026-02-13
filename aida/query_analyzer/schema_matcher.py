"""
Schema Matcher for NL2Task

Validates and matches extracted entities from NL parsing against the actual database schema.
Handles entity resolution, column matching, and relationship validation.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from difflib import SequenceMatcher

from aida.db.profile import DatabaseSchema, TableSchema


@dataclass
class MatchResult:
    """Result of schema matching operation"""
    matched: bool
    entity_table: Optional[str] = None
    entity_col: Optional[str] = None
    time_column: Optional[str] = None
    confidence: float = 0.0
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class SchemaMatcher:
    """
    Matches extracted NL entities against database schema.

    Provides fuzzy matching for table/column names and validates
    that extracted entities exist in the schema.
    """

    def __init__(self, db_schema: DatabaseSchema, similarity_threshold: float = 0.6):
        """
        Initialize schema matcher.

        Args:
            db_schema: The database schema to match against
            similarity_threshold: Minimum similarity score for fuzzy matching (0-1)
        """
        self.db_schema = db_schema
        self.similarity_threshold = similarity_threshold

        # Build lookup indices
        self._table_names = list(db_schema.tables.keys())
        self._table_names_lower = {name.lower(): name for name in self._table_names}

    def match_entity_table(self, extracted_table: str) -> Tuple[Optional[str], float]:
        """
        Match extracted table name against schema.

        Args:
            extracted_table: Table name extracted from NL

        Returns:
            Tuple of (matched_table_name, confidence_score)
        """
        extracted_lower = extracted_table.lower()

        # Exact match (case-insensitive)
        if extracted_lower in self._table_names_lower:
            return self._table_names_lower[extracted_lower], 1.0

        # Fuzzy match
        best_match = None
        best_score = 0.0

        for table_name in self._table_names:
            score = self._similarity(extracted_lower, table_name.lower())
            if score > best_score and score >= self.similarity_threshold:
                best_score = score
                best_match = table_name

        return best_match, best_score

    def match_column(
        self,
        table_name: str,
        extracted_col: str
    ) -> Tuple[Optional[str], float]:
        """
        Match extracted column name against table schema.

        Args:
            table_name: The table to search in
            extracted_col: Column name extracted from NL

        Returns:
            Tuple of (matched_column_name, confidence_score)
        """
        if table_name not in self.db_schema.tables:
            return None, 0.0

        table = self.db_schema.tables[table_name]
        extracted_lower = extracted_col.lower()

        # Exact match
        for col in table.columns:
            if col.lower() == extracted_lower:
                return col, 1.0

        # Fuzzy match
        best_match = None
        best_score = 0.0

        for col in table.columns:
            score = self._similarity(extracted_lower, col.lower())
            if score > best_score and score >= self.similarity_threshold:
                best_score = score
                best_match = col

        return best_match, best_score

    def find_entity_column(self, table_name: str) -> Optional[str]:
        """
        Find the primary key / entity column for a table.

        Args:
            table_name: The table to find entity column for

        Returns:
            Primary key column name if found
        """
        if table_name not in self.db_schema.tables:
            return None

        table = self.db_schema.tables[table_name]
        return table.primary_key

    def find_time_column(self, table_name: str) -> Optional[str]:
        """
        Find the time column for a table.

        Args:
            table_name: The table to find time column for

        Returns:
            Time column name if found
        """
        if table_name not in self.db_schema.tables:
            return None

        table = self.db_schema.tables[table_name]
        return table.time_column

    def find_related_time_column(self, entity_table: str) -> Optional[Tuple[str, str]]:
        """
        Find a time column from related tables if entity table has no time column.

        Args:
            entity_table: The entity table name

        Returns:
            Tuple of (table_name, time_column) if found in related tables
        """
        # First check if entity table has time column
        if entity_table in self.db_schema.tables:
            table = self.db_schema.tables[entity_table]
            if table.time_column:
                return entity_table, table.time_column

        # Search in tables that reference this entity table
        for table_name, table in self.db_schema.tables.items():
            for fk_col, ref_table in table.foreign_keys.items():
                if ref_table == entity_table and table.time_column:
                    return table_name, table.time_column

        return None

    def validate_and_match(
        self,
        extracted_table: str,
        extracted_entity_col: Optional[str] = None,
        extracted_time_col: Optional[str] = None
    ) -> MatchResult:
        """
        Validate and match all extracted entities against schema.

        Args:
            extracted_table: Extracted entity table name
            extracted_entity_col: Extracted entity column (optional)
            extracted_time_col: Extracted time column (optional)

        Returns:
            MatchResult with matched entities and confidence
        """
        warnings = []

        # Match entity table
        matched_table, table_confidence = self.match_entity_table(extracted_table)
        if not matched_table:
            return MatchResult(
                matched=False,
                warnings=[f"Could not match table '{extracted_table}' to schema. "
                         f"Available tables: {', '.join(self._table_names)}"]
            )

        if table_confidence < 1.0:
            warnings.append(
                f"Table '{extracted_table}' fuzzy matched to '{matched_table}' "
                f"(confidence: {table_confidence:.2f})"
            )

        # Match or find entity column
        entity_col = None
        if extracted_entity_col:
            entity_col, col_conf = self.match_column(matched_table, extracted_entity_col)
            if not entity_col:
                warnings.append(
                    f"Could not match entity column '{extracted_entity_col}', "
                    f"using primary key instead"
                )

        if not entity_col:
            entity_col = self.find_entity_column(matched_table)
            if not entity_col:
                warnings.append(f"No primary key found for table '{matched_table}'")

        # Match or find time column
        time_col = None
        if extracted_time_col:
            time_col, _ = self.match_column(matched_table, extracted_time_col)

        if not time_col:
            time_result = self.find_related_time_column(matched_table)
            if time_result:
                _, time_col = time_result
            else:
                warnings.append(f"No time column found for table '{matched_table}' or related tables")

        return MatchResult(
            matched=True,
            entity_table=matched_table,
            entity_col=entity_col,
            time_column=time_col,
            confidence=table_confidence,
            warnings=warnings
        )

    def _similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity using SequenceMatcher"""
        return SequenceMatcher(None, s1, s2).ratio()
