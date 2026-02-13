"""
Noise Generator for DatabaseSchema Evaluation

Generates noised versions of DatabaseSchema for testing table/feature selection operators.

The ground truth is the original clean schema. We add noise (irrelevant tables and columns)
to create a noised version, then test if operators can denoise it back to the ground truth.
"""

import copy
import random
from typing import Dict, List, Optional
from dataclasses import dataclass

from aida.db.profile import DatabaseSchema, TableSchema


@dataclass
class NoiseConfig:
    """Configuration for noise generation."""
    # Table noise
    num_noise_tables: int = 5  # Number of irrelevant tables to add
    noise_table_prefix: str = "noise_"  # Prefix for noise table names

    # Column noise - supports both fixed count and range sampling
    num_noise_columns_per_table: int = 3  # Fixed number (for backward compatibility)
    noise_columns_range: Optional[tuple] = None  # (min, max) for range sampling

    # Linking strategy for noise tables
    linking_strategy: str = "random"  # Options: "none", "random", "all"

    # Random seed for reproducibility
    random_seed: Optional[int] = 42

    # Metadata
    difficulty_level: Optional[str] = None  # "A", "B", "C"


class NoiseGenerator:
    """
    Generates noised DatabaseSchema for evaluation.

    Usage:
        >>> generator = NoiseGenerator()
        >>> ground_truth = original_schema
        >>> noised_schema = generator.add_noise(original_schema, config)
        >>> # Test operator
        >>> result = operator(noised_schema)
        >>> # Compare result with ground_truth
    """

    # Common noise column names (system/metadata columns)
    NOISE_COLUMN_NAMES = [
        # Audit columns
        "created_at", "created_by", "created_date", "creation_date",
        "updated_at", "updated_by", "modified_at", "modified_by", "last_modified",
        "deleted_at", "deleted_by", "deletion_date",

        # System columns
        "revision", "version", "row_version", "etag",
        "uuid", "guid", "internal_id",

        # Metadata
        "metadata", "tags", "attributes", "properties",
        "audit_log", "change_log", "history",

        # Technical
        "is_active", "is_deleted", "is_archived", "status_flag",
        "sync_timestamp", "last_sync", "import_date", "export_date",
    ]

    # Noise table themes for generating realistic-looking irrelevant tables
    NOISE_TABLE_THEMES = [
        # System/Admin tables
        ("system_logs", ["log_id", "timestamp", "level", "message", "user_id"]),
        ("audit_trail", ["audit_id", "entity_type", "entity_id", "action", "timestamp", "user_id"]),
        ("config_settings", ["setting_id", "key", "value", "description", "updated_at"]),
        ("user_sessions", ["session_id", "user_id", "start_time", "end_time", "ip_address"]),

        # Internal/Technical tables
        ("migration_history", ["migration_id", "version", "applied_at", "status"]),
        ("queue_jobs", ["job_id", "queue_name", "payload", "status", "created_at"]),
        ("cache_entries", ["cache_key", "value", "expiry", "created_at"]),
        ("error_logs", ["error_id", "timestamp", "error_type", "stack_trace", "user_id"]),

        # Reference/Lookup tables (often not useful for prediction)
        ("countries", ["country_id", "country_code", "country_name"]),
        ("currencies", ["currency_id", "currency_code", "symbol"]),
        ("timezones", ["timezone_id", "timezone_name", "offset"]),
        ("languages", ["language_id", "language_code", "language_name"]),
    ]

    def __init__(self, random_seed: Optional[int] = 42):
        """Initialize noise generator with random seed."""
        self.random_seed = random_seed
        self.random = random.Random(random_seed)

    def add_noise(
        self,
        db_schema: DatabaseSchema,
        config: Optional[NoiseConfig] = None
    ) -> DatabaseSchema:
        """
        Add noise to DatabaseSchema.

        Args:
            db_schema: Original clean DatabaseSchema
            config: Noise configuration

        Returns:
            Noised DatabaseSchema (deep copy with added noise)
        """
        if config is None:
            config = NoiseConfig(random_seed=self.random_seed)

        # Reset random seed for reproducibility
        if config.random_seed is not None:
            self.random = random.Random(config.random_seed)

        # Deep copy to avoid modifying original
        noised_schema = copy.deepcopy(db_schema)

        # Add noise tables
        if config.num_noise_tables > 0:
            self._add_noise_tables(noised_schema, config)

        # Add noise columns to existing tables
        self._add_noise_columns(noised_schema, config)

        # Clear cached description
        noised_schema._cached_description = None

        return noised_schema

    def _add_noise_tables(self, db_schema: DatabaseSchema, config: NoiseConfig):
        """Add irrelevant noise tables to schema with FK connections."""
        # Get original tables (before adding noise)
        original_tables = list(db_schema.tables.keys())

        # Sample noise table themes
        num_tables = min(config.num_noise_tables, len(self.NOISE_TABLE_THEMES))
        selected_themes = self.random.sample(self.NOISE_TABLE_THEMES, num_tables)

        added_tables = []

        for i, (base_name, columns) in enumerate(selected_themes):
            table_name = f"{config.noise_table_prefix}{base_name}"

            # Ensure unique name
            counter = 1
            original_name = table_name
            while table_name in db_schema.tables:
                table_name = f"{original_name}_{counter}"
                counter += 1

            # Create noise table (without FKs first)
            noise_table = TableSchema(
                name=table_name,
                columns=columns.copy(),
                primary_key=columns[0] if columns else None,
                foreign_keys={},
                time_column=None,
                sample_size=self.random.randint(100, 10000)
            )

            # Add table first (needed for both linking approaches)
            db_schema.add_table(noise_table)
            added_tables.append(table_name)

        # Apply linking strategy
        if not original_tables or config.linking_strategy == 'none':
            return

        # Determine link probability: 100% for 'all', 70% for 'random' (default)
        link_probability = 1.0 if config.linking_strategy == 'all' else 0.7

        for noise_table_name in added_tables:
            if self.random.random() >= link_probability:
                continue

            # Randomly choose direction: noise→existing or existing→noise
            if self.random.random() < 0.5:
                noise_table = db_schema.tables[noise_table_name]
                self._connect_noise_to_existing_inplace(noise_table, db_schema, original_tables)
            else:
                self._connect_existing_to_noise(noise_table_name, db_schema, original_tables)

    def _connect_noise_to_existing_inplace(
        self,
        noise_table: TableSchema,
        db_schema: DatabaseSchema,
        original_tables: List[str]
    ):
        """Approach A: Add FK from noise table to existing table (modifies table in-place)."""
        # Select random existing table to reference
        target_table_name = self.random.choice(original_tables)
        target_table = db_schema.tables[target_table_name]

        if not target_table.primary_key:
            return  # Skip if target has no PK

        # Add FK column to noise table
        fk_column = f"{target_table_name}_id"
        noise_table.columns.append(fk_column)
        noise_table.foreign_keys[fk_column] = target_table_name

    def _connect_existing_to_noise(
        self,
        noise_table_name: str,
        db_schema: DatabaseSchema,
        original_tables: List[str]
    ):
        """Approach B: Add FK from existing table to noise table."""
        # Select random existing table to add FK column to
        target_table_name = self.random.choice(original_tables)
        noise_table = db_schema.tables[noise_table_name]

        if not noise_table.primary_key:
            return  # Skip if noise table has no PK

        # Add FK column to existing table
        fk_column = f"{noise_table_name}_id"

        # Use the API with validation
        db_schema.add_column_to_table(
            table_name=target_table_name,
            column_name=fk_column,
            is_foreign_key=True,
            references=(noise_table_name, noise_table.primary_key)
        )

    def _add_noise_columns(self, db_schema: DatabaseSchema, config: NoiseConfig):
        """Add irrelevant noise columns to existing tables."""
        for table_name, table in db_schema.tables.items():
            # Skip noise tables (they're already irrelevant)
            if table_name.startswith(config.noise_table_prefix):
                continue

            # Select random noise columns that don't already exist
            available_noise_cols = [
                col for col in self.NOISE_COLUMN_NAMES
                if col not in table.columns
            ]

            if not available_noise_cols:
                continue

            # Determine number of columns to add
            if config.noise_columns_range is not None:
                # Sample from range
                min_cols, max_cols = config.noise_columns_range
                num_to_add = self.random.randint(min_cols, max_cols + 1)
            else:
                # Use fixed count (backward compatibility)
                num_to_add = config.num_noise_columns_per_table

            # Cap at available columns
            num_to_add = min(num_to_add, len(available_noise_cols))

            # Sample random noise columns
            if num_to_add > 0:
                noise_cols = self.random.sample(available_noise_cols, num_to_add)

                # Use API to add columns
                for col in noise_cols:
                    table.add_column(col, is_foreign_key=False)


def generate_noised_schema(
    original_schema: DatabaseSchema,
    num_noise_tables: int = 5,
    num_noise_columns_per_table: int = 3,
    random_seed: Optional[int] = 42
) -> DatabaseSchema:
    """
    Convenience function to generate noised schema.

    Args:
        original_schema: Clean DatabaseSchema
        num_noise_tables: Number of irrelevant tables to add
        num_noise_columns_per_table: Number of irrelevant columns per table
        random_seed: Random seed for reproducibility

    Returns:
        Noised DatabaseSchema
    """
    config = NoiseConfig(
        num_noise_tables=num_noise_tables,
        num_noise_columns_per_table=num_noise_columns_per_table,
        random_seed=random_seed
    )
    generator = NoiseGenerator(random_seed=random_seed)
    return generator.add_noise(original_schema, config)
