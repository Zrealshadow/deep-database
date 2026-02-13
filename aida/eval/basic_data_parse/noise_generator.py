"""
Noise Generator for DatabaseSchema Evaluation

Generates noised versions of DatabaseSchema for testing table/feature selection operators.

The ground truth is the original clean schema. We add noise (irrelevant tables and columns)
to create a noised version, then test if operators can denoise it back to the ground truth.
"""

import copy
import random
from typing import List, Dict, Optional, Set
from dataclasses import dataclass

from aida.db.profile import DatabaseSchema, TableSchema


@dataclass
class NoiseConfig:
    """Configuration for noise generation."""
    # Table noise
    num_noise_tables: int = 5  # Number of irrelevant tables to add
    noise_table_prefix: str = "noise_"  # Prefix for noise table names

    # Column noise
    num_noise_columns_per_table: int = 3  # Irrelevant columns per table

    # Random seed for reproducibility
    random_seed: Optional[int] = 42


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
        if random_seed is not None:
            random.seed(random_seed)

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
            random.seed(config.random_seed)

        # Deep copy to avoid modifying original
        noised_schema = copy.deepcopy(db_schema)

        # Add noise tables
        if config.num_noise_tables > 0:
            self._add_noise_tables(noised_schema, config)

        # Add noise columns to existing tables
        if config.num_noise_columns_per_table > 0:
            self._add_noise_columns(noised_schema, config)

        # Clear cached description
        noised_schema._cached_description = None

        return noised_schema

    def _add_noise_tables(self, db_schema: DatabaseSchema, config: NoiseConfig):
        """Add irrelevant noise tables to schema with FK connections."""
        # Get original tables (before adding noise)
        original_tables = [t for t in db_schema.tables.keys()]

        # Sample noise table themes
        num_tables = min(config.num_noise_tables, len(self.NOISE_TABLE_THEMES))
        selected_themes = random.sample(self.NOISE_TABLE_THEMES, num_tables)

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
                sample_size=random.randint(100, 10000)
            )

            # Randomly choose FK connection strategy (if there are original tables)
            if original_tables and random.random() < 0.7:  # 70% chance to connect
                if random.random() < 0.5:
                    # Approach A: FK from noise table → existing table
                    self._connect_noise_to_existing(noise_table, db_schema, original_tables)
                    db_schema.add_table(noise_table)
                else:
                    # Approach B: FK from existing table → noise table
                    # Must add table first, then create FK from existing table
                    db_schema.add_table(noise_table)
                    self._connect_existing_to_noise(noise_table.name, db_schema, original_tables)
            else:
                # No FK connection, just add the table
                db_schema.add_table(noise_table)

    def _connect_noise_to_existing(
        self,
        noise_table: TableSchema,
        db_schema: DatabaseSchema,
        original_tables: List[str]
    ):
        """Approach A: Add FK from noise table to existing table."""
        # Select random existing table to reference
        target_table_name = random.choice(original_tables)
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
        target_table_name = random.choice(original_tables)
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

            # Add random noise columns
            num_to_add = min(config.num_noise_columns_per_table, len(available_noise_cols))
            noise_cols = random.sample(available_noise_cols, num_to_add)

            # Use API to add columns
            for col in noise_cols:
                table.add_column(col, is_foreign_key=False)

    def get_ground_truth_tables(self, original_schema: DatabaseSchema) -> Set[str]:
        """
        Get set of ground truth table names.

        Args:
            original_schema: Original clean schema

        Returns:
            Set of table names in ground truth
        """
        return set(original_schema.tables.keys())

    def get_ground_truth_columns(
        self,
        original_schema: DatabaseSchema
    ) -> Dict[str, Set[str]]:
        """
        Get ground truth columns for each table.

        Args:
            original_schema: Original clean schema

        Returns:
            Dict mapping table name to set of column names
        """
        return {
            table_name: set(table.columns)
            for table_name, table in original_schema.tables.items()
        }

    def calculate_metrics(
        self,
        predicted_schema: DatabaseSchema,
        ground_truth_schema: DatabaseSchema,
        level: str = "table"
    ) -> Dict[str, float]:
        """
        Calculate precision, recall, F1 for table or column selection.

        Args:
            predicted_schema: Schema after operator filtering
            ground_truth_schema: Original clean schema
            level: "table" or "column"

        Returns:
            Dict with precision, recall, f1, and counts
        """
        if level == "table":
            return self._calculate_table_metrics(predicted_schema, ground_truth_schema)
        elif level == "column":
            return self._calculate_column_metrics(predicted_schema, ground_truth_schema)
        else:
            raise ValueError(f"Unknown level: {level}")

    def _calculate_table_metrics(
        self,
        predicted_schema: DatabaseSchema,
        ground_truth_schema: DatabaseSchema
    ) -> Dict[str, float]:
        """Calculate metrics for table selection."""
        predicted_tables = set(predicted_schema.tables.keys())
        ground_truth_tables = set(ground_truth_schema.tables.keys())

        true_positives = predicted_tables & ground_truth_tables
        false_positives = predicted_tables - ground_truth_tables
        false_negatives = ground_truth_tables - predicted_tables

        precision = len(true_positives) / len(predicted_tables) if predicted_tables else 0.0
        recall = len(true_positives) / len(ground_truth_tables) if ground_truth_tables else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": len(true_positives),
            "false_positives": len(false_positives),
            "false_negatives": len(false_negatives),
            "predicted_count": len(predicted_tables),
            "ground_truth_count": len(ground_truth_tables),
        }

    def _calculate_column_metrics(
        self,
        predicted_schema: DatabaseSchema,
        ground_truth_schema: DatabaseSchema
    ) -> Dict[str, float]:
        """Calculate metrics for column selection (aggregated across tables)."""
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_predicted = 0
        total_ground_truth = 0

        # Only evaluate tables that exist in ground truth
        for table_name in ground_truth_schema.tables.keys():
            gt_columns = set(ground_truth_schema.tables[table_name].columns)
            total_ground_truth += len(gt_columns)

            if table_name in predicted_schema.tables:
                pred_columns = set(predicted_schema.tables[table_name].columns)
                total_predicted += len(pred_columns)

                tp = pred_columns & gt_columns
                fp = pred_columns - gt_columns
                fn = gt_columns - pred_columns

                total_tp += len(tp)
                total_fp += len(fp)
                total_fn += len(fn)
            else:
                # Table not selected - all ground truth columns are false negatives
                total_fn += len(gt_columns)

        precision = total_tp / total_predicted if total_predicted > 0 else 0.0
        recall = total_tp / total_ground_truth if total_ground_truth > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "predicted_count": total_predicted,
            "ground_truth_count": total_ground_truth,
        }


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
