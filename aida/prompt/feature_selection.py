"""
LLM prompt generators for feature/column selection tasks.

This module provides prompts for filtering out unnecessary columns
from database tables for prediction modeling.
"""

from typing import Dict, List, Any, Optional
from aida.db.profile import DatabaseSchema, TableSchema


class FeatureSelectionPrompt:
    """
    Generates LLM prompts for feature/column selection
    """

    @staticmethod
    def generate_feature_selection_prompt(
        selected_tables: List[str],
        database_schema: DatabaseSchema,
        entity_table: str,
        include_examples: bool = True
    ) -> str:
        """
        Generate a prompt for selecting relevant columns from tables.

        Filters out system/metadata columns like 'created_by', 'updated_at', etc.

        Args:
            selected_tables: List of table names to analyze
            database_schema: Database schema information
            entity_table: The main entity table for the prediction task
            include_examples: Whether to include example scenarios

        Returns:
            LLM prompt string for feature selection
        """

        prompt = f"""# Select Relevant Features for Prediction Task

You are a data science expert. For each table, select the columns that are likely to be useful for prediction modeling.

## Entity Table
**{entity_table}** is the main entity table for this prediction task.

## Tables to Analyze
"""

        # Add table details
        for table_name in selected_tables:
            if table_name in database_schema.tables:
                table = database_schema.tables[table_name]
                prompt += f"\n### {table_name}\n"
                prompt += f"**Columns ({len(table.columns)}):** {', '.join(table.columns)}\n"
                if table.primary_key:
                    prompt += f"**Primary Key:** {table.primary_key}\n"
                if table.foreign_keys:
                    fk_list = ', '.join([f"{fk}->{ref}" for fk, ref in table.foreign_keys.items()])
                    prompt += f"**Foreign Keys:** {fk_list}\n"
                if table.time_column:
                    prompt += f"**Time Column:** {table.time_column}\n"

        prompt += """
## Selection Criteria

For each table, **EXCLUDE** columns that are:

1. **System/Metadata columns**: created_at, created_by, updated_at, updated_by, modified_at, modified_by, deleted_at, deleted_by, version, revision
2. **Audit columns**: created_date, creation_date, last_modified, last_updated, audit_*, log_*
3. **Technical IDs**: uuid, guid (unless they are primary/foreign keys)
4. **Redundant columns**: Columns that duplicate information available elsewhere
5. **Empty/Sparse columns**: Columns with mostly null values (if known)

For each table, **INCLUDE** columns that are:

1. **Primary keys**: Always include (needed for joins)
2. **Foreign keys**: Always include (needed for relationships)
3. **Time columns**: Always include (needed for temporal features)
4. **Predictive features**: Domain-specific attributes that could predict the target
5. **Behavioral features**: User/entity actions, interactions, events
6. **Descriptive features**: Attributes that describe the entity or context
"""

        if include_examples:
            prompt += """
## Examples of Column Filtering

### Example 1: User Table
**Original columns:**
- user_id (PK)
- username
- email
- age
- registration_date
- created_at
- created_by
- updated_at
- last_login
- account_status

**Selected columns:**
- user_id (primary key - needed)
- age (predictive feature)
- registration_date (temporal feature)
- last_login (behavioral feature)
- account_status (descriptive feature)

**Excluded:**
- username (not predictive)
- email (PII, not predictive)
- created_at (system metadata)
- created_by (system metadata)
- updated_at (system metadata)

### Example 2: Order Table
**Original columns:**
- order_id (PK)
- user_id (FK)
- product_id (FK)
- order_date
- quantity
- price
- total_amount
- created_date
- modified_date
- created_by_user

**Selected columns:**
- order_id (primary key)
- user_id (foreign key)
- product_id (foreign key)
- order_date (time column)
- quantity (predictive feature)
- price (predictive feature)
- total_amount (predictive feature)

**Excluded:**
- created_date (system metadata)
- modified_date (system metadata)
- created_by_user (system metadata)
"""

        prompt += """
## Required Output Format

Return a JSON object mapping each table to its selected columns:

```json
{
    "table_name_1": ["column1", "column2", "column3"],
    "table_name_2": ["column1", "column2"],
    ...
}
```

## Important Notes

- Always include primary keys, foreign keys, and time columns
- Focus on columns that could have predictive value
- Remove system/audit columns that don't contribute to predictions
- Consider domain knowledge when selecting features
- When in doubt, include the column (better to have more features than miss important ones)

Please analyze each table and provide your feature selections in the specified JSON format.
"""

        return prompt

    @staticmethod
    def get_system_prompt() -> str:
        """Get system prompt for feature selection."""
        return """You are a data science expert specializing in feature engineering and database analysis.

Your task is to identify relevant columns for prediction modeling by filtering out:
- System metadata (created_at, updated_at, etc.)
- Audit columns
- Non-predictive technical fields

Always preserve:
- Primary keys (for entity identification)
- Foreign keys (for relationships)
- Time columns (for temporal features)
- Domain-specific predictive attributes

Return valid JSON with selected columns for each table."""
