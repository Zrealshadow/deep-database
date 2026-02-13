"""
LLM prompt generators specifically for table selection tasks.

This module provides focused prompt generators for selecting relevant tables
from a database schema for prediction modeling tasks.
"""

from typing import Dict, List, Any, Optional
from aida.db.profile import DatabaseSchema


class TableSelectionPrompt:
    """
    Generates LLM prompts specifically for table selection
    """

    @staticmethod
    def generate_table_selection_prompt(
        task_profile,  # TaskProfile from nl2task
        database_schema: DatabaseSchema,
        max_tables: int = 10,
        include_examples: bool = True,
        focus_on_connectivity: bool = True
    ) -> str:
        """
        Generate a focused prompt for table selection only

        Args:
            task_profile: TaskProfile from BasicTaskParser or PredictionTaskProfile.to_task_profile()
            database_schema: Database schema information
            max_tables: Maximum number of tables to select
            include_examples: Whether to include example scenarios
            focus_on_connectivity: Whether to emphasize table connectivity

        Returns:
            LLM prompt string for table selection
        """

        prompt = f"""# Select Relevant Tables for Prediction Task

You are a database expert. Select the most relevant tables from the database schema to support the following prediction task.

## Prediction Task
{task_profile}
"""

        prompt += f"\n## Database Schema ({database_schema.name})\n"
        if database_schema.description:
            prompt += f"{database_schema.description}\n"

        # Add selection criteria
        entity_ref = task_profile.entity_table if task_profile.entity_table else "the main entity"
        prompt += f"""
## Selection Criteria

Choose up to {max_tables} tables that would be most valuable for the prediction task. Consider:

1. **Entity Relevance**: Tables containing or directly related to {entity_ref}
2. **Feature Potential**: Tables with columns that could serve as predictive features
3. **Data Completeness**: Tables likely to have good data quality and coverage
"""

        if focus_on_connectivity:
            prompt += "4. **Table Connectivity**: Tables that connect well with the entity table through relationships\n"

        # Add temporal alignment if time_duration is specified
        if hasattr(task_profile, 'time_duration') and task_profile.time_duration:
            prompt += "5. **Temporal Alignment**: Tables with appropriate temporal information\n"

        # output format
        prompt += f"""
## Required Output Format

Return a list of selected table names in JSON format as follows:
```json
{{
    "table_name_1":{{}},
    "table_name_2":{{}},
    ...
    "table_name_N":{{}},
}}
"""

        if include_examples:
            prompt += """
## Selection Examples

### High Priority Tables:
- **Entity table itself**: Always include the main entity table
- **Direct relationship tables**: Tables with foreign keys to/from entity table
- **Activity/event tables**: Tables recording entity behaviors or interactions
- **Attribute tables**: Tables with additional features about entities

### Medium Priority Tables:
- **Related entity tables**: Tables about related entities (e.g., products for users)
- **Aggregated data tables**: Tables with pre-computed statistics
- **Context tables**: Tables providing environmental or contextual information

### Low Priority Tables:
- **Reference/lookup tables**: Small tables with mostly categorical mappings
- **System tables**: Tables for technical/administrative purposes
- **Sparse tables**: Tables with many missing values or limited coverage
"""

        prompt += """
## Important Notes

- Focus on tables that directly support the prediction objective
- Prioritize data quality and completeness over quantity
- Consider the computational cost vs. predictive value trade-off
- Ensure temporal consistency for time-based predictions

Please analyze the schema and provide your table selections in the specified JSON format.
"""

        return prompt

    @staticmethod
    def get_system_prompt() -> str:
        """Get system prompt for table selection."""
        return """You are a database expert specializing in schema analysis and table selection.

Your task is to identify the most relevant tables for a prediction task by analyzing:
- Entity relevance
- Feature potential
- Data completeness
- Table connectivity
- Temporal alignment

Return valid JSON with selected table names as keys."""
