"""
LLM prompt generators specifically for table selection tasks.

This module provides focused prompt generators for selecting relevant tables
from a database schema for prediction modeling tasks.
"""

from typing import Dict, List, Any, Optional
from aida.db.profile import DatabaseSchema, PredictionTaskProfile


class TableSelectionPrompt:
    """
    Generates LLM prompts specifically for table selection
    """

    @staticmethod
    def generate_table_selection_prompt(
        prediction_task: PredictionTaskProfile,
        database_schema: DatabaseSchema,
        max_tables: int = 10,
        include_examples: bool = True,
        focus_on_connectivity: bool = True
    ) -> str:
        """
        Generate a focused prompt for table selection only

        Args:
            prediction_task: The prediction task profile
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
**Name:** {prediction_task.name}
**Type:** {prediction_task.task_type.value}
**Description:** {prediction_task.description}
**Target:** Predict {prediction_task.target_column} for {prediction_task.target_entity} entities
**Entity Table:** {prediction_task.entity_table}
"""

        if prediction_task.time_column:
            prompt += f"**Time Column:** {prediction_task.time_column}\n"
        if prediction_task.timedelta:
            prompt += f"**Time Horizon:** {prediction_task.timedelta}\n"

        prompt += f"\n## Database Schema ({database_schema.name})\n"
        if database_schema.description:
            prompt += f"{database_schema.description}\n"

        # Add selection criteria
        prompt += f"""
## Selection Criteria

Choose up to {max_tables} tables that would be most valuable for the prediction task. Consider:

1. **Entity Relevance**: Tables containing or directly related to {prediction_task.target_entity}
2. **Feature Potential**: Tables with columns that could serve as predictive features
3. **Data Completeness**: Tables likely to have good data quality and coverage
"""

        if focus_on_connectivity:
            prompt += "4. **Table Connectivity**: Tables that connect well with the entity table through relationships\n"

        if prediction_task.time_column:
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

#     @staticmethod
#     def generate_table_ranking_prompt(
#         selected_tables: List[str],
#         database_schema: DatabaseSchema,
#         prediction_task: PredictionTaskProfile
#     ) -> str:
#         """
#         Generate a prompt to rank already selected tables by importance

#         Args:
#             selected_tables: List of pre-selected table names
#             database_schema: Database schema information
#             prediction_task: Prediction task profile

#         Returns:
#             LLM prompt for ranking tables
#         """

#         prompt = f"""# Rank Selected Tables by Importance

# You have {len(selected_tables)} pre-selected tables. Rank them by importance for the prediction task.

# ## Prediction Task
# **Target:** Predict {prediction_task.target_column} for {prediction_task.target_entity}
# **Type:** {prediction_task.task_type.value}

# ## Selected Tables
# """

#         for table_name in selected_tables:
#             if table_name in database_schema.tables:
#                 table = database_schema.tables[table_name]
#                 prompt += f"\n**{table_name}**\n"
#                 prompt += f"- {len(table.columns)} columns, {table.sample_size} rows\n"
#                 if table.description:
#                     prompt += f"- {table.description}\n"
#                 if table.foreign_keys:
#                     prompt += f"- Connected to: {', '.join(table.foreign_keys.values())}\n"

#         prompt += f"""
# ## Ranking Task

# Rank these tables from 1 (most important) to {len(selected_tables)} (least important) for the prediction task.

# ## Output Format

# ```json
# {{
#     "ranked_tables": [
#         {{
#             "rank": 1,
#             "table_name": "most_important_table",
#             "importance_score": 10,
#             "reasoning": "Why this table is most critical for the prediction"
#         }},
#         {{
#             "rank": 2,
#             "table_name": "second_table",
#             "importance_score": 8,
#             "reasoning": "Why this table is second most important"
#         }}
#     ]
# }}
# ```

# Provide your ranking with clear reasoning for each table's position.
# """

#         return prompt

#     @staticmethod
#     def generate_table_filtering_prompt(
#         prediction_task: PredictionTaskProfile,
#         database_schema: DatabaseSchema,
#         computational_budget: Optional[str] = None,
#         max_tables: int = 5
#     ) -> str:
#         """
#         Generate a prompt for filtering tables based on constraints

#         Args:
#             prediction_task: Prediction task profile
#             database_schema: Database schema information
#             computational_budget: Description of computational constraints
#             max_tables: Maximum number of tables allowed

#         Returns:
#             LLM prompt for table filtering
#         """

#         prompt = f"""# Filter Tables Under Constraints

# Select the {max_tables} most essential tables for the prediction task under given constraints.

# ## Prediction Task
# **Objective:** {prediction_task.description}
# **Target:** {prediction_task.target_column}
# **Entity:** {prediction_task.target_entity}

# ## Available Tables
# """

#         # Sort tables by size for constraint awareness
#         sorted_tables = sorted(
#             database_schema.tables.items(),
#             key=lambda x: x[1].sample_size or 0,
#             reverse=True
#         )

#         for table_name, table in sorted_tables:
#             prompt += f"- **{table_name}**: {table.sample_size} rows, {len(table.columns)} cols"
#             if table.foreign_keys:
#                 prompt += f" [Connected to: {', '.join(table.foreign_keys.values())}]"
#             prompt += "\n"

#         if computational_budget:
#             prompt += f"\n## Constraints\n{computational_budget}\n"

#         prompt += f"""
# ## Filtering Strategy

# Select exactly {max_tables} tables that provide maximum predictive value under the constraints.

# Prioritize:
# 1. Essential tables (entity table, direct relationships)
# 2. High-value, low-cost tables
# 3. Tables with strong predictive signals
# 4. Well-connected tables in the schema

# ## Output Format

# ```json
# {{
#     "filtered_tables": [
#         {{
#             "table_name": "essential_table",
#             "priority": "high",
#             "justification": "Why this table is essential despite constraints",
#             "estimated_cost": "low/medium/high"
#         }}
#     ],
#     "excluded_tables": [
#         {{
#             "table_name": "excluded_table",
#             "exclusion_reason": "Why this table was excluded"
#         }}
#     ]
# }}
# ```

# Focus on the minimum viable set of tables that still enables effective prediction.
# """

#         return prompt
