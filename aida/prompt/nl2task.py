"""
NL2Task Prompt Templates (Part 1: Coarse-grained Extraction)

Extracts minimum required fields from natural language task descriptions:
- task_type: Classification or regression
- entity_table: The main entity table for prediction
- time_duration: Prediction horizon in days

Part 2 (SQL generation for label computation) is handled separately.
"""

from aida.db.profile import DatabaseSchema


class NL2TaskPrompt:
    """
    Generates prompts for NL to Task Profile conversion (Part 1).

    This is the first step of a two-step pipeline:
    1. Coarse-grained extraction (this step): task_type, entity_table, time_duration
    2. Fine-grained SQL generation (next step): label computation logic
    """

    SYSTEM_PROMPT = """You are an expert at analyzing natural language descriptions of machine learning prediction tasks.

Extract exactly 3 fields from the task description:
1. task_type: BINARY_CLASSIFICATION or REGRESSION
2. entity_table: The main table containing entities to predict
3. time_duration: Prediction time window in days (integer), or null if not specified

Respond with valid JSON only."""

    EXTRACTION_TEMPLATE = """Given the database schema and task description, extract the coarse-grained task profile.

## Database Schema
{schema_description}

## Task Description
"{nl_query}"

## Extraction Rules

**task_type** - Determine from keywords:
- Binary outcomes (yes/no, will/won't, is/isn't) → "BINARY_CLASSIFICATION"
- Numeric outcomes (how much, count, rate, amount, score) → "REGRESSION"

**entity_table** - The main prediction subject:
- Must be an EXACT table name from the schema above
- Identify what entity the prediction is about (users, products, orders, etc.)

**time_duration** - Convert to days (integer):
- "next week" → 7
- "next month" → 30
- "in 90 days" / "next quarter" → 90
- "next half year" → 180
- If no time window mentioned → null

## Output Format
{{
    "task_type": "BINARY_CLASSIFICATION" | "REGRESSION" | null,
    "entity_table": "<exact table name from schema>" | null,
    "time_duration": <integer days> | null
}}

Set a field to null if it cannot be determined from the query or schema."""

    FEW_SHOT_EXAMPLES = """
## Examples

### Example 1
Schema: UserInfo(UserID, Name), SearchStream(UserID, AdID, IsClick, SearchDate)
Query: "Predict if the user will click in the next week"
Output:
{"task_type": "BINARY_CLASSIFICATION", "entity_table": "UserInfo", "time_duration": 7}

### Example 2
Schema: Products(ProductID, Price), Sales(ProductID, Quantity, SaleDate)
Query: "Forecast total sales quantity for each product next month"
Output:
{"task_type": "REGRESSION", "entity_table": "Products", "time_duration": 30}

### Example 3
Schema: users(user_id, name), beer_ratings(user_id, beer_id, rating, created_at)
Query: "Predict whether a user will be active in the next 90 days"
Output:
{"task_type": "BINARY_CLASSIFICATION", "entity_table": "users", "time_duration": 90}

### Example 4
Schema: order(order_id, status), items(item_id, order_id)
Query: "Predict whether an order will be delayed"
Output:
{"task_type": "BINARY_CLASSIFICATION", "entity_table": "order", "time_duration": null}

### Example 5
Schema: customer(customer_id), review(customer_id, product_id, review_time)
Query: "Predict whether a driver will accept the ride"
Output:
{"task_type": "BINARY_CLASSIFICATION", "entity_table": null, "time_duration": null}

### Example 6
Schema: customer(customer_id), review(customer_id, product_id, review_time)
Query: "Analyze the data distribution"
Output:
{"task_type": null, "entity_table": null, "time_duration": null}
"""

    @classmethod
    def generate_extraction_prompt(
        cls,
        nl_query: str,
        db_schema: DatabaseSchema,
        include_examples: bool = True
    ) -> str:
        """
        Generate the extraction prompt.

        Args:
            nl_query: Natural language task description
            db_schema: Database schema for context
            include_examples: Whether to include few-shot examples

        Returns:
            Complete prompt string
        """
        prompt_parts = []

        if include_examples:
            prompt_parts.append(cls.FEW_SHOT_EXAMPLES)

        prompt_parts.append(
            cls.EXTRACTION_TEMPLATE.format(
                schema_description=db_schema.description,
                nl_query=nl_query
            )
        )

        return "\n".join(prompt_parts)

    @classmethod
    def get_system_prompt(cls) -> str:
        """Get the system prompt for the LLM."""
        return cls.SYSTEM_PROMPT
