"""
Evaluation Dataset for BasicTaskParser.

Contains:
1. Positive cases: Valid prediction tasks from DatabaseFactory
2. Negative cases: Invalid queries that should fail parsing
"""

import random
from datetime import timedelta

from relbench.base import TaskType
from utils.data import DatabaseFactory


# Task-specific action/metric mappings
TASK_ACTIONS = {
    "user-clicks": ("click", "clicks"),
    "ad-ctr": ("click-through rate", "CTR"),
    "user-visits": ("visit", "visits"),
    "study-outcome": ("succeed", "outcome"),
    "site-success": ("be successful", "success"),
    "study-adverse": ("have adverse events", "adverse events"),
    "driver-dnf": ("not finish (DNF)", "DNF"),
    "driver-top3": ("finish in top 3", "top 3 finish"),
    "user-churn": ("churn", "churn"),
    "item-churn": ("churn", "churn"),
    "user-ltv": ("spend", "lifetime value"),
    "item-ltv": ("sell", "sales value"),
    "item-sales": ("sell", "sales"),
}

# Query templates for positive cases
BINARY_TEMPLATES = [
    "Predict whether {entity} will {action} in the next {duration}",
    "Determine if {entity} is going to {action} within {duration}",
    "Will {entity} {action} in the next {duration}?",
    "Forecast if {entity} will {action} over the next {duration}",
    "Is {entity} likely to {action} in the next {duration}?",
    "Build a model to predict if {entity} will {action} in the next {duration}",
    "Can you predict whether {entity} will {action} within {duration}?",
    "I need to predict if {entity} will {action} in the next {duration}",
    "Classify whether {entity} will {action} in the next {duration}",
    "Identify if {entity} will {action} over the next {duration}",
]

REGRESSION_TEMPLATES = [
    "Predict how much {entity} will {action} in the next {duration}",
    "Forecast the {metric} for {entity} in the next {duration}",
    "Estimate {entity}'s {metric} over the next {duration}",
    "How much will {entity} {action} in the next {duration}?",
    "Calculate the expected {metric} for {entity} in the next {duration}",
    "Predict the total {metric} for {entity} over the next {duration}",
    "Estimate how much {entity} will {action} within {duration}",
    "Forecast {entity}'s {metric} for the next {duration}",
    "What will be the {metric} for {entity} in the next {duration}?",
]

BINARY_NO_TIME = [
    "Predict whether {entity} will {action}",
    "Will {entity} {action}?",
    "Determine if {entity} is going to {action}",
    "Classify whether {entity} will {action}",
    "Is {entity} likely to {action}?",
    "Forecast if {entity} will {action}",
]

REGRESSION_NO_TIME = [
    "Predict the {metric} for {entity}",
    "Forecast {entity}'s {metric}",
    "Estimate {entity}'s {metric}",
    "How much will {entity} {action}?",
    "Calculate the {metric} for {entity}",
    "What will be the {metric} for {entity}?",
]

# Negative cases: queries that should fail parsing
# Format: (query, failure_reason)
NEGATIVE_CASES = [
    # Not prediction tasks - analysis/exploration
    ("Analyze the distribution of user clicks", "analysis_not_prediction"),
    ("Show me the data schema", "exploration_not_prediction"),
    ("What tables are in the database?", "exploration_not_prediction"),
    ("Describe the relationship between users and orders", "exploration_not_prediction"),
    ("Count how many users we have", "aggregation_not_prediction"),
    ("Summarize the user demographics", "analysis_not_prediction"),
    ("Generate a report of user activity", "reporting_not_prediction"),
    ("Visualize the click patterns", "visualization_not_prediction"),
    ("Explain the data trends", "explanation_not_prediction"),
    ("Find anomalies in the data", "detection_not_prediction"),

    # Ambiguous task type
    ("Tell me about user behavior", "ambiguous_no_target"),
    ("What can you predict from this data?", "ambiguous_no_target"),
    ("Help me understand the users", "ambiguous_no_target"),
    ("Give me insights about the users", "ambiguous_no_target"),
    ("What patterns exist in the data?", "ambiguous_no_target"),

    # No clear entity
    ("Predict the future", "no_entity"),
    ("Make a prediction for next week", "no_entity"),
    ("Forecast something useful", "no_entity"),
    ("Predict outcomes", "no_entity"),
    ("Tell me what will happen", "no_entity"),

    # Invalid/non-existent table references
    ("Predict if the customer will churn", "invalid_table"),  # 'customer' may not exist
    ("Forecast sales for the store table", "invalid_table"),
    ("Will the merchant make a purchase?", "invalid_table"),
    ("Predict if the subscriber will renew", "invalid_table"),

    # Not a question/request
    ("The weather is nice today", "irrelevant"),
    ("Hello, how are you?", "irrelevant"),
    ("Thanks for the help", "irrelevant"),
    ("Machine learning is interesting", "irrelevant"),
    ("I like data science", "irrelevant"),

    # Too vague
    ("Predict user clicks sometime", "vague_query"),
    ("Will users click eventually?", "vague_query"),
    ("Forecast user behavior in the future", "vague_query"),
    ("Predict something about users", "vague_query"),

    # Multiple conflicting entities
    ("Predict if users or products will succeed", "multiple_entities"),
    ("Forecast both user engagement and product sales", "multiple_entities"),

    # # Requests for multiple predictions
    ("Predict user clicks and user visits next week", "multiple_tasks"),
    ("Forecast both churn and revenue for users", "multiple_tasks"),

    # Data manipulation requests (not prediction)
    ("Update the user table", "data_manipulation"),
    ("Delete inactive users", "data_manipulation"),
    ("Insert new records into the database", "data_manipulation"),
    ("Join the user and order tables", "data_manipulation"),

    # Model/algorithm requests (not task definition)
    ("Use XGBoost to predict user churn", "algorithm_specification"),
    ("Train a neural network on the data", "algorithm_specification"),
    ("Apply logistic regression", "algorithm_specification"),

    # Impossibly complex or multi-step requests
    ("First segment users, then predict clicks for each segment", "multi_step_task"),
    ("Create features and then predict churn", "multi_step_task"),

    # Requests for explanation rather than prediction
    ("Why do users churn?", "explanation_request"),
    ("What causes users to click?", "explanation_request"),
    ("Explain user behavior patterns", "explanation_request"),

    # Comparative analysis (not prediction)
    ("Compare users who click vs don't click", "comparative_analysis"),
    ("Which users are more likely to churn?", "comparative_analysis"),

    # Time-related ambiguity
    ("Predict user clicks yesterday", "past_prediction"),
    ("Forecast what happened last month", "past_prediction"),

    # Nonsensical or contradictory
    ("Predict if the table will click", "nonsensical"),
    ("Will the database be active?", "nonsensical"),
]


# Define all possible time durations with their natural language variations
DURATION_VARIATIONS = {
    1: ["day", "24 hours", "1 day", "tomorrow"],
    2: ["2 days", "48 hours", "two days", "couple of days"],
    3: ["3 days", "72 hours", "three days"],
    5: ["5 days", "five days", "work week"],
    7: ["week", "7 days", "one week", "next week"],
    10: ["10 days", "ten days"],
    14: ["2 weeks", "14 days", "two weeks", "fortnight", "next 2 weeks"],
    21: ["3 weeks", "21 days", "three weeks"],
    28: ["4 weeks", "28 days", "four weeks"],
    30: ["month", "30 days", "one month", "next month"],
    45: ["45 days", "6 weeks", "month and a half"],
    60: ["2 months", "60 days", "two months", "next 2 months"],
    90: ["quarter", "90 days", "3 months", "one quarter", "next quarter"],
    120: ["4 months", "120 days", "four months"],
    180: ["half year", "180 days", "6 months", "next 6 months"],
    365: ["year", "365 days", "12 months", "one year", "next year"],
}

# Time duration options to test - derived from DURATION_VARIATIONS keys
TIME_DURATION_OPTIONS = list(DURATION_VARIATIONS.keys())


def format_duration(days=None):
    """
    Convert days to natural language with multiple variations.

    Args:
        days: Number of days (if None, randomly selects one)

    Returns:
        Tuple of (formatted_string, actual_days)
    """
    # If no days specified, randomly select one
    if days is None:
        days = random.choice(TIME_DURATION_OPTIONS)

    # Direct mapping: if in variations, pick random variation
    if days in DURATION_VARIATIONS:
        return random.choice(DURATION_VARIATIONS[days]), days

    # Otherwise, use default format
    return f"{days} days", days


def generate_queries_with_time_variations(
    task_name,
    task_type,
    entity_table,
    original_time_duration,
    num_variations=3
):
    """
    Generate multiple queries for a task, each with a different time duration.

    Args:
        task_name: Task name
        task_type: TaskType
        entity_table: Entity table name
        original_time_duration: Original task time duration (used as one option)
        num_variations: Number of different time durations to generate

    Returns:
        List of (query, time_duration) tuples
    """
    action, metric = TASK_ACTIONS.get(task_name, ("perform action", "value"))
    entity = entity_table.replace("_", " ")

    # Select time durations to test - sample num_variations times
    num_sampling_time = num_variations
    selected_times = random.sample(TIME_DURATION_OPTIONS, min(num_sampling_time, len(TIME_DURATION_OPTIONS)))

    queries = []
    for time_duration in selected_times:
        if time_duration:
            # Generate time expression
            duration_str, actual_days = format_duration(time_duration)

            # Pick template
            if task_type == TaskType.BINARY_CLASSIFICATION:
                templates = BINARY_TEMPLATES
            else:
                templates = REGRESSION_TEMPLATES

            template = random.choice(templates)
            query = template.format(entity=entity, action=action, metric=metric, duration=duration_str)
            queries.append((query, actual_days))
        else:
            # No time window
            if task_type == TaskType.BINARY_CLASSIFICATION:
                templates = BINARY_NO_TIME
            else:
                templates = REGRESSION_NO_TIME

            template = random.choice(templates)
            query = template.format(entity=entity, action=action, metric=metric, duration="")
            queries.append((query, None))

    return queries


def generate_nl_query(task_name, task_type, entity_table, time_duration, mode="random"):
    """
    Generate natural language query/queries from task info.

    Args:
        task_name: Task name
        task_type: TaskType (BINARY_CLASSIFICATION or REGRESSION)
        entity_table: Entity table name
        time_duration: Time duration in days (or None)
        mode: "random" (single query with original time) or "all" (multiple queries with varied times)

    Returns:
        Single (query, time_duration) tuple if mode="random"
        List of (query, time_duration) tuples if mode="all"
    """
    if mode == "all":
        # Generate multiple queries with different time durations
        num_variation_of_time = 5
        return generate_queries_with_time_variations(
            task_name, task_type, entity_table, time_duration, num_variations=num_variation_of_time
        )
    else:
        # Generate single query with original time duration
        action, metric = TASK_ACTIONS.get(task_name, ("perform action", "value"))
        entity = entity_table.replace("_", " ")

        if time_duration:
            duration_str, actual_days = format_duration(time_duration)
            if task_type == TaskType.BINARY_CLASSIFICATION:
                template = random.choice(BINARY_TEMPLATES)
            else:
                template = random.choice(REGRESSION_TEMPLATES)
            query = template.format(entity=entity, action=action, metric=metric, duration=duration_str)
            return query, actual_days
        else:
            if task_type == TaskType.BINARY_CLASSIFICATION:
                template = random.choice(BINARY_NO_TIME)
            else:
                template = random.choice(REGRESSION_NO_TIME)
            query = template.format(entity=entity, action=action, metric=metric, duration="")
            return query, None


def collect_positive_cases(db_names=None, mode="random"):
    """
    Collect positive cases from DatabaseFactory.

    Args:
        db_names: List of database names (None = all registered)
        mode: "random" (one query per task) or "all" (all template variations)
    """
    if db_names is None:
        db_names = DatabaseFactory.get_registered_databases()

    cases = []

    for db_name in db_names:
        task_names = DatabaseFactory.get_registered_tasks(db_name)
        if not task_names:
            continue

        try:
            ds = DatabaseFactory.get_dataset(db_name)
        except Exception as e:
            print(f"Warning: Could not load {db_name}: {e}")
            continue

        for task_name in task_names:
            try:
                task = DatabaseFactory.get_task(db_name, task_name, ds)

                time_duration = None
                if hasattr(task, 'timedelta') and task.timedelta:
                    if isinstance(task.timedelta, timedelta):
                        time_duration = task.timedelta.days

                result = generate_nl_query(
                    task_name, task.task_type, task.entity_table, time_duration,
                    mode=mode
                )

                # Handle both single query and list of queries
                if isinstance(result, list):
                    # mode="all" - list of (query, actual_days) tuples
                    for idx, (nl_query, actual_days) in enumerate(result):
                        cases.append({
                            "db_name": db_name,
                            "task_name": f"{task_name}_var{idx+1}",
                            "nl_query": nl_query,
                            "expected_success": True,
                            "task_type": task.task_type,
                            "entity_table": task.entity_table,
                            "time_duration": actual_days,  # Use actual generated time
                        })
                    print(f"  [+] {db_name}/{task_name} ({len(result)} variations)")
                else:
                    # mode="random" - single (query, actual_days) tuple
                    nl_query, actual_days = result
                    cases.append({
                        "db_name": db_name,
                        "task_name": task_name,
                        "nl_query": nl_query,
                        "expected_success": True,
                        "task_type": task.task_type,
                        "entity_table": task.entity_table,
                        "time_duration": actual_days,  # Use actual generated time
                    })
                    print(f"  [+] {db_name}/{task_name}")

            except Exception as e:
                print(f"Warning: Could not load {db_name}/{task_name}: {e}")

    return cases


def collect_negative_cases(db_name="avito"):
    """Collect negative cases that should fail parsing."""
    cases = []

    for query, reason in NEGATIVE_CASES:
        cases.append({
            "db_name": db_name,
            "task_name": f"negative_{reason}",
            "nl_query": query,
            "expected_success": False,
            "failure_reason": reason,
            "task_type": None,
            "entity_table": None,
            "time_duration": None,
        })
        print(f"  [-] {reason}: {query[:50]}...")

    return cases


def collect_eval_dataset(db_names=None, include_negative=True, mode="all", sample_size=10):
    """
    Collect evaluation dataset.

    Args:
        db_names: Database names for positive cases
        include_negative: Whether to include negative cases
        mode: "random" (sample for dry-run) or "all" (full dataset)
        sample_size: Number of samples to use in random mode (default: 10)
    """
    print("Collecting positive cases...")
    # Pass through the mode parameter to generate appropriate number of queries
    dataset = collect_positive_cases(db_names, mode=mode)

    if include_negative:
        print("\nCollecting negative cases...")
        neg_db = db_names[0] if db_names else "avito"
        dataset.extend(collect_negative_cases(neg_db))

    # If random mode, sample the dataset
    if mode == "random":
        print(f"\nRandom mode: Sampling {sample_size} cases for dry-run...")
        if len(dataset) > sample_size:
            dataset = random.sample(dataset, sample_size)

    return dataset
