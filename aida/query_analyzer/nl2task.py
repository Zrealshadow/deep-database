"""
NL2Task Parsers

Operators for converting natural language to task definitions.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass

try:
    from typing import override  # Python 3.12+
except ImportError:
    from typing_extensions import override  # Python < 3.12

from relbench.base import TaskType

from aida.db.profile import DatabaseSchema
from aida.prompt.nl2task import NL2TaskPrompt
from .base import BaseOperator, BaseResult
from .schema_matcher import SchemaMatcher


@dataclass
class TaskProfile:
    """
    Task profile artifact.

    Pure data container - no validation logic.
    Operators fill different fields:
    - BasicTaskParser: nl_query, task_type, entity_table, time_duration
    - SQLQueryGenerator: sql_template, target_col, entity_col
    """
    # Original natural language query
    nl_query: Optional[str] = None

    # Part 1: Coarse-grained task definition
    task_type: Optional[TaskType] = None
    entity_table: Optional[str] = None
    time_duration: Optional[int] = None  # days

    # Part 2: SQL generation
    sql_template: Optional[str] = None
    target_col: Optional[str] = None
    entity_col: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nl_query": self.nl_query,
            "task_type": str(self.task_type) if self.task_type else None,
            "entity_table": self.entity_table,
            "time_duration": self.time_duration,
            "sql_template": self.sql_template,
            "target_col": self.target_col,
            "entity_col": self.entity_col,
        }

    def __format__(self, format_spec: str = '') -> str:
        """
        Format task profile as a human-readable description.
        Automatically filters out null attributes.

        Usage:
            prompt = f"Task: {task_profile}"
        """
        lines = []

        if self.nl_query:
            lines.append(f"**Query:** {self.nl_query}")

        if self.task_type:
            lines.append(f"**Task Type:** {self.task_type.value}")

        if self.entity_table:
            lines.append(f"**Entity Table:** {self.entity_table}")

        if self.time_duration is not None:
            lines.append(f"**Time Duration:** {self.time_duration} days")

        if self.target_col:
            lines.append(f"**Target Column:** {self.target_col}")

        if self.entity_col:
            lines.append(f"**Entity Column:** {self.entity_col}")

        if self.sql_template:
            lines.append(f"**SQL Template:** {self.sql_template}")

        return '\n'.join(lines) if lines else "Empty TaskProfile"


@dataclass
class BasicParseResult(BaseResult):
    """Result from BasicTaskParser."""
    profile: Optional[TaskProfile] = None

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["profile"] = self.profile.to_dict() if self.profile else None
        return result


class BasicTaskParser(BaseOperator):
    """
    Part 1: Coarse-grained extraction.

    Extracts: task_type, entity_table, time_duration

    Example:
        >>> parser = BasicTaskParser()
        >>> result = parser(
        ...     llm_client=llm_client,
        ...     nl_query="Predict if user will click in next week",
        ...     db_schema=db_schema
        ... )
        >>> if result.success:
        ...     print(result.profile.task_type)
    """

    TASK_TYPE_MAP = {
        "BINARY_CLASSIFICATION": TaskType.BINARY_CLASSIFICATION,
        "REGRESSION": TaskType.REGRESSION,
    }

    def __init__(
        self,
        similarity_threshold: float = 0.6,
        include_examples: bool = True
    ):
        self.similarity_threshold = similarity_threshold
        self.include_examples = include_examples

    @override
    def __build_prompt__(
        self,
        nl_query: str,
        db_schema: DatabaseSchema,
        **kwargs
    ) -> str:
        return NL2TaskPrompt.generate_extraction_prompt(
            nl_query=nl_query,
            db_schema=db_schema,
            include_examples=self.include_examples
        )

    @override
    def __get_system_prompt__(self) -> str:
        return NL2TaskPrompt.get_system_prompt()

    @override
    def __validate__(
        self,
        extraction: Dict[str, Any],
        db_schema: DatabaseSchema,
        **kwargs
    ) -> tuple[List[str], List[str]]:
        errors = []
        warnings = []

        # Check if LLM explicitly returned null for any key fields
        # This indicates the LLM couldn't determine the value with confidence
        # Note: time_duration can be legitimately absent (not in query), but if LLM
        # returns explicit null, it means LLM tried but failed to extract it
        if "task_type" in extraction and extraction["task_type"] is None:
            errors.append("LLM returned null for task_type - task cannot be determined")
        if "entity_table" in extraction and extraction["entity_table"] is None:
            errors.append("LLM returned null for entity_table - entity cannot be determined")
        if "time_duration" in extraction and extraction["time_duration"] is None:
            # If time_duration is explicitly null, LLM couldn't extract time from query
            # This is different from time_duration not being present (which is OK)
            warnings.append("LLM returned null for time_duration - no time window could be determined from query")

        # Validate task_type
        task_type = extraction.get("task_type")
        if not task_type:
            errors.append("Missing required field: task_type")
        elif task_type not in self.TASK_TYPE_MAP:
            errors.append(f"Invalid task_type: {task_type}")

        # Validate entity_table
        entity_table = extraction.get("entity_table")
        if not entity_table:
            errors.append("Missing required field: entity_table")
        else:
            matcher = SchemaMatcher(db_schema, self.similarity_threshold)
            match_result = matcher.validate_and_match(extracted_table=entity_table)
            if not match_result.matched:
                errors.extend(match_result.warnings)
            else:
                # Reject low-confidence fuzzy matches as they may be incorrect
                if match_result.confidence < 0.8:
                    errors.append(
                        f"Low confidence match for entity_table '{entity_table}' -> '{match_result.entity_table}' "
                        f"(confidence: {match_result.confidence:.2f}). This may indicate an invalid table name."
                    )
                elif match_result.confidence < 1.0:
                    warnings.extend(match_result.warnings)
                extraction["_matched_table"] = match_result.entity_table

        # Validate time_duration type
        time_duration = extraction.get("time_duration")
        if not isinstance(time_duration, int):
            errors.append("time_duration must be integer")

        return errors, warnings

    @override
    def __build_result__(
        self,
        extraction: Dict[str, Any],
        warnings: List[str],
        nl_query: str,
        **kwargs
    ) -> BasicParseResult:
        profile = TaskProfile(
            nl_query=nl_query,
            task_type=self.TASK_TYPE_MAP[extraction["task_type"]],
            entity_table=extraction.get("_matched_table", extraction["entity_table"]),
            time_duration=extraction.get("time_duration")
        )

        return BasicParseResult(
            success=True,
            profile=profile,
            warnings=warnings,
            reasoning=extraction.get("reasoning"),
            raw_extraction=extraction
        )



class SQLQueryGenerator(BaseOperator):
    """
    Part 2: SQL generation for label computation.

    Takes TaskProfile from Part 1 and fills SQL-related fields.

    TODO: Implement in next iteration.
    """

    @override
    def __build_prompt__(self, **kwargs) -> str:
        raise NotImplementedError("Part 2 not yet implemented")

    @override
    def __validate__(
        self,
        extraction: Dict[str, Any],
        **kwargs
    ) -> tuple[List[str], List[str]]:
        raise NotImplementedError("Part 2 not yet implemented")

    @override
    def __build_result__(
        self,
        extraction: Dict[str, Any],
        warnings: List[str],
        **kwargs
    ) -> BaseResult:
        raise NotImplementedError("Part 2 not yet implemented")
