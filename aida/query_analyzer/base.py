"""
Base Classes for Query Analyzer Operators

Provides unified interface for LLM-interactive operators.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field

from aida.llm import LLMClient
from .utils import parse_json_response


@dataclass
class BaseResult:
    """
    Base result class for LLM operator responses.

    All operator results inherit from this to share common fields.
    """
    success: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    reasoning: Optional[str] = None
    raw_response: Optional[str] = None
    raw_extraction: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "errors": self.errors,
            "warnings": self.warnings,
            "reasoning": self.reasoning,
        }


class BaseOperator(ABC):
    """
    Abstract base class for LLM-interactive operators.

    Provides unified interface and shared logic for operators that
    interact with LLMs (e.g., BasicTaskParser, SQLQueryGenerator).

    Subclasses implement:
    - __build_prompt__: Generate the prompt for LLM
    - __validate__: Validate the extracted response
    - __build_result__: Build the final result object

    Subclasses may override:
    - __parse_response__: Custom response parsing (default: JSON)
    - __get_system_prompt__: Custom system prompt
    """

    def __call__(
        self,
        llm_client: LLMClient,
        **kwargs
    ) -> BaseResult:
        """
        Execute the operator.

        Args:
            llm_client: LLM client for generation
            **kwargs: Operator-specific arguments

        Returns:
            BaseResult subclass with operation results
        """
        # Step 1: Build prompt
        prompt = self.__build_prompt__(**kwargs)
        system_prompt = self.__get_system_prompt__()

        # Step 2: Call LLM
        try:
            response = llm_client.complete(prompt, system_prompt=system_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return self._create_error_result(f"LLM call failed: {str(e)}")

        # Step 3: Parse response
        extraction = self.__parse_response__(response_text)
        if extraction is None:
            return self._create_error_result(
                f"Failed to parse response: {response_text[:200]}...",
                raw_response=response_text
            )

        # Step 4: Validate extraction
        errors, warnings = self.__validate__(extraction, **kwargs)
        if errors:
            return self._create_error_result(
                errors,
                raw_extraction=extraction,
            )

        # Step 5: Build result
        return self.__build_result__(
            extraction=extraction,
            warnings=warnings,
            **kwargs
        )

    @abstractmethod
    def __build_prompt__(self, **kwargs) -> str:
        """Build the prompt for LLM. Must be implemented by subclass."""
        ...

    @abstractmethod
    def __validate__(
        self,
        extraction: Dict[str, Any],
        **kwargs
    ) -> tuple[List[str], List[str]]:
        """
        Validate extracted response.

        Returns:
            Tuple of (errors, warnings)
        """
        ...

    @abstractmethod
    def __build_result__(
        self,
        extraction: Dict[str, Any],
        warnings: List[str],
        **kwargs
    ) -> BaseResult:
        """Build the final result object. Must be implemented by subclass."""
        ...

    def __parse_response__(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse LLM response. Default: JSON parsing.
        Override for custom parsing logic.
        """
        return parse_json_response(response)

    def __get_system_prompt__(self) -> Optional[str]:
        """
        Get system prompt. Default: None.
        Override to provide custom system prompt.
        """
        return None

    def _create_error_result(
        self,
        errors: Union[str, List[str]],
        raw_response: Optional[str] = None,
        raw_extraction: Optional[Dict[str, Any]] = None,
        reasoning: Optional[str] = None
    ) -> BaseResult:
        """Helper to create error result."""
        if isinstance(errors, str):
            errors = [errors]
        return BaseResult(
            success=False,
            errors=errors,
            raw_response=raw_response,
            raw_extraction=raw_extraction,
            reasoning=reasoning
        )
