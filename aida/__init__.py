"""
AIDA (AI-Driven Database Analysis) Package

Stateless operators for LLM-based database analysis and task definition.

Components:
- db: Database and task profile schemas
- prompt: LLM prompt generators
- query_analyzer: NL query analysis operators
- llm: LLM client adapters
"""

__version__ = "1.0.0"

from .query_analyzer import (
    # Base
    BaseOperator,
    BaseResult,
    # Parser
    BasicTaskParser,
    SQLQueryGenerator,
    BasicParseResult,
    TaskProfile,
)
from .llm import LLMClient, LLMClientFactory, LLMResponse

__all__ = [
    # Base
    'BaseOperator',
    'BaseResult',
    # Parser
    'BasicTaskParser',
    'SQLQueryGenerator',
    'BasicParseResult',
    'TaskProfile',
    # LLM
    'LLMClient',
    'LLMClientFactory',
    'LLMResponse',
]
