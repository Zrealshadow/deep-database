"""
AIDA Query Analyzer Module

Operators for analyzing natural language queries.

Components:
- base: BaseOperator, BaseResult
- parser: BasicTaskParser, SQLQueryGenerator, TaskProfile
- schema_matcher: SchemaMatcher
- json_parser: parse_json_response utility
"""

from .base import BaseOperator, BaseResult
from .nl2task import (
    BasicTaskParser,
    SQLQueryGenerator,
    BasicParseResult,
    TaskProfile,
)
from .data_profile import (
    TableSelector,
    FeatureSelector,
    TableSelectionResult,
    FeatureSelectionResult,
)
from .schema_matcher import SchemaMatcher, MatchResult
from .utils import parse_json_response

__all__ = [
    # Base
    'BaseOperator',
    'BaseResult',
    # NL2Task Parser
    'BasicTaskParser',
    'SQLQueryGenerator',
    'BasicParseResult',
    'TaskProfile',
    # Data Profile
    'TableSelector',
    'FeatureSelector',
    'TableSelectionResult',
    'FeatureSelectionResult',
    # Schema
    'SchemaMatcher',
    'MatchResult',
    # Utils
    'parse_json_response',
]
