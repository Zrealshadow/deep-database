"""
Search Algorithm Module

This module contains various search algorithms for neural architecture search.
"""

from .evolutionary_algorithm import evolutionary_algorithm
from .evaluation import evaluate_architecture_with_proxy, create_evaluation_function

__all__ = [
    "evolutionary_algorithm",
    "evaluate_architecture_with_proxy",
    "create_evaluation_function",
]