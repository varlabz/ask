"""
LLM Model Evaluation System

A comprehensive framework for testing and evaluating LLM models.
"""

from .eval import (
    EvalAnalyzer,
    EvalRunner,
    TestCase,
    TestIteration,
    TestResult,
    create_test_config,
)

__all__ = [
    "TestCase",
    "TestIteration",
    "TestResult",
    "EvalRunner",
    "EvalAnalyzer",
    "create_test_config",
]

__version__ = "0.1.0"
