"""
Model evaluation functions for VaR forecast validation.

This package contains:
- Kupiec test: Unconditional coverage test for VaR validation
- Model comparison: Compare models on violation rates and accuracy
"""

from src.evaluation.kupiec import run_kupiec_test
from src.evaluation.summary import compare_models, rank_models_by_coverage

__all__ = [
    "run_kupiec_test",
    "compare_models",
    "rank_models_by_coverage",
]
