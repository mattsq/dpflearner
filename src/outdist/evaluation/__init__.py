"""Evaluation utilities for outdist models."""

from .evaluator import cross_validate, CVFoldResult

__all__ = ["cross_validate", "CVFoldResult"]