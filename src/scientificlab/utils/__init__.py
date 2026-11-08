"""Shared helpers: exceptions and input validation."""

from scientificlab.utils.exceptions import (
    ExpressionParseError,
    InfiniteSolutionsError,
    InvalidInputError,
    NoSolutionError,
    PhysicalInconsistencyError,
    ScientificLabError,
)

__all__ = [
    "ExpressionParseError",
    "InfiniteSolutionsError",
    "InvalidInputError",
    "NoSolutionError",
    "PhysicalInconsistencyError",
    "ScientificLabError",
]
