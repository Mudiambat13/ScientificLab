"""Shared helpers: exceptions and input validation."""

from scientificlab.utils.exceptions import (
    ConvergenceError,
    ExpressionParseError,
    IllConditionedWarning,
    IncompatibleUnitsError,
    InfiniteSolutionsError,
    InvalidInputError,
    NoSolutionError,
    PhysicalInconsistencyError,
    ScientificLabError,
    SingularMatrixError,
    SolverError,
    UnitError,
    UnknownUnitError,
)

__all__ = [
    "ConvergenceError",
    "ExpressionParseError",
    "IllConditionedWarning",
    "IncompatibleUnitsError",
    "InfiniteSolutionsError",
    "InvalidInputError",
    "NoSolutionError",
    "PhysicalInconsistencyError",
    "ScientificLabError",
    "SingularMatrixError",
    "SolverError",
    "UnitError",
    "UnknownUnitError",
]
