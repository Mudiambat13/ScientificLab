"""Explicit exception hierarchy used across ScientificLab."""


class ScientificLabError(Exception):
    """Base class for all ScientificLab errors."""


class InvalidInputError(ScientificLabError, ValueError):
    """An argument is not a valid number or lies outside its valid domain."""


class NoSolutionError(ScientificLabError):
    """The problem has no solution."""


class InfiniteSolutionsError(ScientificLabError):
    """The problem has infinitely many solutions (underdetermined)."""


class ExpressionParseError(ScientificLabError, ValueError):
    """A textual expression or equation could not be parsed."""


class PhysicalInconsistencyError(ScientificLabError, ValueError):
    """The inputs describe a physically impossible situation."""


class SolverError(ScientificLabError):
    """A symbolic engine could not produce a closed-form or verified result."""


class ConvergenceError(ScientificLabError):
    """An iterative algorithm failed to converge or hit a degenerate case."""


class SingularMatrixError(ScientificLabError):
    """A matrix is singular or numerically singular (cannot be inverted/solved)."""


class UnitError(ScientificLabError, ValueError):
    """A unit is unknown, or a quantity has the wrong units."""


class IncompatibleUnitsError(UnitError):
    """Units have different physical dimensions (e.g. adding metres to seconds)."""


class UnknownUnitError(UnitError):
    """A unit string is not recognised."""


class IllConditionedWarning(UserWarning):
    """The result is returned but may be inaccurate (large condition number)."""
