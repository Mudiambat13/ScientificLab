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
