"""Internal helpers shared by the numerical-methods modules."""

import math
from collections.abc import Callable

from scientificlab.utils.exceptions import ConvergenceError, InvalidInputError

ScalarFunction = Callable[[float], float]
"""A real function of one real variable."""


def require_callable(function: object, name: str) -> None:
    """Raise ``InvalidInputError`` unless ``function`` is callable."""
    if not callable(function):
        raise InvalidInputError(f"{name} must be a callable taking one float.")


def evaluate(function: ScalarFunction, x: float) -> float:
    """Evaluate ``function(x)`` and require a finite real value.

    Raises:
        ConvergenceError: If the function raises an arithmetic error (division
            by zero, overflow), or returns NaN, infinity, or a value that is
            not a real number.
    """
    try:
        value = float(function(x))
    except (TypeError, ValueError, ArithmeticError) as error:
        raise ConvergenceError(
            f"The function did not return a real number at x = {x!r}."
        ) from error
    if not math.isfinite(value):
        raise ConvergenceError(
            f"The function is not finite at x = {x!r} (got {value})."
        )
    return value


def require_max_iterations(max_iter: int) -> int:
    """Return ``max_iter`` after requiring a positive integer."""
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise InvalidInputError(
            f"max_iter must be a positive integer, got {max_iter!r}."
        )
    return max_iter
