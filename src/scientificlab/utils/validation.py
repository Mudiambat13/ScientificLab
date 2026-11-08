"""Input validation helpers (validation is kept separate from computation)."""

import math
from numbers import Real

from scientificlab.utils.exceptions import InvalidInputError


def require_finite(value: float, name: str) -> float:
    """Return ``value`` as a float, requiring a finite real number.

    Args:
        value: The value to check.
        name: Argument name used in the error message.

    Raises:
        InvalidInputError: If ``value`` is not a real number, or is NaN/infinite.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InvalidInputError(f"{name} must be a real number, got {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise InvalidInputError(f"{name} must be finite, got {value!r}.")
    return result


def require_positive(value: float, name: str) -> float:
    """Return ``value`` as a float, requiring ``value > 0``."""
    result = require_finite(value, name)
    if result <= 0.0:
        raise InvalidInputError(f"{name} must be strictly positive, got {value!r}.")
    return result


def require_non_negative(value: float, name: str) -> float:
    """Return ``value`` as a float, requiring ``value >= 0``."""
    result = require_finite(value, name)
    if result < 0.0:
        raise InvalidInputError(f"{name} must be non-negative, got {value!r}.")
    return result
