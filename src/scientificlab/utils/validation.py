"""Input validation helpers (validation is kept separate from computation)."""

import math
from collections.abc import Mapping
from numbers import Real

import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike

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


def require_n_given(values: Mapping[str, float | None], n: int) -> None:
    """Require that exactly ``n`` of the named quantities are provided.

    Used by relations such as ``V = I*R`` where any one quantity is computed from
    the others: the caller leaves the unknown(s) as ``None``.

    Raises:
        InvalidInputError: If the number of non-``None`` values is not ``n``.
    """
    given = [name for name, value in values.items() if value is not None]
    if len(given) != n:
        names = ", ".join(values)
        raise InvalidInputError(
            f"Provide exactly {n} of ({names}); got {len(given)}"
            + (f": {', '.join(given)}." if given else ".")
        )


def as_array(values: ArrayLike, name: str, *, ndim: int) -> npt.NDArray[np.float64]:
    """Convert ``values`` to a finite real float array with exactly ``ndim`` axes.

    Raises:
        InvalidInputError: If the data is not numeric, is ragged, has the wrong
            number of dimensions, is empty, or contains NaN/infinity.
    """
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise InvalidInputError(f"{name} must be a real numeric array.") from error
    if array.ndim != ndim:
        raise InvalidInputError(
            f"{name} must have {ndim} dimension(s), got shape {array.shape}."
        )
    if array.size == 0:
        raise InvalidInputError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise InvalidInputError(f"{name} must contain only finite values.")
    return array


def as_vector(values: ArrayLike, name: str) -> npt.NDArray[np.float64]:
    """Convert ``values`` to a finite 1-D float array."""
    return as_array(values, name, ndim=1)


def as_matrix(values: ArrayLike, name: str) -> npt.NDArray[np.float64]:
    """Convert ``values`` to a finite 2-D float array."""
    return as_array(values, name, ndim=2)


def as_square_matrix(values: ArrayLike, name: str) -> npt.NDArray[np.float64]:
    """Convert ``values`` to a finite square 2-D float array."""
    matrix = as_matrix(values, name)
    if matrix.shape[0] != matrix.shape[1]:
        raise InvalidInputError(f"{name} must be square, got shape {matrix.shape}.")
    return matrix
