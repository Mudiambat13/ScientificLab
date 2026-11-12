"""Matrix inversion and linear systems, with explicit conditioning checks.

Conditioning (2-norm condition number ``kappa``) controls accuracy: roughly
``log10(kappa)`` significant digits of the result are lost. Behaviour here:

* ``kappa >= 1/eps`` (about 4.5e15): numerically singular, so
  :class:`SingularMatrixError` is raised and no result is returned.
* ``kappa > 1e10``: the result is returned together with an
  :class:`IllConditionedWarning`.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike

from scientificlab.utils.exceptions import (
    IllConditionedWarning,
    InvalidInputError,
    SingularMatrixError,
)
from scientificlab.utils.validation import as_square_matrix, as_vector

SINGULAR_CONDITION_NUMBER: float = 1.0 / float(np.finfo(np.float64).eps)
"""Condition number at and above which a matrix is treated as singular."""

WARNING_CONDITION_NUMBER: float = 1e10
"""Condition number above which an :class:`IllConditionedWarning` is emitted."""


@dataclass(frozen=True)
class LinearSystemSolution:
    """Solution of ``A x = b``.

    Attributes:
        solution: The vector ``x``.
        residual_norm: Euclidean norm of ``A x - b`` (a posteriori check).
        condition_number: 2-norm condition number of ``A``.
    """

    solution: npt.NDArray[np.float64]
    residual_norm: float
    condition_number: float


def condition_number(matrix: ArrayLike) -> float:
    """2-norm condition number of a square matrix (``inf`` if singular)."""
    square = as_square_matrix(matrix, "matrix")
    with np.errstate(divide="ignore", invalid="ignore"):
        value = float(np.linalg.cond(square))
    return float("inf") if np.isnan(value) else value


def _checked_condition(square: npt.NDArray[np.float64]) -> float:
    """Return the condition number; reject singular, warn on ill-conditioned."""
    kappa = condition_number(square)
    if kappa >= SINGULAR_CONDITION_NUMBER:
        raise SingularMatrixError(
            "The matrix is singular or numerically singular "
            f"(condition number {kappa:.3g})."
        )
    if kappa > WARNING_CONDITION_NUMBER:
        warnings.warn(
            f"Ill-conditioned matrix (condition number {kappa:.3g}): roughly "
            f"{np.log10(kappa):.0f} significant digits may be lost.",
            IllConditionedWarning,
            stacklevel=3,
        )
    return kappa


def inverse(matrix: ArrayLike) -> npt.NDArray[np.float64]:
    """Inverse of a square matrix.

    Raises:
        InvalidInputError: If ``matrix`` is not a finite square matrix.
        SingularMatrixError: If the matrix is (numerically) singular.
    """
    square = as_square_matrix(matrix, "matrix")
    _checked_condition(square)
    try:
        return np.linalg.inv(square)
    except np.linalg.LinAlgError as error:
        raise SingularMatrixError("The matrix is singular.") from error


def solve_linear_system(a: ArrayLike, b: ArrayLike) -> LinearSystemSolution:
    """Solve the square system ``A x = b`` by LU factorisation with pivoting.

    Only square, non-singular systems are handled; over- and under-determined
    systems (least squares) are not provided.

    Args:
        a: Square coefficient matrix ``A`` (n x n).
        b: Right-hand side vector (length n).

    Returns:
        A :class:`LinearSystemSolution` with the solution, residual norm and
        condition number of ``A``.

    Raises:
        InvalidInputError: If inputs are invalid or shapes are incompatible.
        SingularMatrixError: If ``A`` is (numerically) singular.
    """
    matrix = as_square_matrix(a, "a")
    rhs = as_vector(b, "b")
    if rhs.shape[0] != matrix.shape[0]:
        raise InvalidInputError(
            f"b must have length {matrix.shape[0]} to match a, got {rhs.shape[0]}."
        )
    kappa = _checked_condition(matrix)
    try:
        x = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as error:
        raise SingularMatrixError("The matrix is singular.") from error
    residual = float(np.linalg.norm(matrix @ x - rhs))
    return LinearSystemSolution(
        solution=x, residual_norm=residual, condition_number=kappa
    )
