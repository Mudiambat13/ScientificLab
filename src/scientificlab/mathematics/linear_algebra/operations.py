"""Basic vector and matrix operations (NumPy backend, float64 arithmetic).

All inputs are array-likes (nested lists, tuples, ndarrays) of finite real
numbers and are validated; results are new ``numpy.ndarray`` objects. These
are floating-point computations: results carry rounding errors of the order of
machine epsilon (about 2.2e-16, relative) times a problem-dependent factor.
"""

import math

import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike

from scientificlab.utils.exceptions import InvalidInputError
from scientificlab.utils.validation import (
    as_matrix,
    as_square_matrix,
    as_vector,
    require_finite,
)


def matrix_add(a: ArrayLike, b: ArrayLike) -> npt.NDArray[np.float64]:
    """Element-wise sum of two matrices of identical shape.

    Raises:
        InvalidInputError: If an input is not a finite real matrix, or the
            shapes differ.
    """
    left = as_matrix(a, "a")
    right = as_matrix(b, "b")
    if left.shape != right.shape:
        raise InvalidInputError(
            f"Cannot add matrices of shapes {left.shape} and {right.shape}."
        )
    return left + right


def matrix_multiply(a: ArrayLike, b: ArrayLike) -> npt.NDArray[np.float64]:
    """Matrix product ``a @ b``.

    ``b`` may be a matrix (n x p) or a vector (length n); ``a`` is m x n. The
    result is m x p, or length m when ``b`` is a vector.

    Raises:
        InvalidInputError: If inputs are invalid or the inner dimensions differ.
    """
    left = as_matrix(a, "a")
    right_raw = np.asarray(b)
    right = as_vector(b, "b") if right_raw.ndim == 1 else as_matrix(b, "b")
    if left.shape[1] != right.shape[0]:
        raise InvalidInputError(
            f"Cannot multiply shapes {left.shape} and {right.shape}: "
            "inner dimensions differ."
        )
    return left @ right


def determinant(matrix: ArrayLike) -> float:
    """Determinant of a square matrix (via LU factorisation).

    Because of rounding, the determinant of a singular floating-point matrix is
    usually ~1e-16 rather than exactly 0. Do not test singularity with
    ``det == 0``; use the condition number (see ``inverse`` /
    ``solve_linear_system``). Very large matrices can overflow.

    Raises:
        InvalidInputError: If ``matrix`` is not a finite square matrix, or the
            determinant overflows the float range.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        value = float(np.linalg.det(as_square_matrix(matrix, "matrix")))
    if not math.isfinite(value):
        raise InvalidInputError("The determinant overflowed the floating-point range.")
    return value


def dot_product(a: ArrayLike, b: ArrayLike) -> float:
    """Scalar (dot) product of two vectors of equal length.

    Raises:
        InvalidInputError: If a vector is invalid or the lengths differ.
    """
    left = as_vector(a, "a")
    right = as_vector(b, "b")
    if left.shape != right.shape:
        raise InvalidInputError(
            f"Vectors must have equal length, got {left.shape[0]} and {right.shape[0]}."
        )
    return float(np.dot(left, right))


def vector_norm(vector: ArrayLike, order: float = 2.0) -> float:
    """p-norm of a vector: ``(sum(|x_i|**p))**(1/p)``.

    Args:
        vector: Finite real vector.
        order: ``p >= 1``; use ``float("inf")`` for the maximum norm. The
            default 2 is the Euclidean norm.

    Raises:
        InvalidInputError: If ``order`` is NaN or less than 1.
    """
    values = as_vector(vector, "vector")
    if math.isnan(order) or order < 1:
        raise InvalidInputError(f"order must be >= 1 (or inf), got {order!r}.")
    if not math.isinf(order):
        order = require_finite(order, "order")
    return float(np.linalg.norm(values, ord=order))
