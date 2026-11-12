"""Eigenvalues and eigenvectors of real square matrices (NumPy/LAPACK)."""

import warnings
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike

from scientificlab.utils.exceptions import IllConditionedWarning
from scientificlab.utils.validation import as_square_matrix

_IMAG_TOLERANCE = 1e-12
_DEFECTIVE_CONDITION = 1e8


@dataclass(frozen=True)
class EigenDecomposition:
    """Eigen-decomposition ``A v_k = lambda_k v_k``.

    Attributes:
        values: Eigenvalues sorted by (real part, imaginary part); real
            (``float64``) when all are real, otherwise ``complex128``.
        vectors: Matrix whose **column** ``k`` is the unit-norm eigenvector of
            ``values[k]``. Eigenvectors are defined up to a (possibly complex)
            scalar factor, so their sign is arbitrary.
        symmetric: Whether the matrix was exactly symmetric (then eigenvalues
            are real and eigenvectors orthonormal).
    """

    values: npt.NDArray[np.float64] | npt.NDArray[np.complex128]
    vectors: npt.NDArray[np.float64] | npt.NDArray[np.complex128]
    symmetric: bool


def eigen_decomposition(matrix: ArrayLike) -> EigenDecomposition:
    """Compute eigenvalues and eigenvectors of a real square matrix.

    Exactly symmetric matrices use the symmetric solver (real eigenvalues,
    orthonormal eigenvectors, higher accuracy). Other matrices use the general
    solver, and eigenvalues may be complex (conjugate pairs).

    Limitation: for defective matrices (fewer independent eigenvectors than the
    dimension, e.g. ``[[1, 1], [0, 1]]``) the returned eigenvectors are nearly
    parallel; an :class:`IllConditionedWarning` is emitted when that is
    detected. Repeated eigenvalues of non-symmetric matrices are sensitive to
    rounding errors.

    Raises:
        InvalidInputError: If ``matrix`` is not a finite real square matrix.
    """
    square = as_square_matrix(matrix, "matrix")
    if np.array_equal(square, square.T):
        sym_values, sym_vectors = np.linalg.eigh(square)
        return EigenDecomposition(sym_values, sym_vectors, symmetric=True)

    values, vectors = np.linalg.eig(square)
    order = np.lexsort((values.imag, values.real))
    values = values[order]
    vectors = vectors[:, order]

    scale = np.maximum(1.0, np.abs(values))
    if np.all(np.abs(values.imag) <= _IMAG_TOLERANCE * scale):
        values = values.real.astype(np.float64)
        if np.all(np.abs(vectors.imag) <= _IMAG_TOLERANCE):
            vectors = vectors.real.astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        vector_condition = float(np.linalg.cond(vectors))
    if not vector_condition < _DEFECTIVE_CONDITION:
        warnings.warn(
            "Eigenvectors are nearly linearly dependent: the matrix may be "
            "defective, so the eigenvectors are unreliable.",
            IllConditionedWarning,
            stacklevel=2,
        )
    return EigenDecomposition(values, vectors, symmetric=False)


def eigenvalues(
    matrix: ArrayLike,
) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
    """Eigenvalues of a real square matrix, sorted (see ``eigen_decomposition``)."""
    return eigen_decomposition(matrix).values


def eigenvectors(
    matrix: ArrayLike,
) -> npt.NDArray[np.float64] | npt.NDArray[np.complex128]:
    """Unit-norm eigenvectors as columns (see ``eigen_decomposition``)."""
    return eigen_decomposition(matrix).vectors
