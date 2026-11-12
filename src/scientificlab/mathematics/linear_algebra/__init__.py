"""Linear algebra: vector/matrix operations, inverses, linear systems, eigenproblems.

Floating-point (NumPy/LAPACK) computations with explicit validation and
conditioning checks: singular matrices raise ``SingularMatrixError`` and
ill-conditioned problems emit ``IllConditionedWarning``.
"""

from scientificlab.mathematics.linear_algebra.eigen import (
    EigenDecomposition,
    eigen_decomposition,
    eigenvalues,
    eigenvectors,
)
from scientificlab.mathematics.linear_algebra.operations import (
    determinant,
    dot_product,
    matrix_add,
    matrix_multiply,
    vector_norm,
)
from scientificlab.mathematics.linear_algebra.systems import (
    LinearSystemSolution,
    condition_number,
    inverse,
    solve_linear_system,
)

__all__ = [
    "EigenDecomposition",
    "LinearSystemSolution",
    "condition_number",
    "determinant",
    "dot_product",
    "eigen_decomposition",
    "eigenvalues",
    "eigenvectors",
    "inverse",
    "matrix_add",
    "matrix_multiply",
    "solve_linear_system",
    "vector_norm",
]
