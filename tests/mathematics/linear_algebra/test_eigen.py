import numpy as np
import pytest

from scientificlab.mathematics.linear_algebra import (
    eigen_decomposition,
    eigenvalues,
    eigenvectors,
)
from scientificlab.utils.exceptions import IllConditionedWarning, InvalidInputError


def assert_eigenpairs(matrix, result, tol=1e-12):
    a = np.asarray(matrix, dtype=float)
    for k in range(result.values.size):
        vector = result.vectors[:, k]
        assert np.linalg.norm(vector) == pytest.approx(1.0)
        assert np.linalg.norm(a @ vector - result.values[k] * vector) < tol


def test_symmetric_matrix_reference():
    result = eigen_decomposition([[2, 1], [1, 2]])
    assert result.symmetric
    assert result.values == pytest.approx([1.0, 3.0])
    assert_eigenpairs([[2, 1], [1, 2]], result)
    assert result.vectors.T @ result.vectors == pytest.approx(np.eye(2), abs=1e-12)


def test_diagonal_matrix_values_are_sorted():
    assert eigenvalues([[3, 0], [0, 2]]) == pytest.approx([2.0, 3.0])


def test_non_symmetric_real_eigenvalues():
    a = [[2, 1], [0, 3]]
    result = eigen_decomposition(a)
    assert not result.symmetric
    assert result.values.dtype == np.float64
    assert result.values == pytest.approx([2.0, 3.0])
    assert_eigenpairs(a, result)


def test_rotation_matrix_has_complex_eigenvalues():
    a = [[0, -1], [1, 0]]
    result = eigen_decomposition(a)
    assert np.iscomplexobj(result.values)
    assert result.values == pytest.approx([-1j, 1j])
    assert_eigenpairs(a, result)


def test_trace_and_determinant_identities():
    a = np.array([[4.0, 1.0, 2.0], [0.5, 3.0, 1.0], [2.0, 0.0, 1.0]])
    values = eigenvalues(a)
    assert np.sum(values).real == pytest.approx(np.trace(a))
    assert np.prod(values).real == pytest.approx(np.linalg.det(a))


def test_eigenvectors_function_returns_columns():
    vectors = eigenvectors([[2, 0], [0, 3]])
    assert vectors.shape == (2, 2)
    assert abs(vectors[0, 0]) == pytest.approx(1.0)


def test_defective_matrix_warns():
    with pytest.warns(IllConditionedWarning):
        result = eigen_decomposition([[1, 1], [0, 1]])
    assert result.values == pytest.approx([1.0, 1.0])


def test_non_square_is_rejected():
    with pytest.raises(InvalidInputError):
        eigenvalues([[1, 2, 3], [4, 5, 6]])


def test_non_finite_is_rejected():
    with pytest.raises(InvalidInputError):
        eigenvalues([[1, float("nan")], [0, 1]])
