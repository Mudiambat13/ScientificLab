import numpy as np
import pytest

from scientificlab.mathematics.linear_algebra import (
    determinant,
    dot_product,
    matrix_add,
    matrix_multiply,
    vector_norm,
)
from scientificlab.utils.exceptions import InvalidInputError


def test_matrix_add():
    result = matrix_add([[1, 2], [3, 4]], [[10, 20], [30, 40]])
    assert np.array_equal(result, [[11, 22], [33, 44]])


def test_matrix_add_shape_mismatch():
    with pytest.raises(InvalidInputError):
        matrix_add([[1, 2]], [[1], [2]])


def test_matrix_multiply_reference():
    result = matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]])
    assert np.array_equal(result, [[19, 22], [43, 50]])


def test_matrix_vector_product():
    assert np.array_equal(matrix_multiply([[1, 2], [3, 4]], [1, 1]), [3, 7])


def test_rectangular_product_shape():
    a = np.ones((2, 3))
    b = np.ones((3, 4))
    assert matrix_multiply(a, b).shape == (2, 4)


def test_inner_dimension_mismatch():
    with pytest.raises(InvalidInputError):
        matrix_multiply([[1, 2, 3]], [[1, 2]])


def test_multiplication_is_not_commutative_in_general():
    a, b = [[1, 2], [3, 4]], [[0, 1], [1, 0]]
    assert not np.array_equal(matrix_multiply(a, b), matrix_multiply(b, a))


@pytest.mark.parametrize(
    "bad",
    [[[1, 2], [3]], [[1, float("nan")]], [[1, float("inf")]], [], [["a", "b"]], 5],
)
def test_invalid_matrices(bad):
    with pytest.raises(InvalidInputError):
        matrix_add(bad, [[1, 2]])


def test_determinant_reference_values():
    assert determinant([[1, 2], [3, 4]]) == pytest.approx(-2.0)
    assert determinant(np.eye(5)) == pytest.approx(1.0)
    assert determinant([[2, 0, 0], [0, 3, 0], [0, 0, 4]]) == pytest.approx(24.0)


def test_determinant_is_multiplicative():
    a = np.array([[2.0, 1.0], [1.0, 3.0]])
    b = np.array([[1.0, 4.0], [2.0, 5.0]])
    assert determinant(a @ b) == pytest.approx(determinant(a) * determinant(b))


def test_determinant_of_singular_matrix_is_numerically_zero():
    assert determinant([[1, 2], [2, 4]]) == pytest.approx(0.0, abs=1e-12)


def test_determinant_requires_square():
    with pytest.raises(InvalidInputError):
        determinant([[1, 2, 3], [4, 5, 6]])


def test_determinant_overflow_is_reported():
    with pytest.raises(InvalidInputError):
        determinant(np.eye(5) * 1e100)


def test_dot_product():
    assert dot_product([1, 2, 3], [4, 5, 6]) == 32.0
    assert dot_product([1, 0], [0, 1]) == 0.0


def test_dot_product_length_mismatch():
    with pytest.raises(InvalidInputError):
        dot_product([1, 2], [1, 2, 3])


def test_vector_norms():
    assert vector_norm([3, 4]) == pytest.approx(5.0)
    assert vector_norm([3, -4], order=1) == pytest.approx(7.0)
    assert vector_norm([3, -4], order=float("inf")) == pytest.approx(4.0)


def test_cauchy_schwarz_inequality():
    a, b = [1.0, -2.0, 3.0], [4.0, 0.5, -1.0]
    assert abs(dot_product(a, b)) <= vector_norm(a) * vector_norm(b)


@pytest.mark.parametrize("order", [0.5, 0, -1, float("nan")])
def test_invalid_norm_order(order):
    with pytest.raises(InvalidInputError):
        vector_norm([1, 2], order=order)


def test_empty_vector_is_rejected():
    with pytest.raises(InvalidInputError):
        vector_norm([])
