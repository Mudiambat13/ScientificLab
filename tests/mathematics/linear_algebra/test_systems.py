import warnings

import numpy as np
import pytest

from scientificlab.mathematics.linear_algebra import (
    condition_number,
    inverse,
    solve_linear_system,
)
from scientificlab.utils.exceptions import (
    IllConditionedWarning,
    InvalidInputError,
    SingularMatrixError,
)


def hilbert(n: int) -> np.ndarray:
    index = np.arange(n)
    return 1.0 / (index[:, None] + index[None, :] + 1.0)


def test_inverse_reference_values():
    result = inverse([[4, 7], [2, 6]])
    assert result == pytest.approx(np.array([[0.6, -0.7], [-0.2, 0.4]]))


def test_inverse_times_matrix_is_identity():
    a = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 4.0]])
    assert a @ inverse(a) == pytest.approx(np.eye(3), abs=1e-12)


@pytest.mark.parametrize(
    "singular", [[[1, 2], [2, 4]], [[0, 0], [0, 0]], [[1, 1], [1, 1]]]
)
def test_singular_matrix_has_no_inverse(singular):
    with pytest.raises(SingularMatrixError):
        inverse(singular)


def test_inverse_requires_square_matrix():
    with pytest.raises(InvalidInputError):
        inverse([[1, 2, 3], [4, 5, 6]])


def test_solve_linear_system_reference():
    result = solve_linear_system([[2, 1], [1, 3]], [3, 5])
    assert result.solution == pytest.approx([0.8, 1.4])
    assert result.residual_norm < 1e-12


def test_solve_3x3_against_known_solution():
    a = np.array([[1.0, 1.0, 1.0], [0.0, 2.0, 5.0], [2.0, 5.0, -1.0]])
    expected = np.array([5.0, 3.0, -2.0])
    result = solve_linear_system(a, a @ expected)
    assert result.solution == pytest.approx(expected)


def test_solution_reports_condition_number():
    assert solve_linear_system(np.eye(3), [1, 2, 3]).condition_number == pytest.approx(
        1.0
    )


def test_singular_system_raises():
    with pytest.raises(SingularMatrixError):
        solve_linear_system([[1, 2], [2, 4]], [1, 2])


def test_rhs_length_mismatch():
    with pytest.raises(InvalidInputError):
        solve_linear_system([[1, 0], [0, 1]], [1, 2, 3])


def test_non_square_system_is_rejected():
    with pytest.raises(InvalidInputError):
        solve_linear_system([[1, 2, 3], [4, 5, 6]], [1, 2])


def test_condition_numbers():
    assert condition_number(np.eye(4)) == pytest.approx(1.0)
    assert condition_number([[1, 2], [2, 4]]) > 1e15
    assert condition_number(np.zeros((2, 2))) == float("inf")


def test_ill_conditioned_system_warns_but_returns_result():
    a = hilbert(10)  # condition number ~ 1.6e13
    expected = np.ones(10)
    with pytest.warns(IllConditionedWarning):
        result = solve_linear_system(a, a @ expected)
    assert 1e10 < result.condition_number < 1e16
    # Accuracy is genuinely degraded, as the warning says.
    assert np.max(np.abs(result.solution - expected)) > 1e-8


def test_well_conditioned_system_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        solve_linear_system([[4, 1], [1, 3]], [1, 2])


def test_numerically_singular_hilbert_matrix_is_rejected():
    with pytest.raises(SingularMatrixError):
        inverse(hilbert(16))
