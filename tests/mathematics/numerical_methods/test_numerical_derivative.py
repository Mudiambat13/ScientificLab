import math

import pytest

from scientificlab.mathematics.numerical_methods import numerical_derivative
from scientificlab.utils.exceptions import ConvergenceError, InvalidInputError


def test_central_difference_reference_values():
    assert numerical_derivative(lambda x: x * x, 3.0) == pytest.approx(6.0, rel=1e-8)
    assert numerical_derivative(math.sin, 0.0) == pytest.approx(1.0, rel=1e-9)
    assert numerical_derivative(math.exp, 1.0) == pytest.approx(math.e, rel=1e-9)


@pytest.mark.parametrize("scheme", ["forward", "backward"])
def test_one_sided_schemes(scheme):
    result = numerical_derivative(lambda x: x**3, 2.0, scheme=scheme)
    assert result == pytest.approx(12.0, rel=1e-6)


def test_central_is_more_accurate_than_one_sided_with_same_step():
    h = 1e-3
    central = abs(numerical_derivative(math.exp, 1.0, step=h) - math.e)
    forward = abs(
        numerical_derivative(math.exp, 1.0, step=h, scheme="forward") - math.e
    )
    assert central < forward / 100


def test_truncation_error_order_two_for_central_difference():
    def error(h: float) -> float:
        return abs(numerical_derivative(math.sin, 1.0, step=h) - math.cos(1.0))

    assert error(1e-2) / error(5e-3) == pytest.approx(4.0, rel=0.05)


def test_rounding_error_dominates_for_tiny_steps():
    good = abs(numerical_derivative(math.exp, 1.0) - math.e)
    tiny = abs(numerical_derivative(math.exp, 1.0, step=1e-13) - math.e)
    assert tiny > 100 * good


def test_kink_shows_scheme_dependence():
    # |x| is not differentiable at 0; the schemes disagree, signalling the problem.
    assert numerical_derivative(abs, 0.0, scheme="forward") == pytest.approx(1.0)
    assert numerical_derivative(abs, 0.0, scheme="backward") == pytest.approx(-1.0)
    assert numerical_derivative(abs, 0.0, scheme="central") == pytest.approx(0.0)


def test_large_argument_uses_scaled_step():
    assert numerical_derivative(lambda x: x * x, 1e6) == pytest.approx(2e6, rel=1e-6)


def test_invalid_scheme():
    with pytest.raises(InvalidInputError):
        numerical_derivative(math.sin, 0.0, scheme="spectral")


@pytest.mark.parametrize("step", [0.0, -1e-3, float("nan")])
def test_invalid_step(step):
    with pytest.raises(InvalidInputError):
        numerical_derivative(math.sin, 0.0, step=step)


def test_non_callable():
    with pytest.raises(InvalidInputError):
        numerical_derivative("sin", 0.0)  # type: ignore[arg-type]


def test_function_not_finite_near_point():
    with pytest.raises(ConvergenceError):
        numerical_derivative(lambda x: 1.0 / x, 0.0, step=0.0 + 1e-3, scheme="forward")
