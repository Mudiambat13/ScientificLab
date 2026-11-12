import math

import pytest
import sympy as sp

from scientificlab.mathematics.calculus import differentiate, partial_derivative
from scientificlab.mathematics.numerical_methods import numerical_derivative
from scientificlab.utils.exceptions import ExpressionParseError, InvalidInputError

x, y = sp.symbols("x y")


def test_polynomial_derivative_reference_example():
    assert differentiate("x**3 + 2*x**2 + 5", variable="x") == 3 * x**2 + 4 * x


def test_default_variable_is_x():
    assert differentiate("x**2") == 2 * x


def test_higher_order_and_order_zero():
    assert differentiate("x**4", order=2) == 12 * x**2
    assert differentiate("x**4", order=0) == x**4
    assert differentiate("x**4", order=5) == 0


def test_standard_functions():
    assert differentiate("sin(x)") == sp.cos(x)
    assert differentiate("exp(2*x)") == 2 * sp.exp(2 * x)
    assert differentiate("log(x)") == 1 / x


def test_other_variable_name_and_constants():
    t = sp.Symbol("t")
    assert differentiate("a*t**2", variable="t") == 2 * sp.Symbol("a") * t


def test_result_is_exact_not_float():
    assert differentiate("x/3") == sp.Rational(1, 3)


def test_agrees_with_numerical_derivative():
    exact = float(differentiate("x*exp(x)").subs(x, 1.3))
    approx = numerical_derivative(lambda v: v * math.exp(v), 1.3)
    assert approx == pytest.approx(exact, rel=1e-8)


def test_partial_derivatives():
    f = "x**2*y + y**3"
    assert partial_derivative(f, "x") == 2 * x * y
    assert partial_derivative(f, "y") == x**2 + 3 * y**2
    assert partial_derivative(f, ["x", "y"]) == 2 * x
    assert partial_derivative(f, ["x", "x"]) == 2 * y


def test_mixed_partials_commute_for_smooth_functions():
    f = "sin(x*y) + exp(x)*y**2"
    assert (
        sp.simplify(
            partial_derivative(f, ["x", "y"]) - partial_derivative(f, ["y", "x"])
        )
        == 0
    )


@pytest.mark.parametrize("order", [-1, 1.5, True, "2", None])
def test_invalid_order(order):
    with pytest.raises(InvalidInputError):
        differentiate("x**2", order=order)


@pytest.mark.parametrize("name", ["2x", "", "x y", 5])
def test_invalid_variable_name(name):
    with pytest.raises(ExpressionParseError):
        differentiate("x**2", variable=name)


def test_invalid_expression():
    with pytest.raises(ExpressionParseError):
        differentiate("x**")


def test_partial_derivative_needs_a_variable():
    with pytest.raises(InvalidInputError):
        partial_derivative("x*y", [])
