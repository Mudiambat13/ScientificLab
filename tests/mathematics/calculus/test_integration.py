import pytest
import sympy as sp

from scientificlab.mathematics.calculus import (
    definite_integral,
    differentiate,
    indefinite_integral,
)
from scientificlab.mathematics.numerical_methods import adaptive_integral
from scientificlab.utils.exceptions import InvalidInputError, SolverError

x = sp.Symbol("x")


def test_indefinite_integral():
    assert indefinite_integral("3*x**2") == x**3
    assert indefinite_integral("1/x") == sp.log(x)


@pytest.mark.parametrize("integrand", ["cos(x)*exp(x)", "x**2 + 3*x", "1/(1+x**2)"])
def test_fundamental_theorem_round_trip(integrand):
    antiderivative = indefinite_integral(integrand)
    assert sp.simplify(differentiate(antiderivative) - sp.sympify(integrand)) == 0


def test_definite_integrals_are_exact():
    assert definite_integral("sin(x)", "x", 0, "pi") == 2
    assert definite_integral("x**2", "x", 0, 1) == sp.Rational(1, 3)
    assert definite_integral("sin(x)**2", "x", 0, "pi") == sp.pi / 2


def test_improper_integral_with_infinite_bounds():
    assert definite_integral("exp(-x**2)", "x", "-oo", "oo") == sp.sqrt(sp.pi)
    assert definite_integral("exp(-x)", "x", 0, "oo") == 1


def test_float_bounds_give_floating_point_result():
    result = definite_integral("x", "x", 0.0, 1.0)
    assert float(result) == pytest.approx(0.5)


def test_equal_bounds_give_zero():
    assert definite_integral("x**2", "x", 2, 2) == 0


def test_reversed_bounds_negate():
    assert definite_integral("x", "x", 1, 0) == -definite_integral("x", "x", 0, 1)


def test_agrees_with_numerical_quadrature():
    exact = float(definite_integral("exp(-x)*sin(x)", "x", 0, 3))
    numeric = adaptive_integral(
        lambda v: __import__("math").exp(-v) * __import__("math").sin(v), 0, 3
    )
    assert numeric.value == pytest.approx(exact, rel=1e-9)


@pytest.mark.parametrize(
    ("expression", "lower", "upper"),
    [("1/x", 0, 1), ("1/x", 1, "oo"), ("1/x**2", -1, 1)],
)
def test_divergent_integrals_raise(expression, lower, upper):
    with pytest.raises(SolverError):
        definite_integral(expression, "x", lower, upper)


def test_no_closed_form_raises():
    with pytest.raises(SolverError):
        indefinite_integral("exp(x**x)")


def test_bounds_must_not_contain_the_variable():
    with pytest.raises(InvalidInputError):
        definite_integral("x", "x", 0, "x")


def test_boolean_bound_is_rejected():
    with pytest.raises(InvalidInputError):
        definite_integral("x", "x", True, 2)
