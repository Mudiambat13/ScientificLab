import pytest
import sympy as sp

from scientificlab.mathematics.algebra import (
    expand_expression,
    factor_expression,
    parse_expression,
    simplify_expression,
)
from scientificlab.utils.exceptions import ExpressionParseError

x = sp.Symbol("x")


def test_expand_binomial():
    assert expand_expression("(x + 1)**2") == x**2 + 2 * x + 1


def test_caret_is_power():
    assert parse_expression("x^2") == x**2


def test_factor_difference_of_squares():
    assert factor_expression("x**2 - 1") == (x - 1) * (x + 1)


def test_factor_then_expand_round_trip():
    original = "x**3 - 6*x**2 + 11*x - 6"
    assert expand_expression(factor_expression(original)) == parse_expression(original)


def test_simplify_trig_identity():
    assert simplify_expression("sin(x)**2 + cos(x)**2") == 1


def test_simplify_keeps_exact_rationals():
    assert simplify_expression("(x**2 - 1)/(x - 1)") == x + 1


def test_accepts_sympy_expression():
    assert expand_expression((x + 2) ** 2) == x**2 + 4 * x + 4


@pytest.mark.parametrize("bad", ["", "   ", "x +", "2 * (x", "x = 1", 42, None])
def test_invalid_expressions_raise_parse_error(bad):
    with pytest.raises(ExpressionParseError):
        parse_expression(bad)
