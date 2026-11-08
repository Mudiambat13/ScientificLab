import pytest
import sympy as sp

from scientificlab.mathematics.algebra import solve_system
from scientificlab.utils.exceptions import (
    ExpressionParseError,
    InfiniteSolutionsError,
    InvalidInputError,
    NoSolutionError,
)


def test_linear_2x2_system():
    result = solve_system(["x + y = 3", "x - y = 1"])
    assert result.variables == ("x", "y")
    assert result.solutions == ({"x": sp.Integer(2), "y": sp.Integer(1)},)


def test_exact_rational_solution():
    result = solve_system(["3*x + y = 1", "x + 2*y = 1"])
    solution = result.solutions[0]
    assert solution["x"] == sp.Rational(1, 5)
    assert solution["y"] == sp.Rational(2, 5)


def test_three_unknowns():
    result = solve_system(["x + y + z = 6", "2*y + 5*z = -4", "2*x + 5*y - z = 27"])
    assert result.solutions[0] == {
        "x": sp.Integer(5),
        "y": sp.Integer(3),
        "z": sp.Integer(-2),
    }


def test_nonlinear_equation_has_two_solutions():
    result = solve_system(["x**2 = 4"])
    values = sorted(int(s["x"]) for s in result.solutions)
    assert values == [-2, 2]


def test_bare_expression_means_equal_zero():
    result = solve_system(["x - 7"])
    assert result.solutions[0]["x"] == 7


def test_inconsistent_system():
    with pytest.raises(NoSolutionError):
        solve_system(["x + y = 1", "x + y = 2"])


def test_underdetermined_system():
    with pytest.raises(InfiniteSolutionsError):
        solve_system(["x + y = 1"])


def test_no_equations():
    with pytest.raises(InvalidInputError):
        solve_system([])


def test_constant_equations_without_variable():
    with pytest.raises(InvalidInputError):
        solve_system(["2 = 2"])


@pytest.mark.parametrize("bad", ["x == 1", "x = y = 2", "x +", ""])
def test_malformed_equation(bad):
    with pytest.raises(ExpressionParseError):
        solve_system([bad])
