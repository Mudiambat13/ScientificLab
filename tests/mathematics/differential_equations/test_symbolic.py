import pytest
import sympy as sp

from scientificlab.mathematics.differential_equations import (
    SymbolicODESolution,
    solve_ode,
)
from scientificlab.utils.exceptions import (
    ExpressionParseError,
    InvalidInputError,
    SolverError,
)

x, t, k = sp.symbols("x t k")
C1, C2 = sp.symbols("C1 C2")


def same(a, b) -> bool:
    return sp.simplify(a - b) == 0


def test_exponential_growth_with_initial_condition():
    result = solve_ode("y' = y", initial_conditions={"y(0)": 1})
    assert result.order == 1
    assert same(result.solution, sp.exp(x))
    assert result.verified


def test_first_order_general_solution_keeps_constant():
    result = solve_ode("y' + 2*y = 0")
    assert same(result.solution, C1 * sp.exp(-2 * x))
    assert result.verified


def test_integrable_right_hand_side():
    result = solve_ode("y' = 2*x")
    assert same(result.solution, x**2 + C1)
    assert result.verified


def test_bare_expression_means_equals_zero():
    assert same(solve_ode("y' - y").solution, C1 * sp.exp(x))


def test_harmonic_oscillator_with_initial_conditions():
    result = solve_ode("y'' + y = 0", initial_conditions={"y(0)": 0, "y'(0)": 1})
    assert result.order == 2
    assert same(result.solution, sp.sin(x))
    assert result.verified


def test_second_order_with_distinct_real_roots():
    result = solve_ode(
        "y'' - 3*y' + 2*y = 0", initial_conditions={"y(0)": 1, "y'(0)": 0}
    )
    assert same(result.solution, 2 * sp.exp(x) - sp.exp(2 * x))
    assert result.verified


def test_general_second_order_solution_has_two_constants():
    result = solve_ode("y'' = 0")
    assert same(result.solution, C1 + C2 * x)


def test_custom_function_and_variable_names():
    result = solve_ode(
        "u'' + 4*u = 0",
        function="u",
        variable="t",
        initial_conditions={"u(0)": 1, "u'(0)": 0},
    )
    assert same(result.solution, sp.cos(2 * t))
    assert result.verified


def test_newtons_law_of_cooling_with_symbolic_parameter():
    result = solve_ode("T' = -k*(T - 20)", function="T", variable="t")
    assert same(result.solution, 20 + C1 * sp.exp(-k * t))
    assert result.verified


def test_initial_condition_at_nonzero_point_and_rational_value():
    result = solve_ode("y' = y", initial_conditions={"y(1)": "1/2"})
    assert same(result.solution, sp.Rational(1, 2) * sp.exp(x - 1))
    assert result.verified


def test_solution_is_exact_symbolic_expression():
    result = solve_ode("y' = y", initial_conditions={"y(0)": 1})
    assert result.solution == sp.exp(x)


def test_solution_property_rejects_ambiguous_results():
    branches = SymbolicODESolution("e", "y", "x", 1, (x, -x), verified=True)
    assert len(branches.solutions) == 2
    with pytest.raises(InvalidInputError):
        _ = branches.solution


@pytest.mark.parametrize("bad", ["", "   ", "y' +", "y' == y", "y' = y = 2", 5])
def test_unparsable_equations(bad):
    with pytest.raises(ExpressionParseError):
        solve_ode(bad)


def test_equation_without_derivative_is_rejected():
    with pytest.raises(InvalidInputError):
        solve_ode("y = 3")


def test_third_order_is_rejected():
    with pytest.raises(InvalidInputError):
        solve_ode("y''' = y")


@pytest.mark.parametrize(
    "conditions",
    [{"z(0)": 1}, {"y'(0)": 1}, {"y": 1}, {"y(0)": 1, "y(0) ": 2}],
)
def test_malformed_or_excessive_initial_conditions(conditions):
    with pytest.raises(InvalidInputError):
        solve_ode("y' = y", initial_conditions=conditions)


def test_function_and_variable_must_differ():
    with pytest.raises(InvalidInputError):
        solve_ode("x' = x", function="x", variable="x")


def test_invalid_names():
    with pytest.raises(ExpressionParseError):
        solve_ode("y' = y", function="2y")


def test_unsolvable_equation_raises_solver_error(monkeypatch):
    # A genuinely unsolvable ODE makes SymPy search for ~1 minute, so the
    # engine's "not implemented" failure is simulated to keep the suite fast.
    def not_implemented(*args, **kwargs):
        raise NotImplementedError("no method found")

    monkeypatch.setattr(sp, "dsolve", not_implemented)
    with pytest.raises(SolverError, match="Cannot solve"):
        solve_ode("y' = sin(x*y) + exp(x*y**2)")
