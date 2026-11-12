import math

import pytest
import sympy as sp

from scientificlab.solvers import EquationSolver, solve_equation
from scientificlab.utils.exceptions import (
    ConvergenceError,
    ExpressionParseError,
    InfiniteSolutionsError,
    InvalidInputError,
    NoSolutionError,
    SolverError,
)


class TestAlgebraic:
    def test_quadratic_exact_and_complete(self):
        r = solve_equation("x**2 - 4 = 0")
        assert r.solutions == (-2, 2)
        assert (r.exact, r.complete, r.verified) == (True, True, True)
        assert r.method == "symbolic"
        assert r.variable == "x"

    def test_irrational_roots_stay_symbolic(self):
        r = solve_equation("x**2 = 2")
        assert r.solutions == (-sp.sqrt(2), sp.sqrt(2))
        assert r.values == pytest.approx((-math.sqrt(2), math.sqrt(2)))

    def test_cubic(self):
        r = solve_equation("x**3 - 6*x**2 + 11*x - 6 = 0")
        assert r.solutions == (1, 2, 3)

    def test_physics_equation_with_values(self):
        r = solve_equation("v = v0 + a*t", "t", values={"v": 20, "v0": 0, "a": 2})
        assert r.solutions == (10,)
        assert r.exact

    def test_symbolic_solution_in_other_symbols(self):
        r = solve_equation("v = v0 + a*t", "t")
        assert r.solutions == ((sp.Symbol("v") - sp.Symbol("v0")) / sp.Symbol("a"),)
        assert r.verified

    def test_float_inputs_make_result_inexact(self):
        r = solve_equation("x**2 - 2.0*x - 3.0 = 0")
        assert r.exact is False
        assert r.complete is True
        assert r.values == pytest.approx((-1.0, 3.0))
        assert any("approximate" in w for w in r.warnings)

    def test_float_values_dict(self):
        r = solve_equation("F = m*a", "a", values={"F": 10.0, "m": 4.0})
        assert r.values == (2.5,)
        assert r.exact is False

    def test_rational_equation_excludes_pole(self):
        r = solve_equation("(x**2 - 1)/(x - 1) = 0")
        assert r.solutions == (-1,)  # x = 1 is a removable pole, not a root

    def test_unknown_inferred_when_single_symbol(self):
        assert solve_equation("2*x + 6 = 0").solutions == (-3,)

    def test_complex_domain(self):
        r = solve_equation("x**2 + 1 = 0", domain="complex")
        assert set(r.solutions) == {sp.I, -sp.I}
        assert r.complete

    def test_real_domain_no_real_root(self):
        with pytest.raises(NoSolutionError, match="real"):
            solve_equation("x**2 + 1 = 0")


class TestHonesty:
    def test_transcendental_with_closed_form_is_flagged_incomplete(self):
        r = solve_equation("exp(x) = 2")
        assert r.solutions == (sp.log(2),)
        assert r.complete is False
        assert any("incomplete" in w for w in r.warnings)
        assert r.verified

    def test_periodic_solution_set_is_not_pretended_finite(self):
        with pytest.raises(InfiniteSolutionsError, match="infinite"):
            solve_equation("sin(x) = 0")

    def test_no_closed_form_requires_initial_guess(self):
        with pytest.raises(SolverError, match="initial_guess"):
            solve_equation("cos(x) = x")

    def test_numeric_with_initial_guess_is_marked_inexact(self):
        r = solve_equation("cos(x) = x", initial_guess=0.7)
        assert r.method == "numeric"
        assert (r.exact, r.complete) == (False, False)
        assert r.verified
        assert r.values[0] == pytest.approx(0.7390851332151607, rel=1e-12)
        assert any("one root" in w for w in r.warnings)

    def test_numeric_needs_all_other_symbols(self):
        with pytest.raises(InvalidInputError, match="missing"):
            solve_equation("cos(x) = k*x", "x", initial_guess=0.5)

    def test_numeric_convergence_failure_is_raised(self):
        with pytest.raises(ConvergenceError):
            solve_equation("exp(x) + 1 = 0", initial_guess=0.0)

    def test_identity_and_contradiction(self):
        with pytest.raises(InfiniteSolutionsError, match="identity"):
            solve_equation("2*x + 3 = 2*x + 3", "x")
        with pytest.raises(NoSolutionError, match="false"):
            solve_equation("x + 1 = x", "x")

    def test_solution_is_verified_by_substitution(self):
        r = solve_equation("x**5 - 3*x + 1 = 0", initial_guess=0.3)
        x = r.values[0]
        assert x**5 - 3 * x + 1 == pytest.approx(0.0, abs=1e-12)


class TestInputValidation:
    def test_ambiguous_unknown(self):
        with pytest.raises(InvalidInputError, match="specify `variable`"):
            solve_equation("a*x + b = 0")

    def test_unknown_given_as_value(self):
        with pytest.raises(InvalidInputError, match="both"):
            solve_equation("x + 1 = 0", "x", values={"x": 1})

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), "2", True])
    def test_bad_values(self, bad):
        with pytest.raises(InvalidInputError):
            solve_equation("x + y = 0", "x", values={"y": bad})

    def test_bad_variable_and_domain(self):
        with pytest.raises(InvalidInputError):
            solve_equation("x = 1", "1x")
        with pytest.raises(InvalidInputError):
            EquationSolver(domain="quaternion")

    @pytest.mark.parametrize("text", ["", "x == 1", "x = y = 2", "x +* 2 = 0"])
    def test_unparsable_equations(self, text):
        with pytest.raises(ExpressionParseError):
            solve_equation(text)


class TestSystems:
    def test_linear_system(self):
        r = EquationSolver().solve_system(["x + y = 3", "x - y = 1"])
        assert r.solutions == ({"x": 2, "y": 1},)
        assert (r.exact, r.complete, r.verified) == (True, True, True)
        assert r.variables == ("x", "y")

    def test_nonlinear_system_returns_all_solutions(self):
        r = EquationSolver().solve_system(["x**2 + y**2 = 25", "x - y = 1"])
        assert {tuple(s.values()) for s in r.solutions} == {(4, 3), (-3, -4)}
        assert r.verified

    def test_system_with_values_and_explicit_variables(self):
        r = EquationSolver().solve_system(
            ["F = m*a", "F = k*x"], ["a", "x"], values={"F": 6, "m": 2, "k": 3}
        )
        assert r.solutions == ({"a": 3, "x": 2},)

    def test_inconsistent_and_underdetermined(self):
        with pytest.raises(NoSolutionError):
            EquationSolver().solve_system(["x + y = 1", "x + y = 2"])
        with pytest.raises(InfiniteSolutionsError):
            EquationSolver().solve_system(["x + y = 1"])

    def test_real_domain_filters_complex_solutions(self):
        with pytest.raises(NoSolutionError, match="real"):
            EquationSolver().solve_system(["x**2 + y**2 = -1", "x = y"])

    def test_numeric_system(self):
        r = EquationSolver().solve_system(
            ["x*y = 2", "x + y = 3.5"], initial_guess={"x": 1.5, "y": 1.0}
        )
        assert r.method == "numeric"
        assert (r.exact, r.complete, r.verified) == (False, False, True)
        x, y = (float(r.solutions[0][k]) for k in "xy")
        assert x * y == pytest.approx(2.0)
        assert x + y == pytest.approx(3.5)

    def test_numeric_system_errors(self):
        solver = EquationSolver()
        with pytest.raises(InvalidInputError, match="lacks"):
            solver.solve_system(["x*y = 2", "x + y = 3"], initial_guess={"x": 1.0})
        with pytest.raises(ConvergenceError):
            solver.solve_system(
                ["x**2 + y**2 = -1", "x = y"], initial_guess={"x": 1.0, "y": 1.0}
            )

    def test_system_input_errors(self):
        solver = EquationSolver()
        with pytest.raises(InvalidInputError):
            solver.solve_system([])
        with pytest.raises(InvalidInputError, match="no unknown"):
            solver.solve_system(["1 = 1"])

    def test_non_polynomial_system_flagged(self):
        r = EquationSolver().solve_system(["exp(x) = y", "y = 2"])
        assert r.complete is False
        assert any("incomplete" in w for w in r.warnings)
