import math

import pytest

from scientificlab.mathematics.numerical_methods import (
    bisection,
    newton_raphson,
    secant,
)
from scientificlab.utils.exceptions import ConvergenceError, InvalidInputError

SQRT2 = math.sqrt(2.0)
DOTTIE = 0.7390851332151607  # unique real root of cos(x) = x


def square_minus_two(x: float) -> float:
    return x * x - 2.0


class TestNewtonRaphson:
    def test_square_root_of_two(self):
        result = newton_raphson(square_minus_two, 1.0, lambda x: 2 * x)
        assert result.root == pytest.approx(SQRT2, abs=1e-14)
        assert result.residual < 1e-14
        assert result.iterations <= 8  # quadratic convergence

    def test_without_derivative_uses_finite_differences(self):
        result = newton_raphson(square_minus_two, 1.0)
        assert result.root == pytest.approx(SQRT2, abs=1e-9)

    def test_dottie_number_reference_value(self):
        result = newton_raphson(
            lambda x: math.cos(x) - x, 1.0, lambda x: -math.sin(x) - 1
        )
        assert result.root == pytest.approx(DOTTIE, abs=1e-14)

    def test_negative_root_selected_by_starting_point(self):
        result = newton_raphson(square_minus_two, -1.0, lambda x: 2 * x)
        assert result.root == pytest.approx(-SQRT2, abs=1e-14)

    def test_exact_root_at_start(self):
        result = newton_raphson(lambda x: x - 3.0, 3.0)
        assert result.root == 3.0
        assert result.iterations == 0

    def test_multiple_root_converges_only_linearly(self):
        simple = newton_raphson(lambda x: x * x - 1.0, 2.0, lambda x: 2 * x)
        double = newton_raphson(lambda x: (x - 1.0) ** 2, 2.0, lambda x: 2 * (x - 1.0))
        assert double.root == pytest.approx(1.0, abs=1e-6)
        assert double.iterations > 3 * simple.iterations

    def test_zero_derivative_is_reported(self):
        with pytest.raises(ConvergenceError, match="derivative"):
            newton_raphson(square_minus_two, 0.0, lambda x: 2 * x)

    def test_divergence_is_reported(self):
        # Newton's method diverges for arctan when |x0| > ~1.39.
        with pytest.raises(ConvergenceError):
            newton_raphson(math.atan, 2.0, lambda x: 1 / (1 + x * x))

    def test_max_iterations_exhausted(self):
        with pytest.raises(ConvergenceError, match="did not converge"):
            newton_raphson(square_minus_two, 100.0, lambda x: 2 * x, max_iter=3)

    def test_nan_function_value(self):
        with pytest.raises(ConvergenceError):
            newton_raphson(lambda x: float("nan"), 1.0, lambda x: 1.0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"tol": 0.0},
            {"tol": -1.0},
            {"max_iter": 0},
            {"max_iter": True},
            {"x0": float("inf")},
        ],
    )
    def test_invalid_arguments(self, kwargs):
        arguments = {"x0": 1.0} | kwargs
        with pytest.raises(InvalidInputError):
            newton_raphson(square_minus_two, **arguments)

    def test_non_callable_function(self):
        with pytest.raises(InvalidInputError):
            newton_raphson("x", 1.0)  # type: ignore[arg-type]


class TestBisection:
    def test_square_root_of_two(self):
        result = bisection(square_minus_two, 0.0, 2.0)
        assert result.root == pytest.approx(SQRT2, abs=1e-11)

    def test_dottie_number_reference_value(self):
        result = bisection(lambda x: math.cos(x) - x, 0.0, 1.0)
        assert result.root == pytest.approx(DOTTIE, abs=1e-11)

    def test_iteration_count_matches_theory(self):
        result = bisection(square_minus_two, 0.0, 2.0, tol=1e-3)
        assert result.iterations <= math.ceil(math.log2(2.0 / 1e-3)) + 1

    def test_reversed_interval_is_accepted(self):
        assert bisection(square_minus_two, 2.0, 0.0).root == pytest.approx(
            SQRT2, abs=1e-11
        )

    def test_root_at_endpoint(self):
        assert bisection(lambda x: x - 1.0, 1.0, 5.0).root == 1.0
        assert bisection(lambda x: x - 5.0, 1.0, 5.0).root == 5.0

    def test_no_sign_change_is_rejected(self):
        with pytest.raises(InvalidInputError, match="opposite signs"):
            bisection(square_minus_two, 2.0, 3.0)

    def test_pole_is_visible_in_residual(self):
        result = bisection(lambda x: 1.0 / x, -1.0, 2.0)
        assert result.residual > 1e6  # sign change comes from a pole, not a root

    def test_identical_endpoints(self):
        with pytest.raises(InvalidInputError):
            bisection(square_minus_two, 1.0, 1.0)

    def test_tolerance_below_machine_resolution_still_terminates(self):
        result = bisection(square_minus_two, 0.0, 2.0, tol=1e-30)
        assert result.root == pytest.approx(SQRT2, abs=1e-15)

    def test_max_iterations_exhausted(self):
        with pytest.raises(ConvergenceError):
            bisection(square_minus_two, 0.0, 2.0, tol=1e-12, max_iter=5)

    def test_invalid_tolerance(self):
        with pytest.raises(InvalidInputError):
            bisection(square_minus_two, 0.0, 2.0, tol=0.0)


class TestSecant:
    def test_square_root_of_two(self):
        result = secant(square_minus_two, 1.0, 2.0)
        assert result.root == pytest.approx(SQRT2, abs=1e-14)
        assert result.iterations <= 10

    def test_dottie_number_reference_value(self):
        result = secant(lambda x: math.cos(x) - x, 0.0, 1.0)
        assert result.root == pytest.approx(DOTTIE, abs=1e-14)

    def test_starting_point_already_a_root(self):
        result = secant(lambda x: x - 2.0, 2.0, 3.0)
        assert result.root == 2.0
        assert result.residual == 0.0

    def test_identical_starting_points(self):
        with pytest.raises(InvalidInputError):
            secant(square_minus_two, 1.0, 1.0)

    def test_flat_secant_is_reported(self):
        with pytest.raises(ConvergenceError, match="Flat secant"):
            secant(lambda x: 5.0, 0.0, 1.0)

    def test_max_iterations_exhausted(self):
        with pytest.raises(ConvergenceError):
            secant(square_minus_two, 100.0, 101.0, max_iter=2)
