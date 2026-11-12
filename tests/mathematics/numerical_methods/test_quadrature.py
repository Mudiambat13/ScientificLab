import math

import pytest

from scientificlab.mathematics.numerical_methods import (
    adaptive_integral,
    simpson_rule,
    trapezoid_rule,
)
from scientificlab.utils.exceptions import ConvergenceError, InvalidInputError


class TestTrapezoid:
    def test_exact_for_linear_functions(self):
        assert trapezoid_rule(lambda x: 2 * x + 1, 0.0, 4.0, n=3) == pytest.approx(20.0)

    def test_sine_over_half_period(self):
        assert trapezoid_rule(math.sin, 0.0, math.pi, n=1000) == pytest.approx(
            2.0, abs=5e-6
        )

    def test_error_decreases_by_four_when_n_doubles(self):
        e1 = abs(trapezoid_rule(math.sin, 0.0, math.pi, n=50) - 2.0)
        e2 = abs(trapezoid_rule(math.sin, 0.0, math.pi, n=100) - 2.0)
        assert e1 / e2 == pytest.approx(4.0, rel=0.05)

    def test_reversed_bounds_negate(self):
        forward = trapezoid_rule(math.exp, 0.0, 1.0, n=200)
        assert trapezoid_rule(math.exp, 1.0, 0.0, n=200) == pytest.approx(-forward)

    def test_empty_interval(self):
        assert trapezoid_rule(math.exp, 2.0, 2.0) == 0.0

    @pytest.mark.parametrize("n", [0, -3, 2.5, True])
    def test_invalid_n(self, n):
        with pytest.raises(InvalidInputError):
            trapezoid_rule(math.sin, 0.0, 1.0, n=n)

    def test_non_finite_function_value(self):
        with pytest.raises(ConvergenceError):
            trapezoid_rule(lambda x: 1.0 / x, 0.0, 1.0, n=10)


class TestSimpson:
    def test_exact_for_cubics(self):
        assert simpson_rule(lambda x: x**3, 0.0, 2.0, n=2) == pytest.approx(
            4.0, abs=1e-14
        )
        assert simpson_rule(lambda x: x**2, 0.0, 1.0, n=2) == pytest.approx(
            1 / 3, abs=1e-15
        )

    def test_sine_over_half_period(self):
        assert simpson_rule(math.sin, 0.0, math.pi, n=100) == pytest.approx(
            2.0, abs=1e-7
        )

    def test_error_decreases_by_sixteen_when_n_doubles(self):
        e1 = abs(simpson_rule(math.sin, 0.0, math.pi, n=10) - 2.0)
        e2 = abs(simpson_rule(math.sin, 0.0, math.pi, n=20) - 2.0)
        assert e1 / e2 == pytest.approx(16.0, rel=0.1)

    def test_more_accurate_than_trapezoid_for_same_n(self):
        simpson_error = abs(simpson_rule(math.exp, 0.0, 1.0, n=20) - (math.e - 1))
        trapezoid_error = abs(trapezoid_rule(math.exp, 0.0, 1.0, n=20) - (math.e - 1))
        assert simpson_error < trapezoid_error / 100

    def test_reversed_bounds_negate(self):
        assert simpson_rule(math.exp, 1.0, 0.0, n=20) == pytest.approx(
            -simpson_rule(math.exp, 0.0, 1.0, n=20)
        )

    @pytest.mark.parametrize("n", [1, 3, 7, 0, -2, 4.0, True])
    def test_n_must_be_even_integer_at_least_two(self, n):
        with pytest.raises(InvalidInputError):
            simpson_rule(math.sin, 0.0, 1.0, n=n)


class TestAdaptive:
    def test_smooth_integrand(self):
        result = adaptive_integral(math.sin, 0.0, math.pi)
        assert result.value == pytest.approx(2.0, abs=1e-12)
        assert result.error_estimate < 1e-8

    def test_integrable_singularity(self):
        result = adaptive_integral(lambda x: x**-0.5, 0.0, 1.0)
        assert result.value == pytest.approx(2.0, abs=1e-8)

    def test_infinite_bounds(self):
        assert adaptive_integral(
            lambda x: math.exp(-x), 0.0, math.inf
        ).value == pytest.approx(1.0)
        gaussian = adaptive_integral(lambda x: math.exp(-x * x), -math.inf, math.inf)
        assert gaussian.value == pytest.approx(math.sqrt(math.pi))

    def test_divergent_integral_is_reported(self):
        with pytest.raises(ConvergenceError):
            adaptive_integral(lambda x: 1.0 / x, 1.0, math.inf)

    def test_non_finite_integrand_is_reported(self):
        with pytest.raises(ConvergenceError):
            adaptive_integral(lambda x: float("nan"), 0.0, 1.0)

    def test_nan_bound(self):
        with pytest.raises(InvalidInputError):
            adaptive_integral(math.sin, float("nan"), 1.0)

    @pytest.mark.parametrize("kwargs", [{"abs_tol": 0.0}, {"rel_tol": -1.0}])
    def test_invalid_tolerances(self, kwargs):
        with pytest.raises(InvalidInputError):
            adaptive_integral(math.sin, 0.0, 1.0, **kwargs)
