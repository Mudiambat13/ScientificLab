import math

import pytest
import sympy as sp

from scientificlab.mathematics.calculus import limit, series_sum, taylor_series
from scientificlab.utils.exceptions import InvalidInputError, SolverError

x = sp.Symbol("x")


class TestLimit:
    def test_classic_limits(self):
        assert limit("sin(x)/x", "x", 0) == 1
        assert limit("(x**2 - 1)/(x - 1)", "x", 1) == 2
        assert limit("(1 + 1/x)**x", "x", "oo") == sp.E
        assert limit("1/x", "x", "oo") == 0

    def test_one_sided_limits_diverge_in_opposite_directions(self):
        assert limit("1/x", "x", 0, "+") == sp.oo
        assert limit("1/x", "x", 0, "-") == -sp.oo

    def test_two_sided_limit_of_even_pole_is_infinite(self):
        assert limit("1/x**2", "x", 0) == sp.oo

    def test_two_sided_limit_that_does_not_exist_raises(self):
        with pytest.raises(SolverError):
            limit("1/x", "x", 0)

    def test_oscillating_limit_raises(self):
        with pytest.raises(SolverError):
            limit("sin(1/x)", "x", 0)

    def test_invalid_direction(self):
        with pytest.raises(InvalidInputError):
            limit("1/x", "x", 0, "left")


class TestTaylor:
    def test_maclaurin_sine(self):
        assert taylor_series("sin(x)", "x", 0, 5) == x - x**3 / 6 + x**5 / 120

    def test_maclaurin_exponential(self):
        assert taylor_series("exp(x)", "x", 0, 3) == 1 + x + x**2 / 2 + x**3 / 6

    def test_order_zero_is_the_value_at_the_point(self):
        assert taylor_series("cos(x)", "x", 0, 0) == 1

    def test_expansion_around_another_point(self):
        expected = sp.expand((x - 1) - (x - 1) ** 2 / 2 + (x - 1) ** 3 / 3)
        assert taylor_series("log(x)", "x", 1, 3) == expected

    def test_polynomial_is_its_own_series(self):
        assert taylor_series("x**2 + 1", "x", 0, 5) == x**2 + 1

    def test_error_is_of_the_next_order(self):
        poly = taylor_series("sin(x)", "x", 0, 5)
        approx = float(poly.subs(x, 0.1))
        # Remainder ~ x**7 / 5040 ~ 2e-11
        assert abs(approx - math.sin(0.1)) < 1e-10
        assert abs(approx - math.sin(0.1)) > 1e-13  # truncation really is present

    @pytest.mark.parametrize("order", [-1, 2.5, True])
    def test_invalid_order(self, order):
        with pytest.raises(InvalidInputError):
            taylor_series("sin(x)", "x", 0, order)


class TestSeriesSum:
    def test_basel_problem(self):
        assert series_sum("1/n**2", "n", 1, "oo") == sp.pi**2 / 6

    def test_geometric_series(self):
        assert series_sum("(1/2)**n", "n", 0, "oo") == 2

    def test_finite_sum(self):
        assert series_sum("n", "n", 1, 100) == 5050

    def test_divergent_series_raises(self):
        with pytest.raises(SolverError):
            series_sum("1/n", "n", 1, "oo")

    def test_no_closed_form_raises(self):
        with pytest.raises(SolverError):
            series_sum("1/n**n", "n", 1, "oo")
