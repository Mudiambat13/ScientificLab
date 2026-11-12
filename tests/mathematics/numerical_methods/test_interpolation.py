import numpy as np
import pytest

from scientificlab.mathematics.numerical_methods import (
    linear_interpolation,
    polynomial_interpolation,
)
from scientificlab.utils.exceptions import InvalidInputError


class TestLinearInterpolation:
    XS = (0.0, 1.0, 2.0)
    YS = (0.0, 10.0, 20.0)

    def test_interior_points(self):
        assert linear_interpolation(self.XS, self.YS, 0.5) == pytest.approx(5.0)
        assert linear_interpolation(self.XS, self.YS, 1.25) == pytest.approx(12.5)

    def test_nodes_are_reproduced(self):
        for x, y in zip(self.XS, self.YS, strict=True):
            assert linear_interpolation(self.XS, self.YS, x) == pytest.approx(y)

    def test_non_uniform_spacing(self):
        assert linear_interpolation([0, 1, 5], [0, 2, 10], 3.0) == pytest.approx(6.0)

    def test_out_of_range_is_rejected_by_default(self):
        with pytest.raises(InvalidInputError, match="outside"):
            linear_interpolation(self.XS, self.YS, 3.0)
        with pytest.raises(InvalidInputError):
            linear_interpolation(self.XS, self.YS, -0.1)

    def test_explicit_extrapolation(self):
        assert linear_interpolation(
            self.XS, self.YS, 3.0, extrapolate=True
        ) == pytest.approx(30.0)
        assert linear_interpolation(
            self.XS, self.YS, -1.0, extrapolate=True
        ) == pytest.approx(-10.0)

    def test_second_order_accuracy(self):
        def error(n: int) -> float:
            xs = np.linspace(0.0, 1.0, n + 1)
            return abs(
                linear_interpolation(xs, xs**2, 0.5 + 0.25 / n) - (0.5 + 0.25 / n) ** 2
            )

        assert error(10) / error(20) == pytest.approx(4.0, rel=0.3)

    @pytest.mark.parametrize(
        ("xs", "ys"),
        [
            ([0, 1, 1], [0, 1, 2]),
            ([2, 1, 0], [0, 1, 2]),
            ([0], [1]),
            ([0, 1], [1, 2, 3]),
        ],
    )
    def test_invalid_data(self, xs, ys):
        with pytest.raises(InvalidInputError):
            linear_interpolation(xs, ys, 0.5)

    def test_non_finite_query(self):
        with pytest.raises(InvalidInputError):
            linear_interpolation(self.XS, self.YS, float("nan"))


class TestPolynomialInterpolation:
    def test_recovers_a_parabola_exactly(self):
        poly = polynomial_interpolation([0, 1, 2], [0, 1, 4])
        assert poly.degree == 2
        assert poly(3.0) == pytest.approx(9.0)
        assert poly(-1.5) == pytest.approx(2.25)

    def test_reproduces_nodes(self):
        xs = [0.0, 0.5, 2.0, 3.5]
        ys = [1.0, -2.0, 0.5, 4.0]
        poly = polynomial_interpolation(xs, ys)
        for x, y in zip(xs, ys, strict=True):
            assert poly(x) == pytest.approx(y, abs=1e-12)

    def test_recovers_a_cubic_from_four_points_at_array_inputs(self):
        def cubic(x):
            return 2 * x**3 - x**2 + 3 * x - 5

        nodes = np.array([-1.0, 0.0, 1.0, 2.0])
        poly = polynomial_interpolation(nodes, cubic(nodes))
        query = np.linspace(-2, 3, 7)
        result = poly(query)
        assert isinstance(result, np.ndarray)
        assert result == pytest.approx(cubic(query), abs=1e-10)

    def test_node_order_is_irrelevant(self):
        a = polynomial_interpolation([0, 1, 2], [0, 1, 4])
        b = polynomial_interpolation([2, 0, 1], [4, 0, 1])
        assert a(1.7) == pytest.approx(b(1.7))

    def test_single_node_is_a_constant(self):
        assert polynomial_interpolation([3.0], [7.0])(100.0) == 7.0

    def test_runge_phenomenon_is_a_real_limitation(self):
        def runge(x):
            return 1.0 / (1.0 + 25.0 * x * x)

        nodes = np.linspace(-1.0, 1.0, 15)
        poly = polynomial_interpolation(nodes, runge(nodes))
        # Perfect at the nodes, but badly wrong between them near the ends.
        assert poly(0.97) == pytest.approx(runge(0.97), abs=10.0)
        assert abs(poly(0.97) - runge(0.97)) > 0.5

    def test_scalar_in_scalar_out(self):
        assert isinstance(polynomial_interpolation([0, 1], [0, 1])(0.5), float)

    def test_duplicate_nodes_are_rejected(self):
        with pytest.raises(InvalidInputError, match="distinct"):
            polynomial_interpolation([0, 1, 1], [0, 1, 2])

    def test_length_mismatch(self):
        with pytest.raises(InvalidInputError):
            polynomial_interpolation([0, 1], [0, 1, 2])

    def test_non_finite_evaluation_point(self):
        with pytest.raises(InvalidInputError):
            polynomial_interpolation([0, 1], [0, 1])(float("inf"))
