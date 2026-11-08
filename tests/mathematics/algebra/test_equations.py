import cmath
import math

import pytest

from scientificlab.mathematics.algebra import (
    RootNature,
    solve_linear,
    solve_quadratic,
)
from scientificlab.utils.exceptions import (
    InfiniteSolutionsError,
    InvalidInputError,
    NoSolutionError,
)


class TestSolveLinear:
    def test_basic(self):
        assert solve_linear(2, -6).root == pytest.approx(3.0)

    def test_negative_zero_is_normalised(self):
        assert math.copysign(1.0, solve_linear(5, 0).root) == 1.0

    def test_roots_tuple(self):
        assert solve_linear(1, -4).roots == (4.0,)

    def test_a_zero_b_nonzero_has_no_solution(self):
        with pytest.raises(NoSolutionError):
            solve_linear(0, 3)

    def test_a_zero_b_zero_has_infinite_solutions(self):
        with pytest.raises(InfiniteSolutionsError):
            solve_linear(0, 0)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), "2", None, True])
    def test_invalid_input(self, bad):
        with pytest.raises(InvalidInputError):
            solve_linear(bad, 1)


class TestSolveQuadratic:
    def test_two_real_roots(self):
        result = solve_quadratic(a=1, b=-5, c=6)
        assert result.nature is RootNature.TWO_REAL
        assert result.roots == (pytest.approx(2.0), pytest.approx(3.0))
        assert result.discriminant == pytest.approx(1.0)

    def test_roots_sorted_ascending_for_negative_a(self):
        result = solve_quadratic(-1, 5, -6)
        assert result.roots == (pytest.approx(2.0), pytest.approx(3.0))

    def test_double_root(self):
        result = solve_quadratic(1, -2, 1)
        assert result.nature is RootNature.DOUBLE_REAL
        assert result.roots == (1.0, 1.0)

    def test_complex_roots(self):
        result = solve_quadratic(1, 2, 5)  # roots -1 +/- 2i
        assert result.nature is RootNature.COMPLEX_CONJUGATE
        assert result.roots[0] == pytest.approx(complex(-1, -2))
        assert result.roots[1] == pytest.approx(complex(-1, 2))

    def test_pure_imaginary_roots(self):
        result = solve_quadratic(1, 0, 1)
        assert result.roots == (complex(0, -1), complex(0, 1))

    def test_zero_constant_term(self):
        result = solve_quadratic(1, -3, 0)
        assert result.roots == (0.0, 3.0)

    def test_b_zero_real_roots(self):
        result = solve_quadratic(1, 0, -4)
        assert result.roots == (-2.0, 2.0)

    def test_stable_against_cancellation(self):
        # Exact roots of x**2 - 1e8*x + 1 are ~1e-8 and ~1e8 (Vieta: product 1).
        # The textbook formula returns 7.45e-9 for the small root (26% error).
        result = solve_quadratic(1, -1e8, 1)
        small, large = result.roots
        assert large == pytest.approx(1e8, rel=1e-14)
        assert small == pytest.approx(1e-8, rel=1e-12)
        assert small * large == pytest.approx(1.0, rel=1e-12)

    @pytest.mark.parametrize(
        ("a", "b", "c"),
        [(1, -5, 6), (2, 3, -7), (0.5, -1.5, 1.0), (3, 2, 5), (-4, 0, 9)],
    )
    def test_vieta_relations_and_residual(self, a, b, c):
        r1, r2 = solve_quadratic(a, b, c).roots
        assert r1 + r2 == pytest.approx(-b / a)
        assert r1 * r2 == pytest.approx(c / a)
        for r in (r1, r2):
            assert abs(a * r * r + b * r + c) < 1e-12

    def test_complex_roots_match_cmath(self):
        a, b, c = 2.0, 3.0, 4.0
        disc = cmath.sqrt(b * b - 4 * a * c)
        expected = sorted(
            [(-b + disc) / (2 * a), (-b - disc) / (2 * a)], key=lambda z: z.imag
        )
        result = solve_quadratic(a, b, c).roots
        assert result[0] == pytest.approx(expected[0])
        assert result[1] == pytest.approx(expected[1])

    def test_discriminant_tolerance_makes_double_root(self):
        result = solve_quadratic(1, -2, 1 + 1e-13, discriminant_tolerance=1e-9)
        assert result.nature is RootNature.DOUBLE_REAL

    def test_a_zero_is_rejected(self):
        with pytest.raises(InvalidInputError):
            solve_quadratic(0, 2, 1)

    def test_overflowing_discriminant_is_rejected(self):
        with pytest.raises(InvalidInputError):
            solve_quadratic(1, 1e200, 1)

    def test_negative_tolerance_is_rejected(self):
        with pytest.raises(InvalidInputError):
            solve_quadratic(1, 0, 1, discriminant_tolerance=-1.0)

    @pytest.mark.parametrize("bad", [float("nan"), float("-inf"), "x", None])
    def test_invalid_coefficients(self, bad):
        with pytest.raises(InvalidInputError):
            solve_quadratic(1, bad, 1)
