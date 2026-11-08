import pytest

from scientificlab.mathematics.algebra import polynomial_roots
from scientificlab.utils.exceptions import (
    InfiniteSolutionsError,
    InvalidInputError,
    NoSolutionError,
)


def test_cubic_with_integer_roots():
    result = polynomial_roots([1, -6, 11, -6])  # (x-1)(x-2)(x-3)
    assert result.degree == 3
    assert result.roots == pytest.approx((1.0, 2.0, 3.0))
    assert all(isinstance(r, float) for r in result.roots)


def test_complex_roots_of_x2_plus_1():
    result = polynomial_roots([1, 0, 1])
    assert result.roots[0] == pytest.approx(-1j)
    assert result.roots[1] == pytest.approx(1j)


def test_quartic_x4_minus_1():
    result = polynomial_roots([1, 0, 0, 0, -1])
    expected = sorted([-1, 1, -1j, 1j], key=lambda z: (z.real, z.imag))
    assert len(result.roots) == 4
    for got, want in zip(result.roots, expected, strict=True):
        assert got == pytest.approx(want, abs=1e-12)


def test_leading_zeros_are_stripped():
    result = polynomial_roots([0, 0, 1, -5, 6])
    assert result.degree == 2
    assert result.coefficients == (1.0, -5.0, 6.0)
    assert result.roots == pytest.approx((2.0, 3.0))


def test_residuals_are_small():
    coefficients = [2.0, -3.0, -11.0, 6.0]  # roots -2, 0.5, 3
    for root in polynomial_roots(coefficients).roots:
        value = 0.0
        for c in coefficients:
            value = value * root + c
        assert abs(value) < 1e-10


def test_non_zero_constant_has_no_roots():
    with pytest.raises(NoSolutionError):
        polynomial_roots([0, 4])


def test_zero_polynomial_has_infinite_roots():
    with pytest.raises(InfiniteSolutionsError):
        polynomial_roots([0, 0, 0])


def test_empty_input_is_rejected():
    with pytest.raises(InvalidInputError):
        polynomial_roots([])


def test_non_finite_coefficient_is_rejected():
    with pytest.raises(InvalidInputError):
        polynomial_roots([1, float("nan"), 1])
