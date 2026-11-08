"""Closed-form solvers for first- and second-degree equations."""

import cmath
import math
from dataclasses import dataclass
from enum import StrEnum

from scientificlab.utils.exceptions import (
    InfiniteSolutionsError,
    InvalidInputError,
    NoSolutionError,
)
from scientificlab.utils.validation import require_finite, require_non_negative


@dataclass(frozen=True)
class LinearSolution:
    """Solution of ``a*x + b = 0`` with ``a != 0``."""

    a: float
    b: float
    root: float

    @property
    def roots(self) -> tuple[float]:
        """The single root, as a 1-tuple (same shape as other solvers)."""
        return (self.root,)


class RootNature(StrEnum):
    """Nature of the roots of a quadratic equation."""

    TWO_REAL = "two_real"
    DOUBLE_REAL = "double_real"
    COMPLEX_CONJUGATE = "complex_conjugate"


@dataclass(frozen=True)
class QuadraticSolution:
    """Solution of ``a*x**2 + b*x + c = 0``.

    Attributes:
        a: Quadratic coefficient (non-zero).
        b: Linear coefficient.
        c: Constant coefficient.
        discriminant: ``b**2 - 4*a*c`` computed in floating point.
        nature: Whether the roots are two real, one double real, or complex.
        roots: The two roots. Real roots are ``float`` sorted ascending; complex
            roots are ``complex`` ordered by increasing imaginary part.
    """

    a: float
    b: float
    c: float
    discriminant: float
    nature: RootNature
    roots: tuple[float | complex, float | complex]


def solve_linear(a: float, b: float) -> LinearSolution:
    """Solve ``a*x + b = 0``.

    Args:
        a: Coefficient of ``x``.
        b: Constant term.

    Returns:
        The unique solution ``x = -b / a``.

    Raises:
        InvalidInputError: If a coefficient is not a finite real number.
        InfiniteSolutionsError: If ``a == 0`` and ``b == 0`` (every x works).
        NoSolutionError: If ``a == 0`` and ``b != 0`` (contradiction).
    """
    a = require_finite(a, "a")
    b = require_finite(b, "b")
    if a == 0.0:
        if b == 0.0:
            raise InfiniteSolutionsError("0*x + 0 = 0 holds for every x.")
        raise NoSolutionError(f"0*x + {b} = 0 has no solution.")
    return LinearSolution(a=a, b=b, root=-b / a + 0.0)


def solve_quadratic(
    a: float,
    b: float,
    c: float,
    *,
    discriminant_tolerance: float = 0.0,
) -> QuadraticSolution:
    """Solve ``a*x**2 + b*x + c = 0`` over the complex numbers.

    Real roots use the numerically stable formulation (the root of larger
    magnitude is computed first, the other from Vieta's relation
    ``x1 * x2 = c / a``), which avoids catastrophic cancellation when
    ``b**2 >> 4*a*c``.

    The discriminant is evaluated in floating point. A discriminant with
    ``abs(D) <= discriminant_tolerance`` is treated as zero (double root). The
    default tolerance is exactly 0, so nearly-double roots from inexact
    coefficients (e.g. ``0.1``) may be reported as two close real roots or as a
    complex pair.

    Args:
        a: Quadratic coefficient; must be non-zero.
        b: Linear coefficient.
        c: Constant coefficient.
        discriminant_tolerance: Non-negative threshold under which the
            discriminant is considered zero.

    Returns:
        A :class:`QuadraticSolution`.

    Raises:
        InvalidInputError: If a coefficient is not finite, ``a == 0`` (use
            :func:`solve_linear`), or the discriminant overflows.
    """
    a = require_finite(a, "a")
    b = require_finite(b, "b")
    c = require_finite(c, "c")
    tolerance = require_non_negative(discriminant_tolerance, "discriminant_tolerance")
    if a == 0.0:
        raise InvalidInputError(
            "a must be non-zero for a quadratic equation; use solve_linear instead."
        )

    discriminant = b * b - 4.0 * a * c
    if not math.isfinite(discriminant):
        raise InvalidInputError("Coefficients are too large: discriminant overflowed.")

    if abs(discriminant) <= tolerance:
        root = -b / (2.0 * a) + 0.0
        return QuadraticSolution(
            a, b, c, discriminant, RootNature.DOUBLE_REAL, (root, root)
        )

    if discriminant > 0.0:
        sqrt_d = math.sqrt(discriminant)
        q = -0.5 * (b + math.copysign(sqrt_d, b))
        x1 = q / a
        x2 = c / q
        low, high = sorted((x1 + 0.0, x2 + 0.0))
        return QuadraticSolution(
            a, b, c, discriminant, RootNature.TWO_REAL, (low, high)
        )

    real_part = -b / (2.0 * a) + 0.0
    imag_part = cmath.sqrt(discriminant).imag / (2.0 * abs(a))
    return QuadraticSolution(
        a,
        b,
        c,
        discriminant,
        RootNature.COMPLEX_CONJUGATE,
        (complex(real_part, -imag_part), complex(real_part, imag_part)),
    )
