"""Numerical root finding for polynomials of arbitrary degree."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from scientificlab.utils.exceptions import (
    InfiniteSolutionsError,
    InvalidInputError,
    NoSolutionError,
)
from scientificlab.utils.validation import require_finite, require_non_negative


@dataclass(frozen=True)
class PolynomialRoots:
    """Numerical roots of a polynomial.

    Attributes:
        coefficients: Coefficients, highest degree first, leading zeros removed.
        degree: Degree of the polynomial (equals the number of roots counted
            with multiplicity).
        roots: Roots sorted by (real part, imaginary part). Roots whose
            imaginary part is negligible are returned as ``float``.
    """

    coefficients: tuple[float, ...]
    degree: int
    roots: tuple[float | complex, ...]


def polynomial_roots(
    coefficients: Sequence[float],
    *,
    imag_tolerance: float = 1e-9,
) -> PolynomialRoots:
    """Find all complex roots of a polynomial numerically.

    The roots are the eigenvalues of the companion matrix (``numpy.roots``).
    These are floating-point **approximations**, not symbolic results.

    Limitations:
        * Roots of multiplicity ``m > 1`` are ill-conditioned: expect errors of
          order ``eps**(1/m)`` (about 1e-8 for a double root), and a real double
          root may come back as a close complex pair.
        * Polynomials of high degree with widely varying coefficients can be
          badly conditioned. Check residuals for critical uses.

    Args:
        coefficients: Coefficients from the highest degree to the constant term,
            e.g. ``[1, -6, 11, -6]`` for ``x**3 - 6*x**2 + 11*x - 6``.
        imag_tolerance: A root ``z`` is returned as real when
            ``abs(z.imag) <= imag_tolerance * max(1, abs(z))``.

    Returns:
        A :class:`PolynomialRoots`.

    Raises:
        InvalidInputError: On an empty sequence, non-finite coefficient, or
            negative tolerance.
        InfiniteSolutionsError: If every coefficient is zero.
        NoSolutionError: If the polynomial is a non-zero constant.
    """
    tolerance = require_non_negative(imag_tolerance, "imag_tolerance")
    values = [
        require_finite(c, f"coefficients[{i}]") for i, c in enumerate(coefficients)
    ]
    if not values:
        raise InvalidInputError("coefficients must not be empty.")

    while values and values[0] == 0.0:
        values.pop(0)
    if not values:
        raise InfiniteSolutionsError("The zero polynomial vanishes everywhere.")
    if len(values) == 1:
        raise NoSolutionError(f"The constant polynomial {values[0]} has no roots.")

    raw = np.roots(np.array(values, dtype=float))
    cleaned: list[float | complex] = []
    for z in raw:
        value = complex(z)
        if abs(value.imag) <= tolerance * max(1.0, abs(value)):
            cleaned.append(float(value.real))
        else:
            cleaned.append(value)
    cleaned.sort(key=lambda r: (complex(r).real, complex(r).imag))
    return PolynomialRoots(
        coefficients=tuple(values),
        degree=len(values) - 1,
        roots=tuple(cleaned),
    )
