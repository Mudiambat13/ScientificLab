"""Reflection and refraction at a plane interface (geometrical optics).

Conventions: angles are in degrees and measured from the **normal** to the
surface (not from the surface). Refractive indices are relative to vacuum and
strictly positive; ordinary materials have ``n >= 1`` (air about 1.0003,
water 1.33, glass 1.5). Light is treated as rays: diffraction, polarisation
(Fresnel coefficients) and dispersion (index depends on wavelength) are
ignored, so give the index for the wavelength of interest.
"""

import math
from dataclasses import dataclass

from scientificlab.utils.exceptions import InvalidInputError, PhysicalInconsistencyError
from scientificlab.utils.validation import require_finite, require_positive

_RATIO_TOLERANCE = 1e-12


@dataclass(frozen=True)
class RefractionResult:
    """Outcome of a ray hitting an interface.

    Attributes:
        incidence_angle: Angle of incidence, in degrees.
        reflection_angle: Angle of the reflected ray, in degrees (law of
            reflection: equal to the incidence angle).
        refraction_angle: Angle of the transmitted ray in degrees, or ``None``
            when there is no transmitted ray (total internal reflection).
        total_internal_reflection: ``True`` if all the light is reflected.
        critical_angle: Critical angle in degrees if ``n1 > n2`` (total
            internal reflection possible), otherwise ``None``.
    """

    incidence_angle: float
    reflection_angle: float
    refraction_angle: float | None
    total_internal_reflection: bool
    critical_angle: float | None


def reflection_angle(incidence_angle: float) -> float:
    """Law of reflection: the reflected angle equals the incidence angle (degrees)."""
    return _incidence(incidence_angle)


def critical_angle(n1: float, n2: float) -> float:
    """Critical angle for total internal reflection: ``asin(n2/n1)`` (degrees).

    Args:
        n1: Index of the medium the light comes from.
        n2: Index of the medium it tries to enter.

    Raises:
        PhysicalInconsistencyError: If ``n1 <= n2``: there is no critical angle
            when light enters a denser (or equal) medium.
    """
    index_1 = require_positive(n1, "n1")
    index_2 = require_positive(n2, "n2")
    if index_1 <= index_2:
        raise PhysicalInconsistencyError(
            "A critical angle exists only when n1 > n2 "
            f"(got n1={index_1}, n2={index_2})."
        )
    return math.degrees(math.asin(index_2 / index_1))


def snell_refraction(n1: float, n2: float, incidence_angle: float) -> RefractionResult:
    """Apply Snell-Descartes' law ``n1 sin(theta_1) = n2 sin(theta_2)``.

    Total internal reflection is reported in the result, not raised, because
    it is a normal physical outcome. A ratio exceeding 1 by less than 1e-12
    (rounding at exactly the critical angle) is treated as grazing refraction
    (90 degrees).

    Args:
        n1: Refractive index of the incident medium (> 0).
        n2: Refractive index of the second medium (> 0).
        incidence_angle: Angle of incidence in degrees, in ``[0, 90]``.

    Raises:
        InvalidInputError: On invalid indices or an angle outside ``[0, 90]``.
    """
    index_1 = require_positive(n1, "n1")
    index_2 = require_positive(n2, "n2")
    theta = _incidence(incidence_angle)
    critical = math.degrees(math.asin(index_2 / index_1)) if index_1 > index_2 else None
    sine = index_1 * math.sin(math.radians(theta)) / index_2
    if sine > 1.0 + _RATIO_TOLERANCE:
        return RefractionResult(theta, theta, None, True, critical)
    refracted = math.degrees(math.asin(min(sine, 1.0)))
    return RefractionResult(theta, theta, refracted, False, critical)


def _incidence(angle: float) -> float:
    value = require_finite(angle, "incidence_angle")
    if not 0.0 <= value <= 90.0:
        raise InvalidInputError(
            "incidence_angle must be in [0, 90] degrees from the normal, "
            f"got {angle!r}."
        )
    return value
