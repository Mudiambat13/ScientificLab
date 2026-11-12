"""Thin lenses and simple optical instruments (SI units: m, diopter).

Sign convention ("real-is-positive"), light travelling from left to right:

* ``object_distance > 0`` for a real object in front of the lens
  (``< 0`` for a virtual object).
* ``image_distance > 0`` for a real image behind the lens (``< 0`` for a
  virtual image on the object side).
* ``focal_length > 0`` for a converging lens, ``< 0`` for a diverging lens.
* Linear magnification ``m = -d_i/d_o``: ``m < 0`` inverted, ``m > 0`` upright.

Model: thin lens in the paraxial approximation (rays close to the axis and
small angles); aberrations and lens thickness are ignored.
"""

import math
from dataclasses import dataclass
from numbers import Real

from scientificlab.utils.exceptions import InvalidInputError, PhysicalInconsistencyError
from scientificlab.utils.validation import (
    require_finite,
    require_n_given,
    require_positive,
)

NEAR_POINT: float = 0.25
"""Conventional near point of the normal eye in m (distance of most distinct vision)."""


@dataclass(frozen=True)
class LensResult:
    """Image formed by a thin lens.

    Attributes:
        focal_length: Focal length in m (> 0 converging, < 0 diverging).
        object_distance: Object distance in m.
        image_distance: Image distance in m (> 0 real, < 0 virtual).
        magnification: Linear magnification ``-d_i/d_o``.
        real_image: ``True`` if the image is real (can be projected on a screen).
        upright: ``True`` if the image has the same orientation as the object.
    """

    focal_length: float
    object_distance: float
    image_distance: float
    magnification: float
    real_image: bool
    upright: bool


def _nonzero(value: float, name: str) -> float:
    number = require_finite(value, name)
    if number == 0.0:
        raise InvalidInputError(f"{name} must be non-zero.")
    return number


def _invert_sum(total: float, message: str) -> float:
    """Return ``1/total``, rejecting an infinite result (``total == 0``)."""
    if total == 0.0:
        raise PhysicalInconsistencyError(message)
    return 1.0 / total


def thin_lens_equation(
    focal_length: float | None = None,
    object_distance: float | None = None,
    image_distance: float | None = None,
) -> LensResult:
    """Solve ``1/f = 1/d_o + 1/d_i`` for the missing quantity.

    Exactly two of the three arguments must be given (see the module docstring
    for the sign convention).

    Raises:
        InvalidInputError: If not exactly two values are given, or a value is
            zero or not finite.
        PhysicalInconsistencyError: If the result would be at infinity: the
            object at the focal point (image at infinity), or an image at
            ``-d_o`` (infinite focal length: not a lens).
    """
    require_n_given(
        {
            "focal_length": focal_length,
            "object_distance": object_distance,
            "image_distance": image_distance,
        },
        2,
    )
    f = None if focal_length is None else _nonzero(focal_length, "focal_length")
    d_o = (
        None
        if object_distance is None
        else _nonzero(object_distance, "object_distance")
    )
    d_i = None if image_distance is None else _nonzero(image_distance, "image_distance")
    if f is None:
        assert d_o is not None and d_i is not None
        f = _invert_sum(
            1.0 / d_o + 1.0 / d_i,
            "1/d_o + 1/d_i = 0: the focal length would be infinite (not a lens).",
        )
    elif d_i is None:
        assert d_o is not None
        d_i = _invert_sum(
            1.0 / f - 1.0 / d_o,
            "The object is at the focal point: the image forms at infinity.",
        )
    else:
        d_o = _invert_sum(
            1.0 / f - 1.0 / d_i,
            "The image is at the focal point: the object would be at infinity.",
        )
    assert d_o is not None
    magnification = -d_i / d_o
    return LensResult(
        focal_length=f,
        object_distance=d_o,
        image_distance=d_i,
        magnification=magnification,
        real_image=d_i > 0.0,
        upright=magnification > 0.0,
    )


def linear_magnification(image_distance: float, object_distance: float) -> float:
    """Linear magnification ``m = -d_i / d_o`` (dimensionless, signed)."""
    return -_nonzero(image_distance, "image_distance") / _nonzero(
        object_distance, "object_distance"
    )


def lens_power(focal_length: float) -> float:
    """Optical power ``1/f`` in diopters (m^-1); negative for diverging lenses."""
    return 1.0 / _nonzero(focal_length, "focal_length")


def lensmaker_focal_length(
    refractive_index: float,
    radius_1: float,
    radius_2: float,
    medium_index: float = 1.0,
) -> float:
    """Focal length of a thin lens from its geometry (lensmaker's equation).

    ``1/f = (n_lens/n_medium - 1) * (1/R_1 - 1/R_2)`` (m).

    Radii follow the sign convention: ``R > 0`` if the centre of curvature is
    on the outgoing (right) side of the surface. A biconvex lens has
    ``R_1 > 0`` and ``R_2 < 0``. Use ``math.inf`` for a flat surface.

    Args:
        refractive_index: Index of the lens material (> 0).
        radius_1: Radius of curvature of the first surface in m (non-zero).
        radius_2: Radius of curvature of the second surface in m (non-zero).
        medium_index: Index of the surrounding medium (default 1, air).

    Raises:
        PhysicalInconsistencyError: If the lens has no optical power (flat
            plate, equal curvatures, or the same index as the medium).
    """
    n_lens = require_positive(refractive_index, "refractive_index")
    n_medium = require_positive(medium_index, "medium_index")
    r1 = _radius(radius_1, "radius_1")
    r2 = _radius(radius_2, "radius_2")
    power = (n_lens / n_medium - 1.0) * (1.0 / r1 - 1.0 / r2)
    if power == 0.0:
        raise PhysicalInconsistencyError(
            "The lens has zero optical power (infinite focal length)."
        )
    return 1.0 / power


def _radius(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or math.isnan(value):
        raise InvalidInputError(
            f"{name} must be a real number or math.inf, got {value!r}."
        )
    if value == 0:
        raise InvalidInputError(
            f"{name} must be non-zero (use math.inf for a flat surface)."
        )
    return float(value)


def magnifier_angular_magnification(
    focal_length: float,
    relaxed_eye: bool = False,
    near_point: float = NEAR_POINT,
) -> float:
    """Angular magnification of a simple magnifier.

    ``M = 1 + N/f`` with the image at the near point ``N`` (maximum
    magnification), or ``M = N/f`` with the image at infinity (relaxed eye).

    Args:
        focal_length: Focal length in m (> 0, converging lens).
        relaxed_eye: ``True`` for the image at infinity.
        near_point: Near point ``N`` in m (default 0.25).
    """
    f = require_positive(focal_length, "focal_length")
    n = require_positive(near_point, "near_point")
    return n / f if relaxed_eye else 1.0 + n / f


def telescope_magnification(
    objective_focal_length: float,
    eyepiece_focal_length: float,
) -> float:
    """Angular magnification of a telescope in normal adjustment: ``f_o / f_e``.

    Both focal lengths in m (> 0; magnitude of the magnification, the image of
    a Keplerian telescope is inverted).
    """
    return require_positive(objective_focal_length, "objective_focal_length") / (
        require_positive(eyepiece_focal_length, "eyepiece_focal_length")
    )
