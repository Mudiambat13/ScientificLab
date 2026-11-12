"""Optics: refraction (Snell-Descartes), thin lenses, magnification."""

from scientificlab.physics.optics.lenses import (
    NEAR_POINT,
    LensResult,
    lens_power,
    lensmaker_focal_length,
    linear_magnification,
    magnifier_angular_magnification,
    telescope_magnification,
    thin_lens_equation,
)
from scientificlab.physics.optics.refraction import (
    RefractionResult,
    critical_angle,
    reflection_angle,
    snell_refraction,
)

__all__ = [
    "NEAR_POINT",
    "LensResult",
    "RefractionResult",
    "critical_angle",
    "lens_power",
    "lensmaker_focal_length",
    "linear_magnification",
    "magnifier_angular_magnification",
    "reflection_angle",
    "snell_refraction",
    "telescope_magnification",
    "thin_lens_equation",
]
