"""Work, energy and power in SI units (J, W).

Conventions:
    * Angles are given in degrees and measured between the force and the
      displacement (or velocity) vectors.
    * Gravitational potential energy is ``m*g*h`` with ``h`` measured from a
      chosen reference level (``h`` may be negative); valid for heights small
      compared with the Earth's radius (uniform field).
    * Cosines of exact right angles are returned as ~1e-16, not exactly 0.
"""

import math

from scientificlab.constants import STANDARD_GRAVITY
from scientificlab.utils.validation import (
    require_finite,
    require_non_negative,
    require_positive,
)


def _cosine_degrees(angle_degrees: float) -> float:
    """Cosine of an angle given in degrees, after validation."""
    return math.cos(math.radians(require_finite(angle_degrees, "angle_degrees")))


def kinetic_energy(mass: float, velocity: float) -> float:
    """Translational kinetic energy ``E_k = m*v**2 / 2`` (J).

    Args:
        mass: Mass in kg (strictly positive).
        velocity: Velocity in m/s; only its square matters, so the sign is free.
    """
    v = require_finite(velocity, "velocity")
    return 0.5 * require_positive(mass, "mass") * v * v


def potential_energy(
    mass: float,
    height: float,
    gravity: float = STANDARD_GRAVITY,
) -> float:
    """Gravitational potential energy ``E_p = m*g*h`` (J).

    Args:
        mass: Mass in kg (strictly positive).
        height: Height above the reference level in m (may be negative).
        gravity: Gravitational acceleration in m/s**2 (strictly positive).
    """
    return (
        require_positive(mass, "mass")
        * require_positive(gravity, "gravity")
        * require_finite(height, "height")
    )


def mechanical_energy(
    mass: float,
    velocity: float,
    height: float,
    gravity: float = STANDARD_GRAVITY,
) -> float:
    """Total mechanical energy ``E = E_k + E_p`` (J)."""
    return kinetic_energy(mass, velocity) + potential_energy(mass, height, gravity)


def work(force: float, displacement: float, angle_degrees: float = 0.0) -> float:
    """Work of a constant force: ``W = F * d * cos(theta)`` (J).

    Args:
        force: Force magnitude in N (non-negative).
        displacement: Displacement magnitude in m (non-negative).
        angle_degrees: Angle between force and displacement, in degrees. Between
            90 and 270 degrees the work is negative.
    """
    # Magnitudes only: the direction is carried by the angle.
    f = require_non_negative(force, "force")
    d = require_non_negative(displacement, "displacement")
    return f * d * _cosine_degrees(angle_degrees)


def average_power(energy_transferred: float, time: float) -> float:
    """Average power ``P = W / t`` (W).

    Args:
        energy_transferred: Work done or energy transferred, in J.
        time: Duration in s (strictly positive).
    """
    return require_finite(energy_transferred, "energy_transferred") / require_positive(
        time, "time"
    )


def power_from_force(
    force: float, velocity: float, angle_degrees: float = 0.0
) -> float:
    """Instantaneous power of a force: ``P = F * v * cos(theta)`` (W).

    Args:
        force: Force magnitude in N (non-negative).
        velocity: Speed in m/s (non-negative).
        angle_degrees: Angle between force and velocity, in degrees.
    """
    f = require_non_negative(force, "force")
    v = require_non_negative(velocity, "velocity")
    return f * v * _cosine_degrees(angle_degrees)
