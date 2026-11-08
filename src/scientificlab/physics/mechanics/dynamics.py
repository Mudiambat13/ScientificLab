"""Newtonian dynamics in SI units (kg, m, s, N, kg*m/s).

Conventions:
    * Forces and accelerations along one axis are signed scalars.
    * Masses are strictly positive.
    * Models are classical (non-relativistic) with point masses.
"""

from scientificlab.constants import STANDARD_GRAVITY
from scientificlab.utils.validation import (
    require_finite,
    require_non_negative,
    require_positive,
)


def net_force(mass: float, acceleration: float) -> float:
    """Newton's second law: ``F = m * a`` (N).

    Args:
        mass: Mass in kg (strictly positive).
        acceleration: Signed acceleration in m/s**2.
    """
    return require_positive(mass, "mass") * require_finite(acceleration, "acceleration")


def acceleration_from_force(force: float, mass: float) -> float:
    """Acceleration produced by a net force: ``a = F / m`` (m/s**2)."""
    return require_finite(force, "force") / require_positive(mass, "mass")


def weight(mass: float, gravity: float = STANDARD_GRAVITY) -> float:
    """Weight (gravitational force magnitude): ``W = m * g`` (N).

    Args:
        mass: Mass in kg (strictly positive).
        gravity: Local gravitational acceleration in m/s**2 (strictly positive);
            defaults to standard gravity 9.80665.
    """
    return require_positive(mass, "mass") * require_positive(gravity, "gravity")


def friction_force(coefficient: float, normal_force: float) -> float:
    """Coulomb friction magnitude: ``F = mu * N`` (N).

    Valid as the kinetic friction force, or as the **maximum** static friction
    force. It does not tell whether an object actually slides.

    Args:
        coefficient: Dimensionless friction coefficient ``mu`` (non-negative).
        normal_force: Magnitude of the normal force in N (non-negative).
    """
    return require_non_negative(coefficient, "coefficient") * require_non_negative(
        normal_force, "normal_force"
    )


def centripetal_force(mass: float, speed: float, radius: float) -> float:
    """Net force needed for circular motion: ``F = m * v**2 / r`` (N)."""
    v = require_finite(speed, "speed")
    return require_positive(mass, "mass") * v * v / require_positive(radius, "radius")


def momentum(mass: float, velocity: float) -> float:
    """Linear momentum ``p = m * v`` (kg*m/s), signed along the axis."""
    return require_positive(mass, "mass") * require_finite(velocity, "velocity")
