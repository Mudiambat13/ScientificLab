"""Fluid dynamics: flow rates, continuity and Bernoulli (SI units).

Model for the continuity and Bernoulli relations: steady flow of an
**incompressible, inviscid** fluid along a streamline. Viscous losses,
turbulence, compressibility (gases at high speed, Mach > 0.3) and pumps or
turbines are not modelled, so real pressure drops are larger than predicted.
"""

from scientificlab.constants import STANDARD_GRAVITY
from scientificlab.utils.exceptions import PhysicalInconsistencyError
from scientificlab.utils.validation import (
    require_finite,
    require_non_negative,
    require_positive,
)


def volumetric_flow_rate(area: float, velocity: float) -> float:
    """Volumetric flow rate ``Q = A * v`` (m**3/s).

    Args:
        area: Cross-section area in m**2 (> 0).
        velocity: Mean fluid speed through it in m/s (>= 0).
    """
    return require_positive(area, "area") * require_non_negative(velocity, "velocity")


def mass_flow_rate(fluid_density: float, area: float, velocity: float) -> float:
    """Mass flow rate ``m_dot = rho * A * v`` (kg/s)."""
    return require_positive(fluid_density, "fluid_density") * volumetric_flow_rate(
        area, velocity
    )


def continuity_velocity(area_1: float, velocity_1: float, area_2: float) -> float:
    """Speed after a change of section: ``v_2 = A_1 v_1 / A_2`` (m/s).

    Incompressible continuity, from ``A_1 v_1 = A_2 v_2`` (conservation of mass).
    """
    return volumetric_flow_rate(area_1, velocity_1) / require_positive(area_2, "area_2")


def bernoulli_pressure(
    fluid_density: float,
    pressure_1: float,
    velocity_1: float,
    height_1: float,
    velocity_2: float,
    height_2: float,
    gravity: float = STANDARD_GRAVITY,
) -> float:
    """Pressure at point 2 from Bernoulli's equation (Pa).

    ``P_1 + rho v_1**2/2 + rho g h_1 = P_2 + rho v_2**2/2 + rho g h_2``

    Pressures are **absolute**. If you work with gauge pressures, the result
    is the gauge pressure at point 2 but the negative-pressure check below is
    then not meaningful.

    Args:
        fluid_density: Density in kg/m**3 (> 0).
        pressure_1: Absolute pressure at point 1 in Pa (>= 0).
        velocity_1: Speed at point 1 in m/s (>= 0).
        height_1: Height of point 1 above a common reference in m.
        velocity_2: Speed at point 2 in m/s (>= 0).
        height_2: Height of point 2 above the same reference in m.
        gravity: Gravitational acceleration in m/s**2 (> 0).

    Raises:
        PhysicalInconsistencyError: If the computed absolute pressure at point 2
            is negative: a liquid cannot sustain it (cavitation), so the
            requested flow is impossible under this model.
    """
    rho = require_positive(fluid_density, "fluid_density")
    p_1 = require_non_negative(pressure_1, "pressure_1")
    v_1 = require_non_negative(velocity_1, "velocity_1")
    v_2 = require_non_negative(velocity_2, "velocity_2")
    h_1 = require_finite(height_1, "height_1")
    h_2 = require_finite(height_2, "height_2")
    g = require_positive(gravity, "gravity")
    p_2 = p_1 + 0.5 * rho * (v_1 * v_1 - v_2 * v_2) + rho * g * (h_1 - h_2)
    if p_2 < 0.0:
        raise PhysicalInconsistencyError(
            f"Bernoulli gives a negative absolute pressure ({p_2:.6g} Pa) at point 2: "
            "the fluid would cavitate, so this flow is impossible."
        )
    return p_2
