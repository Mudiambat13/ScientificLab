"""Fluid statics (SI units: kg/m**3, Pa, N, m, m**2, m**3).

Model: incompressible fluid (constant density) at rest in a uniform
gravitational field. Hydrostatic formulas are accurate for liquids over modest
depths, but not for gases over large height differences (density varies).
"""

from scientificlab.constants import STANDARD_GRAVITY
from scientificlab.utils.exceptions import PhysicalInconsistencyError
from scientificlab.utils.validation import (
    require_non_negative,
    require_positive,
)


def density(mass: float, volume: float) -> float:
    """Mass density ``rho = m / V`` (kg/m**3).

    Args:
        mass: Mass in kg (> 0).
        volume: Volume in m**3 (> 0).
    """
    return require_positive(mass, "mass") / require_positive(volume, "volume")


def pressure(force: float, area: float) -> float:
    """Pressure from a force acting normally on a surface: ``P = F / A`` (Pa).

    Args:
        force: Normal force magnitude in N (>= 0).
        area: Contact area in m**2 (> 0).
    """
    return require_non_negative(force, "force") / require_positive(area, "area")


def hydrostatic_pressure(
    fluid_density: float,
    depth: float,
    gravity: float = STANDARD_GRAVITY,
    surface_pressure: float = 0.0,
) -> float:
    """Pressure at a depth ``h`` in a fluid at rest: ``P = P_0 + rho * g * h`` (Pa).

    With the default ``surface_pressure = 0`` the result is the **gauge**
    pressure ``rho*g*h``; pass the atmospheric pressure (101325 Pa) for the
    absolute pressure.

    Args:
        fluid_density: Density in kg/m**3 (> 0).
        depth: Depth below the free surface in m (>= 0).
        gravity: Gravitational acceleration in m/s**2 (> 0).
        surface_pressure: Pressure at the free surface in Pa (>= 0).
    """
    return require_non_negative(surface_pressure, "surface_pressure") + (
        require_positive(fluid_density, "fluid_density")
        * require_positive(gravity, "gravity")
        * require_non_negative(depth, "depth")
    )


def hydraulic_force(input_force: float, input_area: float, output_area: float) -> float:
    """Output force of a hydraulic press: ``F_2 = F_1 * A_2 / A_1`` (N).

    Pascal's principle: pressure applied to an enclosed incompressible fluid
    is transmitted undiminished, so ``F_1/A_1 = F_2/A_2``. The force gain is
    paid for by a proportionally smaller displacement of the output piston
    (energy is conserved). Friction and the height difference between pistons
    are ignored.

    Args:
        input_force: Force on the input piston in N (>= 0).
        input_area: Input piston area in m**2 (> 0).
        output_area: Output piston area in m**2 (> 0).
    """
    return (
        require_non_negative(input_force, "input_force")
        * require_positive(output_area, "output_area")
        / require_positive(input_area, "input_area")
    )


def buoyant_force(
    fluid_density: float,
    displaced_volume: float,
    gravity: float = STANDARD_GRAVITY,
) -> float:
    """Archimedes' buoyant force: ``F_b = rho_fluid * V_displaced * g`` (N).

    The force equals the weight of the displaced fluid and points upward.

    Args:
        fluid_density: Density of the fluid in kg/m**3 (> 0).
        displaced_volume: Volume of fluid displaced (the submerged volume of
            the body) in m**3 (>= 0).
        gravity: Gravitational acceleration in m/s**2 (> 0).
    """
    return (
        require_positive(fluid_density, "fluid_density")
        * require_non_negative(displaced_volume, "displaced_volume")
        * require_positive(gravity, "gravity")
    )


def floats(object_density: float, fluid_density_value: float) -> bool:
    """Whether a body of uniform density floats freely in a fluid.

    ``True`` if the body is less dense than the fluid (it rises or floats);
    ``False`` if it is as dense or denser (neutral buoyancy counts as not
    floating at the surface).
    """
    return require_positive(object_density, "object_density") < require_positive(
        fluid_density_value, "fluid_density_value"
    )


def submerged_fraction(object_density: float, fluid_density_value: float) -> float:
    """Fraction of a floating body's volume below the surface: ``rho_o / rho_f``.

    Raises:
        PhysicalInconsistencyError: If the body is as dense or denser than the
            fluid: it does not float in equilibrium at the surface (it is fully
            submerged), so no fraction below 1 exists.
    """
    rho_object = require_positive(object_density, "object_density")
    rho_fluid = require_positive(fluid_density_value, "fluid_density_value")
    if rho_object >= rho_fluid:
        raise PhysicalInconsistencyError(
            f"A body of density {rho_object} does not float in a fluid of "
            f"density {rho_fluid}."
        )
    return rho_object / rho_fluid
