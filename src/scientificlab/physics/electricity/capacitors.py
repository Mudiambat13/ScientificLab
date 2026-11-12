"""Charge and capacitors (SI units: C, F, V, J, A, s, m).

Conventions: ideal capacitor, linear dielectric, no leakage, no fringing
field. Voltage may be negative (reversed polarity); capacitance is positive.
"""

from scientificlab.constants import VACUUM_PERMITTIVITY
from scientificlab.utils.exceptions import InvalidInputError, PhysicalInconsistencyError
from scientificlab.utils.validation import (
    require_finite,
    require_non_negative,
    require_positive,
)


def charge_from_current(current: float, time: float) -> float:
    """Charge moved by a constant current: ``Q = I * t`` (C)."""
    return require_finite(current, "current") * require_non_negative(time, "time")


def capacitor_charge(capacitance: float, voltage: float) -> float:
    """Charge stored on a capacitor: ``Q = C * V`` (C)."""
    return require_positive(capacitance, "capacitance") * require_finite(
        voltage, "voltage"
    )


def capacitance_from_charge(charge: float, voltage: float) -> float:
    """Capacitance from ``C = Q / V`` (F).

    Raises:
        InvalidInputError: If ``voltage`` is 0 (capacitance undetermined) or an
            argument is invalid.
        PhysicalInconsistencyError: If charge and voltage have opposite signs
            (negative capacitance).
    """
    q = require_finite(charge, "charge")
    v = require_finite(voltage, "voltage")
    if v == 0.0:
        raise InvalidInputError("voltage must be non-zero to determine a capacitance.")
    if q == 0.0 or (q > 0.0) != (v > 0.0):
        raise PhysicalInconsistencyError(
            f"Charge and voltage must be non-zero with the same sign (Q={q}, V={v})."
        )
    return q / v


def capacitor_energy(capacitance: float, voltage: float) -> float:
    """Energy stored in a capacitor: ``E = C * V**2 / 2`` (J)."""
    v = require_finite(voltage, "voltage")
    return 0.5 * require_positive(capacitance, "capacitance") * v * v


def parallel_plate_capacitance(
    area: float,
    separation: float,
    relative_permittivity: float = 1.0,
) -> float:
    """Capacitance of a parallel-plate capacitor: ``C = eps_r*eps_0*A/d`` (F).

    Valid when the plate dimensions are much larger than the separation
    (edge effects neglected). ``eps_0`` is the CODATA 2022 value.

    Args:
        area: Plate area in m**2 (strictly positive).
        separation: Plate separation in m (strictly positive).
        relative_permittivity: Dielectric constant ``eps_r`` (>= 1 for ordinary
            matter; 1 for vacuum, ~1.0006 for air).
    """
    return (
        require_positive(relative_permittivity, "relative_permittivity")
        * VACUUM_PERMITTIVITY
        * require_positive(area, "area")
        / require_positive(separation, "separation")
    )
