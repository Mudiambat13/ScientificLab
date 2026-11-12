"""DC circuits: Ohm's law, power, energy and resistor networks (SI units).

Conventions:
    * Voltage in V, current in A, resistance in ohm, power in W, energy in J.
    * Voltage and current are signed (direction of the reference axes);
      resistance is positive, so voltage and current share the same sign.
    * Ideal lumped elements, steady DC state, constant resistance (no
      temperature dependence).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from scientificlab.utils.exceptions import InvalidInputError, PhysicalInconsistencyError
from scientificlab.utils.validation import (
    require_finite,
    require_n_given,
    require_non_negative,
    require_positive,
)


@dataclass(frozen=True)
class OhmsLawResult:
    """Consistent set of quantities of a resistor.

    Attributes:
        voltage: Voltage across the resistor, in V.
        current: Current through it, in A.
        resistance: Resistance, in ohm (strictly positive).
        power: Dissipated power ``V*I``, in W (never negative).
    """

    voltage: float
    current: float
    resistance: float
    power: float


def ohms_law(
    voltage: float | None = None,
    current: float | None = None,
    resistance: float | None = None,
) -> OhmsLawResult:
    """Solve ``V = I * R`` for the missing quantity.

    Exactly two of the three arguments must be given.

    Args:
        voltage: Voltage in V (signed).
        current: Current in A (signed).
        resistance: Resistance in ohm (strictly positive).

    Returns:
        An :class:`OhmsLawResult` with all three quantities and the power.

    Raises:
        InvalidInputError: If not exactly two quantities are given, a value is
            not a finite real number, a given resistance is not positive, or
            ``voltage == current == 0`` (resistance undetermined).
        PhysicalInconsistencyError: If voltage and current have opposite signs
            (negative resistance, which a passive resistor cannot have), or
            one is zero while the other is not.
    """
    require_n_given(
        {"voltage": voltage, "current": current, "resistance": resistance}, 2
    )
    if resistance is not None:
        r = require_positive(resistance, "resistance")
        if voltage is not None:
            v = require_finite(voltage, "voltage")
            i = v / r
        else:
            i = require_finite(current, "current")  # type: ignore[arg-type]
            v = i * r
        return OhmsLawResult(v, i, r, v * i)

    v = require_finite(voltage, "voltage")  # type: ignore[arg-type]
    i = require_finite(current, "current")  # type: ignore[arg-type]
    if v == 0.0 and i == 0.0:
        raise InvalidInputError(
            "Resistance is undetermined when voltage and current are 0."
        )
    if i == 0.0:
        raise PhysicalInconsistencyError(
            "Non-zero voltage with zero current implies infinite resistance."
        )
    if v == 0.0 or (v > 0.0) != (i > 0.0):
        raise PhysicalInconsistencyError(
            "Voltage and current must have the same sign for a positive resistance "
            f"(got V={v}, I={i})."
        )
    return OhmsLawResult(v, i, v / i, v * i)


def electric_power(
    voltage: float | None = None,
    current: float | None = None,
    resistance: float | None = None,
) -> float:
    """Electrical power in W from two of ``voltage``, ``current``, ``resistance``.

    Uses ``P = V*I``, ``P = I**2 * R`` or ``P = V**2 / R``. The result is the
    power **absorbed** by the element; for ``V*I`` it is negative when the
    element delivers power (opposite sign convention for V and I).

    Raises:
        InvalidInputError: If not exactly two quantities are given, or a value
            is invalid (a given resistance must be strictly positive).
    """
    require_n_given(
        {"voltage": voltage, "current": current, "resistance": resistance}, 2
    )
    if voltage is not None and current is not None:
        return require_finite(voltage, "voltage") * require_finite(current, "current")
    r = require_positive(resistance, "resistance")  # type: ignore[arg-type]
    if current is not None:
        i = require_finite(current, "current")
        return i * i * r
    v = require_finite(voltage, "voltage")  # type: ignore[arg-type]
    return v * v / r


def electric_energy(power: float, time: float) -> float:
    """Energy for a constant power: ``E = P * t`` (J).

    Args:
        power: Power in W (signed).
        time: Duration in s (non-negative).
    """
    return require_finite(power, "power") * require_non_negative(time, "time")


def series_resistance(resistances: Sequence[float]) -> float:
    """Equivalent resistance of resistors in series: ``R = sum(R_i)`` (ohm).

    Args:
        resistances: Non-negative resistances (0 models an ideal wire); at
            least one value.
    """
    values = _resistances(resistances)
    return math.fsum(values)


def parallel_resistance(resistances: Sequence[float]) -> float:
    """Equivalent resistance in parallel: ``1/R = sum(1/R_i)`` (ohm).

    Args:
        resistances: Non-negative resistances; at least one value. If any
            resistance is 0 (ideal short circuit), the result is exactly 0.
    """
    values = _resistances(resistances)
    if any(value == 0.0 for value in values):
        return 0.0
    return 1.0 / math.fsum(1.0 / value for value in values)


def _resistances(resistances: Sequence[float]) -> list[float]:
    if isinstance(resistances, (str, bytes)) or not isinstance(resistances, Sequence):
        raise InvalidInputError("resistances must be a sequence of numbers.")
    if len(resistances) == 0:
        raise InvalidInputError("At least one resistance is required.")
    return [
        require_non_negative(value, f"resistances[{index}]")
        for index, value in enumerate(resistances)
    ]
