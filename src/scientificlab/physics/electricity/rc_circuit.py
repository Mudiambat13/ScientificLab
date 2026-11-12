"""Series RC circuit driven by a constant source (SI units: V, A, ohm, F, s).

Model: ideal resistor and capacitor in series with a DC source ``V_s`` closed
at ``t = 0``, capacitor initially at ``V_0``. The capacitor voltage obeys
``dV/dt = (V_s - V)/(R*C)`` and relaxes exponentially with time constant
``tau = R*C`` towards ``V_s``. Discharge through the resistor is the special
case ``V_s = 0``. Sources, wires and components are ideal (no leakage,
no internal resistance, no inductance).
"""

import math

from scientificlab.utils.exceptions import PhysicalInconsistencyError
from scientificlab.utils.validation import (
    require_finite,
    require_non_negative,
    require_positive,
)


def rc_time_constant(resistance: float, capacitance: float) -> float:
    """Time constant ``tau = R * C`` (s).

    After ``tau`` the capacitor has covered ``1 - 1/e`` (63.2 %) of the way to
    the source voltage.
    """
    return require_positive(resistance, "resistance") * require_positive(
        capacitance, "capacitance"
    )


def rc_voltage(
    source_voltage: float,
    resistance: float,
    capacitance: float,
    time: float,
    initial_voltage: float = 0.0,
) -> float:
    """Capacitor voltage ``V(t) = V_s + (V_0 - V_s) * exp(-t/tau)`` (V).

    Charging when ``|V_s| > |V_0|`` on the same side, discharging for
    ``source_voltage = 0``.

    Args:
        source_voltage: Source voltage ``V_s`` in V (signed).
        resistance: Resistance in ohm (> 0).
        capacitance: Capacitance in F (> 0).
        time: Time since the circuit was closed, in s (>= 0).
        initial_voltage: Capacitor voltage at ``t = 0`` in V.
    """
    vs = require_finite(source_voltage, "source_voltage")
    v0 = require_finite(initial_voltage, "initial_voltage")
    t = require_non_negative(time, "time")
    tau = rc_time_constant(resistance, capacitance)
    return vs + (v0 - vs) * math.exp(-t / tau)


def rc_discharge_voltage(
    initial_voltage: float,
    resistance: float,
    capacitance: float,
    time: float,
) -> float:
    """Voltage of a capacitor discharging through a resistor (V).

    ``V(t) = V_0 * exp(-t/tau)``; equivalent to :func:`rc_voltage` with
    ``source_voltage = 0``.
    """
    return rc_voltage(0.0, resistance, capacitance, time, initial_voltage)


def rc_current(
    source_voltage: float,
    resistance: float,
    capacitance: float,
    time: float,
    initial_voltage: float = 0.0,
) -> float:
    """Circuit current ``i(t) = (V_s - V_0)/R * exp(-t/tau)`` (A).

    Positive when current flows from the source into the capacitor (charging),
    negative when the capacitor discharges.
    """
    vs = require_finite(source_voltage, "source_voltage")
    v0 = require_finite(initial_voltage, "initial_voltage")
    t = require_non_negative(time, "time")
    r = require_positive(resistance, "resistance")
    tau = rc_time_constant(resistance, capacitance)
    return (vs - v0) / r * math.exp(-t / tau)


def rc_time_to_voltage(
    source_voltage: float,
    resistance: float,
    capacitance: float,
    target_voltage: float,
    initial_voltage: float = 0.0,
) -> float:
    """Time for the capacitor to reach ``target_voltage`` (s).

    ``t = -tau * ln((V_s - V_t)/(V_s - V_0))``.

    Raises:
        PhysicalInconsistencyError: If the target is never reached: the
            capacitor only approaches ``V_s`` asymptotically, so the target must
            lie between ``V_0`` (inclusive) and ``V_s`` (exclusive), and
            ``V_0 != V_s``.
    """
    vs = require_finite(source_voltage, "source_voltage")
    v0 = require_finite(initial_voltage, "initial_voltage")
    vt = require_finite(target_voltage, "target_voltage")
    tau = rc_time_constant(resistance, capacitance)
    if v0 == vs:
        raise PhysicalInconsistencyError(
            "The capacitor is already at the source voltage and does not evolve."
        )
    ratio = (vs - vt) / (vs - v0)
    if not 0.0 < ratio <= 1.0:
        raise PhysicalInconsistencyError(
            f"The target {vt} V is never reached: V(t) stays between {v0} V and "
            f"{vs} V (approaching {vs} V only asymptotically)."
        )
    return -tau * math.log(ratio) + 0.0
