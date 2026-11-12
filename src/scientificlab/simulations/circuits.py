"""Series RC circuit: voltage, current and energy balance over time."""

import math

import numpy as np

from scientificlab.physics.electricity import rc_time_constant
from scientificlab.simulations._common import SimulationResult, sample_times
from scientificlab.utils.validation import require_finite, require_positive


def simulate_rc_circuit(
    source_voltage: float,
    resistance: float,
    capacitance: float,
    *,
    initial_voltage: float = 0.0,
    duration: float | None = None,
    points: int = 500,
) -> SimulationResult:
    """Series RC circuit connected to a constant source at ``t = 0``.

    Closed-form solution ``V(t) = V_s + (V_0 - V_s) exp(-t / tau)`` with
    ``tau = R C``; current ``I = (V_s - V) / R``.

    Args:
        source_voltage: Source voltage ``V_s`` in V (0 = discharge).
        resistance: Resistance in ohm (> 0).
        capacitance: Capacitance in F (> 0).
        initial_voltage: Capacitor voltage at ``t = 0`` in V.
        duration: Simulated time in s (> 0). Default: ``5 tau``.
        points: Number of samples (>= 2).

    Returns:
        Series ``voltage (V)`` (across the capacitor), ``current (A)`` (positive
        from the source into the capacitor), ``stored energy (J)``,
        ``dissipated energy (J)`` (in R, closed form) and
        ``supplied energy (J)`` (by the source). Energy conservation
        ``supplied = stored change + dissipated`` holds to rounding error.
        Summary: ``tau`` and ``final_voltage`` (steady state ``V_s``).

    Raises:
        InvalidInputError: For invalid parameters.
    """
    vs = require_finite(source_voltage, "source_voltage")
    v0 = require_finite(initial_voltage, "initial_voltage")
    tau = rc_time_constant(resistance, capacitance)
    r = require_positive(resistance, "resistance")
    cap = require_positive(capacitance, "capacitance")
    span = 5.0 * tau if duration is None else require_positive(duration, "duration")
    t = sample_times(span, points)
    decay = np.exp(-t / tau)
    v = vs + (v0 - vs) * decay
    i = (vs - v0) / r * decay
    stored = 0.5 * cap * v * v
    dissipated = 0.5 * cap * (vs - v0) ** 2 * (1.0 - decay * decay)
    supplied = vs * cap * (v - v0)
    return SimulationResult(
        "RC circuit",
        "analytic",
        t,
        {
            "voltage (V)": v,
            "current (A)": i,
            "stored energy (J)": stored,
            "dissipated energy (J)": dissipated,
            "supplied energy (J)": supplied,
        },
        {
            "source_voltage": vs,
            "resistance": r,
            "capacitance": cap,
            "initial_voltage": v0,
        },
        {"tau": tau, "final_voltage": vs, "time_to_99_percent": tau * math.log(100.0)},
        ("Ideal source, ideal resistor and capacitor; no leakage or ESR.",),
    )
