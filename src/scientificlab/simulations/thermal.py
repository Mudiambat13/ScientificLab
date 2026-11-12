"""Newton's law of cooling."""

import math

import numpy as np

from scientificlab.simulations._common import SimulationResult, sample_times
from scientificlab.utils.exceptions import InvalidInputError, NoSolutionError
from scientificlab.utils.validation import require_positive


def _check_kelvin(value: float, name: str) -> float:
    kelvin = require_positive(value, name)  # absolute temperatures must be > 0 K
    return kelvin


def simulate_cooling(
    initial_temperature: float,
    ambient_temperature: float,
    cooling_constant: float,
    *,
    duration: float | None = None,
    points: int = 500,
) -> SimulationResult:
    """Newton's law of cooling ``dT/dt = -k (T - T_amb)``, closed form.

    ``T(t) = T_amb + (T_0 - T_amb) exp(-k t)``. The law holds for a body with a
    roughly uniform temperature and a small temperature excess (convection
    dominated); it ignores radiation (which grows as ``T**4``) and phase
    changes. The body approaches the ambient temperature but never crosses it.

    Args:
        initial_temperature: ``T_0`` in K (> 0).
        ambient_temperature: ``T_amb`` in K (> 0), assumed constant.
        cooling_constant: ``k`` in 1/s (> 0).
        duration: Simulated time in s (> 0). Default: 5 time constants.
        points: Number of samples (>= 2).

    Returns:
        Series ``temperature (K)`` and summary ``time_constant`` (``1/k``),
        ``half_life`` (time to halve the excess, ``ln 2 / k``) and
        ``final_temperature``.

    Raises:
        InvalidInputError: For invalid parameters (including temperatures
            at or below absolute zero).
    """
    t0 = _check_kelvin(initial_temperature, "initial_temperature")
    ta = _check_kelvin(ambient_temperature, "ambient_temperature")
    k = require_positive(cooling_constant, "cooling_constant")
    span = 5.0 / k if duration is None else require_positive(duration, "duration")
    t = sample_times(span, points)
    temperature = ta + (t0 - ta) * np.exp(-k * t)
    return SimulationResult(
        "Cooling",
        "analytic",
        t,
        {"temperature (K)": temperature},
        {"initial_temperature": t0, "ambient_temperature": ta, "cooling_constant": k},
        {
            "time_constant": 1.0 / k,
            "half_life": math.log(2.0) / k,
            "final_temperature": float(temperature[-1]),
        },
        (
            "Newton's law: uniform body, constant ambient, no radiation or "
            "phase change.",
        ),
    )


def cooling_time_to_reach(
    initial_temperature: float,
    ambient_temperature: float,
    cooling_constant: float,
    target_temperature: float,
) -> float:
    """Time in s for the body to reach ``target_temperature`` (Newton's law).

    ``t = ln((T_0 - T_amb) / (T_target - T_amb)) / k``.

    Raises:
        NoSolutionError: If the target is never reached: it lies beyond the
            ambient temperature (or equals it), or on the wrong side of the
            initial temperature.
        InvalidInputError: For invalid parameters.
    """
    t0 = _check_kelvin(initial_temperature, "initial_temperature")
    ta = _check_kelvin(ambient_temperature, "ambient_temperature")
    k = require_positive(cooling_constant, "cooling_constant")
    target = _check_kelvin(target_temperature, "target_temperature")
    if t0 == ta:
        raise InvalidInputError("The body is already at ambient temperature.")
    ratio = (target - ta) / (t0 - ta)
    if ratio <= 0.0:
        raise NoSolutionError(
            "The ambient temperature is only approached asymptotically: a target "
            "at or beyond it is never reached."
        )
    if ratio > 1.0:
        raise NoSolutionError(
            "The target lies on the other side of the initial temperature: "
            "it was never reached after t = 0."
        )
    return -math.log(ratio) / k
