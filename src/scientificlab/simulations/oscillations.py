"""Damped harmonic oscillator (mass-spring-damper), closed-form solution."""

import math

import numpy as np

from scientificlab.simulations._common import SimulationResult, sample_times
from scientificlab.utils.validation import (
    require_finite,
    require_non_negative,
    require_positive,
)

_CRITICAL_TOL = 1e-9


def simulate_harmonic_oscillator(
    mass: float,
    stiffness: float,
    *,
    damping: float = 0.0,
    initial_position: float = 1.0,
    initial_velocity: float = 0.0,
    duration: float | None = None,
    points: int = 1000,
) -> SimulationResult:
    """Solve ``m x'' + c x' + k x = 0`` exactly for given initial conditions.

    The regime follows the damping ratio ``zeta = c / (2 sqrt(k m))``:
    undamped (``c = 0``), underdamped (``zeta < 1``), critically damped
    (``|zeta - 1| < 1e-9``) or overdamped (``zeta > 1``).

    Args:
        mass: Mass in kg (> 0).
        stiffness: Spring constant in N/m (> 0).
        damping: Damping coefficient ``c`` in kg/s (>= 0).
        initial_position: ``x(0)`` in m.
        initial_velocity: ``x'(0)`` in m/s.
        duration: Simulated time in s (> 0). Default: 5 periods of the
            undamped oscillation ``2 pi / omega_0``.
        points: Number of samples (>= 2).

    Returns:
        Series ``x (m)``, ``v (m/s)``, ``energy (J)`` (kinetic + elastic) and
        summary ``omega0``, ``damping_ratio``, ``regime`` and, for the
        oscillating regimes, ``period``. Energy never increases for
        ``damping >= 0``.

    Raises:
        InvalidInputError: For invalid parameters.
    """
    m = require_positive(mass, "mass")
    k = require_positive(stiffness, "stiffness")
    c = require_non_negative(damping, "damping")
    x0 = require_finite(initial_position, "initial_position")
    v0 = require_finite(initial_velocity, "initial_velocity")
    w0 = math.sqrt(k / m)
    zeta = c / (2.0 * math.sqrt(k * m))
    span = 5.0 * 2.0 * math.pi / w0 if duration is None else duration
    if duration is not None:
        span = require_positive(duration, "duration")
    t = sample_times(span, points)
    summary: dict[str, float | str] = {"omega0": w0, "damping_ratio": zeta}

    if zeta < 1.0 - _CRITICAL_TOL:
        wd = w0 * math.sqrt(1.0 - zeta * zeta)
        decay = zeta * w0
        b = (v0 + decay * x0) / wd
        env = np.exp(-decay * t)
        cos, sin = np.cos(wd * t), np.sin(wd * t)
        x = env * (x0 * cos + b * sin)
        v = env * ((-decay * x0 + b * wd) * cos + (-decay * b - wd * x0) * sin)
        summary["regime"] = "undamped" if c == 0.0 else "underdamped"
        summary["period"] = 2.0 * math.pi / wd
        summary["damped_omega"] = wd
    elif zeta <= 1.0 + _CRITICAL_TOL:
        b = v0 + w0 * x0
        env = np.exp(-w0 * t)
        x = (x0 + b * t) * env
        v = (b - w0 * (x0 + b * t)) * env
        summary["regime"] = "critically damped"
    else:
        root = w0 * math.sqrt(zeta * zeta - 1.0)
        r1, r2 = -zeta * w0 + root, -zeta * w0 - root
        c1 = (v0 - r2 * x0) / (r1 - r2)
        c2 = x0 - c1
        x = c1 * np.exp(r1 * t) + c2 * np.exp(r2 * t)
        v = c1 * r1 * np.exp(r1 * t) + c2 * r2 * np.exp(r2 * t)
        summary["regime"] = "overdamped"

    energy = 0.5 * m * v * v + 0.5 * k * x * x
    summary["initial_energy"] = float(energy[0])
    summary["final_energy"] = float(energy[-1])
    return SimulationResult(
        "Harmonic oscillator",
        "analytic",
        t,
        {"x (m)": x, "v (m/s)": v, "energy (J)": energy},
        {
            "mass": m,
            "stiffness": k,
            "damping": c,
            "initial_position": x0,
            "initial_velocity": v0,
        },
        summary,
        ("Linear spring and viscous (linear) damping; small-amplitude model.",),
    )
