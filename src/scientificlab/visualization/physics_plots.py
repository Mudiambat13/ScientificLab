"""Ready-made physics plots built on the physics functions (SI units)."""

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from scientificlab.physics.electricity import rc_time_constant, rc_voltage
from scientificlab.physics.mechanics import accelerated_position
from scientificlab.physics.thermodynamics import ideal_gas
from scientificlab.utils.exceptions import InvalidInputError
from scientificlab.utils.validation import require_finite, require_positive
from scientificlab.visualization._common import finish, new_figure


def plot_rc_charging(
    source_voltage: float,
    resistance: float,
    capacitance: float,
    *,
    initial_voltage: float = 0.0,
    duration_in_tau: float = 5.0,
    points: int = 400,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Capacitor voltage ``V(t)`` of an RC circuit, with the time constant marked.

    Plots from 0 to ``duration_in_tau * tau`` using
    :func:`scientificlab.physics.electricity.rc_voltage`.

    Raises:
        InvalidInputError: For non-positive R, C or duration, or ``points < 2``.
    """
    tau = rc_time_constant(resistance, capacitance)
    span = require_positive(duration_in_tau, "duration_in_tau")
    if points < 2:
        raise InvalidInputError("points must be >= 2.")
    t = np.linspace(0.0, span * tau, points)
    v = [
        rc_voltage(source_voltage, resistance, capacitance, float(x), initial_voltage)
        for x in t
    ]
    fig, ax = new_figure(
        f"RC circuit (tau = {tau:.3g} s)", "time (s)", "capacitor voltage (V)"
    )
    ax.plot(t, v, label="V(t)")
    ax.axhline(source_voltage, ls="--", color="gray", label="source voltage")
    ax.axvline(tau, ls=":", color="gray", label="tau")
    return finish(fig, ax, save_path, show)


def plot_position_vs_time(
    initial_position: float,
    initial_velocity: float,
    acceleration: float,
    duration: float,
    *,
    points: int = 400,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Position of a body under constant acceleration over ``[0, duration]`` s.

    Uses :func:`scientificlab.physics.mechanics.accelerated_position`.

    Raises:
        InvalidInputError: If ``duration <= 0`` or ``points < 2``.
    """
    end = require_positive(duration, "duration")
    require_finite(acceleration, "acceleration")
    if points < 2:
        raise InvalidInputError("points must be >= 2.")
    t = np.linspace(0.0, end, points)
    x = [
        accelerated_position(initial_position, initial_velocity, acceleration, float(s))
        for s in t
    ]
    fig, ax = new_figure("Uniformly accelerated motion", "time (s)", "position (m)")
    ax.plot(t, x, label=f"a = {acceleration:g} m/s²")
    return finish(fig, ax, save_path, show)


def plot_isotherms(
    amount: float,
    temperatures: list[float],
    volume_range: tuple[float, float],
    *,
    points: int = 300,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Ideal-gas isotherms ``P(V)`` at the given absolute temperatures.

    Pressure comes from :func:`scientificlab.physics.thermodynamics.ideal_gas`.
    Temperatures are in kelvin (not Celsius); the ideal-gas model is invalid
    near condensation, which this plot does not show.

    Raises:
        InvalidInputError: For an empty temperature list, a non-positive
            amount/temperature/volume or an invalid range.
    """
    if not temperatures:
        raise InvalidInputError("At least one temperature is required.")
    lo = require_positive(volume_range[0], "volume_min")
    hi = require_positive(volume_range[1], "volume_max")
    if lo >= hi:
        raise InvalidInputError("volume_range must satisfy min < max.")
    if points < 2:
        raise InvalidInputError("points must be >= 2.")
    volumes = np.linspace(lo, hi, points)
    fig, ax = new_figure("Ideal-gas isotherms", "volume (m³)", "pressure (Pa)")
    for temperature in temperatures:
        pressures = [
            ideal_gas(volume=float(v), amount=amount, temperature=temperature).pressure
            for v in volumes
        ]
        ax.plot(volumes, pressures, label=f"T = {temperature:g} K")
    return finish(fig, ax, save_path, show)
