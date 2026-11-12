"""Plots of simulation results (:mod:`scientificlab.simulations`)."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from scientificlab.simulations import SimulationResult, WaveResult
from scientificlab.utils.exceptions import InvalidInputError
from scientificlab.visualization._common import finish


def plot_simulation(
    result: SimulationResult,
    quantities: Sequence[str] | None = None,
    *,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """One panel per quantity against time, sharing the time axis.

    Args:
        result: Output of any ``simulate_*`` function returning a
            :class:`SimulationResult`.
        quantities: Series labels to draw (default: all, in order).
        save_path: Optional ``.png``, ``.svg`` or ``.pdf`` file.
        show: Open an interactive window.

    Raises:
        InvalidInputError: If a requested quantity does not exist or the list
            is empty.
    """
    keys = list(result.series) if quantities is None else list(quantities)
    if not keys:
        raise InvalidInputError("At least one quantity is required.")
    unknown = [k for k in keys if k not in result.series]
    if unknown:
        raise InvalidInputError(
            f"Unknown quantities {unknown}; available: {list(result.series)}."
        )
    fig = Figure(figsize=(7.0, 2.2 * len(keys) + 1.0), layout="constrained")
    axes = fig.subplots(len(keys), 1, sharex=True, squeeze=False)[:, 0]
    for ax, key in zip(axes, keys, strict=True):
        ax.plot(result.time, result.series[key])
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.3)
    axes[0].set_title(f"{result.name} ({result.method})")
    axes[-1].set_xlabel("time (s)")
    return finish(fig, axes[-1], save_path, show)


def plot_trajectory(
    result: SimulationResult,
    *,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Trajectory ``y`` against ``x`` of a projectile simulation (equal axes).

    Raises:
        InvalidInputError: If the result has no ``x (m)`` and ``y (m)`` series.
    """
    if "x (m)" not in result.series or "y (m)" not in result.series:
        raise InvalidInputError("The result has no x (m) / y (m) series.")
    fig = Figure(figsize=(7.0, 4.5), layout="constrained")
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(result.series["x (m)"], result.series["y (m)"])
    ax.set_title(f"{result.name} trajectory")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    return finish(fig, ax, save_path, show)


def plot_wave_snapshots(
    result: WaveResult,
    times: Sequence[float] | None = None,
    *,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Displacement profiles at chosen times (nearest stored time step).

    Args:
        result: Output of :func:`scientificlab.simulations.simulate_string_wave`.
        times: Times in s (default: 5 evenly spaced over the simulation).
        save_path: Optional ``.png``, ``.svg`` or ``.pdf`` file.
        show: Open an interactive window.

    Raises:
        InvalidInputError: If a time lies outside the simulated interval.
    """
    wanted = (
        np.linspace(0.0, result.time[-1], 5)
        if times is None
        else np.asarray(times, dtype=float)
    )
    if wanted.size == 0 or np.any(wanted < 0) or np.any(wanted > result.time[-1]):
        raise InvalidInputError(
            f"times must be a non-empty set within [0, {result.time[-1]:g}] s."
        )
    fig = Figure(figsize=(7.0, 4.5), layout="constrained")
    ax = fig.add_subplot(1, 1, 1)
    for value in wanted:
        k = int(np.argmin(np.abs(result.time - value)))
        ax.plot(result.x, result.displacement[k], label=f"t = {result.time[k]:.4g} s")
    ax.set_title("Wave on a string (fixed ends)")
    ax.set_xlabel("position (m)")
    ax.set_ylabel("displacement (m)")
    ax.grid(True, alpha=0.3)
    return finish(fig, ax, save_path, show)
