"""Result containers shared by all simulations."""

from dataclasses import dataclass, field

import numpy as np

from scientificlab.utils.exceptions import InvalidInputError


@dataclass(frozen=True)
class SimulationResult:
    """Time series produced by a simulation.

    Attributes:
        name: Human-readable simulation name.
        method: How the numbers were obtained: ``"analytic"`` (closed-form
            solution evaluated at the sample times) or ``"numeric (...)"``.
        time: Sample times in s, strictly increasing, starting at 0.
        series: Quantity label (with unit, e.g. ``"x (m)"``) -> values, each the
            same length as ``time``.
        parameters: The validated input parameters (SI units).
        summary: Derived scalar results (range, period, time constant, ...).
        notes: Model assumptions and caveats that apply to this result.
    """

    name: str
    method: str
    time: np.ndarray
    series: dict[str, np.ndarray]
    parameters: dict[str, float]
    summary: dict[str, float | str]
    notes: tuple[str, ...] = field(default=())


@dataclass(frozen=True)
class WaveResult:
    """Space-time field produced by :func:`simulate_string_wave`.

    Attributes:
        x: Grid positions in m, shape ``(nx,)``.
        time: Times in s, shape ``(nt,)``.
        displacement: Displacement ``u[k, i]`` at ``time[k]``, ``x[i]``, in m.
        energy: Total energy per unit linear density (J/(kg/m)) at the half
            steps ``energy_time`` (one value per time step).
        energy_time: Times in s at which ``energy`` is defined, ``t_k + dt/2``.
        method: Numerical scheme used.
        parameters: Validated input parameters (SI units).
        summary: Derived scalar results (Courant number, steps, ...).
        notes: Model assumptions and caveats.
    """

    x: np.ndarray
    time: np.ndarray
    displacement: np.ndarray
    energy: np.ndarray
    energy_time: np.ndarray
    method: str
    parameters: dict[str, float]
    summary: dict[str, float | str]
    notes: tuple[str, ...] = field(default=())


def sample_times(duration: float, points: int) -> np.ndarray:
    """``points`` evenly spaced times in ``[0, duration]``.

    Raises:
        InvalidInputError: If ``points`` is not an integer >= 2.
    """
    if isinstance(points, bool) or not isinstance(points, int) or points < 2:
        raise InvalidInputError("points must be an integer >= 2.")
    return np.linspace(0.0, duration, points)
