"""One-dimensional wave propagation on a string with fixed ends."""

from collections.abc import Callable

import numpy as np

from scientificlab.simulations._common import WaveResult
from scientificlab.utils.exceptions import InvalidInputError
from scientificlab.utils.validation import require_positive

_MAX_CELLS = 5_000_000
_END_TOL = 1e-6

Field = Callable[[np.ndarray], np.ndarray]


def simulate_string_wave(
    length: float,
    wave_speed: float,
    initial_displacement: Field,
    *,
    initial_velocity: Field | None = None,
    duration: float,
    nx: int = 201,
    courant: float = 0.9,
) -> WaveResult:
    """Solve ``u_tt = c**2 u_xx`` on ``[0, L]`` with fixed ends ``u(0) = u(L) = 0``.

    Scheme: explicit second-order finite differences (leapfrog), with a
    second-order start-up step. It is **conditionally stable**: the Courant
    number ``c dt / dx`` must be <= 1 (checked). At 1 the scheme is exact for
    a string released from rest (displacement-only data); with an initial
    velocity, or below 1, it is second-order accurate with slight numerical
    dispersion (short wavelengths travel slightly slower).

    Args:
        length: String length ``L`` in m (> 0).
        wave_speed: Wave speed ``c`` in m/s (> 0).
        initial_displacement: Function ``u0(x)`` taking an array of positions
            (m) and returning displacements (m). Must vanish at both ends.
        initial_velocity: Function ``v0(x)`` (m/s) or ``None`` for a string at
            rest. Must vanish at both ends.
        duration: Simulated time in s (> 0). The last stored time is the
            largest whole number of time steps not exceeding it.
        nx: Number of grid points (>= 3).
        courant: Courant number in ``(0, 1]``.

    Returns:
        A :class:`WaveResult` with ``x``, ``time``, ``displacement[k, i]`` and
        the discrete energy per unit linear density at the half steps
        ``energy_time``, conserved up to discretisation error.

    Raises:
        InvalidInputError: For invalid parameters, an unstable Courant number,
            initial data not vanishing at the ends, non-finite initial data, or
            a grid too large (> 5 million cells).
    """
    length = require_positive(length, "length")
    c = require_positive(wave_speed, "wave_speed")
    span = require_positive(duration, "duration")
    if isinstance(nx, bool) or not isinstance(nx, int) or nx < 3:
        raise InvalidInputError("nx must be an integer >= 3.")
    if not 0.0 < courant <= 1.0:
        raise InvalidInputError(
            f"courant must be in (0, 1] for a stable scheme, got {courant!r}."
        )
    x = np.linspace(0.0, length, nx)
    dx = length / (nx - 1)
    dt = courant * dx / c
    steps = int(span / dt + 1e-9)  # tolerance guards against 124.999999 -> 124
    if steps < 1:
        raise InvalidInputError("duration is shorter than one time step.")
    if (steps + 1) * nx > _MAX_CELLS:
        raise InvalidInputError(
            "Grid too large; reduce nx or duration (limit 5 million values)."
        )

    u0 = _field(initial_displacement, x, "initial_displacement")
    v0 = (
        np.zeros_like(x)
        if initial_velocity is None
        else _field(initial_velocity, x, "initial_velocity")
    )
    scale = max(1.0, float(np.max(np.abs(u0))))
    if max(abs(u0[0]), abs(u0[-1]), abs(v0[0]), abs(v0[-1])) > _END_TOL * scale:
        raise InvalidInputError(
            "Initial data must vanish at both ends (fixed-end string)."
        )

    # Ends are fixed: ends within tolerance of 0 are set exactly to 0.
    u0[[0, -1]] = 0.0
    v0[[0, -1]] = 0.0
    r2 = (c * dt / dx) ** 2
    u = np.zeros((steps + 1, nx))
    u[0] = u0
    u[1 if steps >= 1 else 0, 1:-1] = (
        u0[1:-1] + dt * v0[1:-1] + 0.5 * r2 * (u0[2:] - 2.0 * u0[1:-1] + u0[:-2])
    )
    for k in range(1, steps):
        u[k + 1, 1:-1] = (
            2.0 * u[k, 1:-1]
            - u[k - 1, 1:-1]
            + r2 * (u[k, 2:] - 2.0 * u[k, 1:-1] + u[k, :-2])
        )
    time = dt * np.arange(steps + 1)
    energy = _energy(u, dt, dx, c)
    return WaveResult(
        x,
        time,
        u,
        energy,
        time[:-1] + 0.5 * dt,
        "finite differences (leapfrog, 2nd order)",
        {
            "length": length,
            "wave_speed": c,
            "duration": float(time[-1]),
            "nx": float(nx),
            "courant": courant,
        },
        {"dx": dx, "dt": dt, "steps": float(steps), "courant": courant},
        (
            "Ideal string: linear, no damping, fixed ends (waves reflect with a "
            "sign flip).",
            "Courant < 1 introduces small numerical dispersion.",
        ),
    )


def _field(function: Field, x: np.ndarray, name: str) -> np.ndarray:
    if not callable(function):
        raise InvalidInputError(f"{name} must be a function of x.")
    try:
        values = np.asarray(function(x), dtype=float)
    except (TypeError, ValueError, ArithmeticError) as exc:
        raise InvalidInputError(f"{name} could not be evaluated: {exc}") from exc
    if values.shape != x.shape:
        raise InvalidInputError(f"{name} must return an array shaped like x.")
    if not np.all(np.isfinite(values)):
        raise InvalidInputError(f"{name} must be finite everywhere.")
    return values


def _energy(u: np.ndarray, dt: float, dx: float, c: float) -> np.ndarray:
    """Discrete energy per unit density at the half steps ``t_k + dt/2``.

    ``E = sum(((u_{k+1} - u_k) / dt)**2) * dx / 2
    + c**2 * sum(u_x(k) * u_x(k+1)) * dx / 2``: this time-staggered form is
    conserved up to rounding by the leapfrog scheme with fixed ends (apart from
    the start-up step), whereas the naive form oscillates slightly.
    """
    velocity = (u[1:] - u[:-1]) / dt
    gradient = np.diff(u, axis=1) / dx
    kinetic = 0.5 * np.sum(velocity**2, axis=1) * dx
    potential = 0.5 * c * c * np.sum(gradient[:-1] * gradient[1:], axis=1) * dx
    return np.asarray(kinetic + potential)
