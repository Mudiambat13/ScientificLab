"""Projectile motion and free fall, with optional quadratic air drag.

Model: point mass in a uniform gravitational field ``g`` (flat ground at
``y = 0``, no wind, no spin, no Earth rotation). Optional drag is quadratic in
speed, ``a_drag = -c * |v| * v`` with ``c = rho * C_d * A / (2 * m)`` in 1/m
(``drag``). ``drag = 0`` is the vacuum case, solved in closed form.
"""

import math

import numpy as np
from scipy.integrate import solve_ivp

from scientificlab.constants import STANDARD_GRAVITY
from scientificlab.simulations._common import SimulationResult, sample_times
from scientificlab.utils.exceptions import InvalidInputError, SolverError
from scientificlab.utils.validation import (
    require_finite,
    require_non_negative,
    require_positive,
)

_RTOL = 1e-10
_ATOL = 1e-10


def simulate_projectile(
    speed: float,
    angle_degrees: float,
    *,
    height: float = 0.0,
    gravity: float = STANDARD_GRAVITY,
    drag: float = 0.0,
    points: int = 500,
) -> SimulationResult:
    """Trajectory of a projectile launched from height ``height``.

    Args:
        speed: Launch speed in m/s (> 0).
        angle_degrees: Elevation above the horizontal in degrees, ``0 <= a <= 90``.
        height: Launch height above the ground in m (>= 0). Must be > 0 when
            ``angle_degrees == 0`` (otherwise the body never leaves the ground).
        gravity: Gravitational acceleration in m/s**2 (> 0).
        drag: Quadratic drag coefficient ``c`` in 1/m (>= 0); 0 = vacuum.
        points: Number of samples between launch and impact (>= 2).

    Returns:
        A :class:`SimulationResult` with series ``x (m)``, ``y (m)``,
        ``vx (m/s)``, ``vy (m/s)`` and summary ``flight_time``, ``range``,
        ``max_height``, ``impact_speed``. Without drag the solution is exact
        (``method = "analytic"``); with drag it is integrated with RK45
        (relative tolerance 1e-10) up to ground impact.

    Raises:
        InvalidInputError: For invalid parameters.
        SolverError: If the integration fails (not expected for valid input).
    """
    v0 = require_positive(speed, "speed")
    angle = require_finite(angle_degrees, "angle_degrees")
    if not 0.0 <= angle <= 90.0:
        raise InvalidInputError("angle_degrees must be between 0 and 90.")
    h0 = require_non_negative(height, "height")
    g = require_positive(gravity, "gravity")
    c = require_non_negative(drag, "drag")
    if angle == 0.0 and h0 == 0.0:
        raise InvalidInputError(
            "A horizontal launch from the ground never leaves the ground; "
            "give height > 0 or angle_degrees > 0."
        )
    theta = math.radians(angle)
    vx0, vy0 = v0 * math.cos(theta), v0 * math.sin(theta)
    if angle == 90.0:
        vx0 = 0.0

    if c == 0.0:
        t_end = (vy0 + math.sqrt(vy0 * vy0 + 2.0 * g * h0)) / g
        t = sample_times(t_end, points)
        x = vx0 * t
        y = h0 + vy0 * t - 0.5 * g * t * t
        y[-1] = 0.0  # exact impact
        vx = np.full_like(t, vx0)
        vy = vy0 - g * t
        max_height = h0 + (vy0 * vy0 / (2.0 * g) if vy0 > 0 else 0.0)
        method = "analytic"
        notes = ("Vacuum: no air resistance.",)
    else:

        def rhs(_t: float, s: np.ndarray) -> list[float]:
            speed_now = math.hypot(s[2], s[3])
            return [s[2], s[3], -c * speed_now * s[2], -g - c * speed_now * s[3]]

        def ground(_t: float, s: np.ndarray) -> float:
            return float(s[1])

        ground.terminal = True  # type: ignore[attr-defined]
        ground.direction = -1  # type: ignore[attr-defined]

        def apex(_t: float, s: np.ndarray) -> float:
            return float(s[3])

        apex.direction = -1  # type: ignore[attr-defined]

        sol = solve_ivp(
            rhs,
            (0.0, 1e6),
            [0.0, h0, vx0, vy0],
            events=(ground, apex),
            dense_output=True,
            rtol=_RTOL,
            atol=_ATOL,
        )
        if sol.t_events[0].size == 0:
            raise SolverError("Integration did not reach the ground.")
        t_end = float(sol.t_events[0][0])
        t = sample_times(t_end, points)
        x, y, vx, vy = sol.sol(t)
        y[-1] = 0.0
        max_height = float(sol.y_events[1][0][1]) if sol.t_events[1].size else h0
        method = "numeric (RK45, rtol=1e-10)"
        notes = (
            "Quadratic air drag a = -c*|v|*v with constant c; no wind, "
            "constant air density.",
        )

    summary: dict[str, float | str] = {
        "flight_time": float(t_end),
        "range": float(x[-1]),
        "max_height": float(max_height),
        "impact_speed": float(math.hypot(vx[-1], vy[-1])),
    }
    return SimulationResult(
        "Projectile",
        method,
        t,
        {"x (m)": x, "y (m)": y, "vx (m/s)": vx, "vy (m/s)": vy},
        {
            "speed": v0,
            "angle_degrees": angle,
            "height": h0,
            "gravity": g,
            "drag": c,
        },
        summary,
        notes,
    )


def simulate_free_fall(
    height: float,
    *,
    gravity: float = STANDARD_GRAVITY,
    drag: float = 0.0,
    points: int = 500,
) -> SimulationResult:
    """Fall from rest from ``height`` metres, with optional quadratic drag.

    Both cases use closed-form solutions. With ``c = drag > 0`` the terminal
    speed is ``v_t = sqrt(g / c)`` and ``v(t) = v_t * tanh(g t / v_t)``,
    ``y(t) = h - (v_t**2 / g) * ln(cosh(g t / v_t))``.

    Args:
        height: Initial height in m (> 0).
        gravity: Gravitational acceleration in m/s**2 (> 0).
        drag: Quadratic drag coefficient in 1/m (>= 0).
        points: Samples between release and impact (>= 2).

    Returns:
        Series ``y (m)`` (height above ground), ``v (m/s)`` (downward speed,
        positive) and summary ``fall_time``, ``impact_speed`` and, with drag,
        ``terminal_speed``.

    Raises:
        InvalidInputError: For invalid parameters.
    """
    h = require_positive(height, "height")
    g = require_positive(gravity, "gravity")
    c = require_non_negative(drag, "drag")
    summary: dict[str, float | str]
    if c == 0.0:
        t_end = math.sqrt(2.0 * h / g)
        t = sample_times(t_end, points)
        y = h - 0.5 * g * t * t
        v = g * t
        summary = {"fall_time": t_end, "impact_speed": g * t_end}
        notes: tuple[str, ...] = ("Vacuum: no air resistance.",)
    else:
        vt = math.sqrt(g / c)
        a = g * h / (vt * vt)  # = c * h, dimensionless
        # arccosh(exp(a)) = a + ln(1 + sqrt(1 - exp(-2a))), stable for large a.
        t_end = (vt / g) * (a + math.log1p(math.sqrt(-math.expm1(-2.0 * a))))
        t = sample_times(t_end, points)
        arg = g * t / vt
        v = vt * np.tanh(arg)
        # ln(cosh(u)) = u + ln1p(exp(-2u)) - ln 2, stable for large u.
        log_cosh = arg + np.log1p(np.exp(-2.0 * arg)) - math.log(2.0)
        y = h - (vt * vt / g) * log_cosh
        y[-1] = 0.0
        summary = {
            "fall_time": t_end,
            "impact_speed": float(v[-1]),
            "terminal_speed": vt,
        }
        notes = (
            "Quadratic air drag a = -c*v**2 with constant c; the body approaches "
            "the terminal speed but never exceeds it.",
        )
    return SimulationResult(
        "Free fall",
        "analytic",
        t,
        {"y (m)": y, "v (m/s)": v},
        {"height": h, "gravity": g, "drag": c},
        summary,
        notes,
    )
