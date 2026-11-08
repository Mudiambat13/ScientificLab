"""Kinematics in SI units (m, s, m/s, m/s**2, rad, rad/s).

Conventions:
    * One-dimensional motion along a signed axis: positions, velocities and
      accelerations are signed scalars (positive = positive axis direction).
    * Speeds and distances are non-negative magnitudes.
    * Functions take and return plain floats expressed in SI base/derived units.
"""

import math
from collections.abc import Callable

from scientificlab.constants import STANDARD_GRAVITY
from scientificlab.utils.exceptions import InvalidInputError, PhysicalInconsistencyError
from scientificlab.utils.validation import (
    require_finite,
    require_non_negative,
    require_positive,
)


def distance_traveled(speed: float, time: float) -> float:
    """Distance covered at constant speed: ``d = v * t`` (m).

    Args:
        speed: Constant speed in m/s (non-negative).
        time: Duration in s (non-negative).
    """
    return require_non_negative(speed, "speed") * require_non_negative(time, "time")


def average_speed(distance: float, time: float) -> float:
    """Average speed ``distance / time`` (m/s).

    Args:
        distance: Total path length in m (non-negative).
        time: Elapsed time in s (strictly positive).
    """
    return require_non_negative(distance, "distance") / require_positive(time, "time")


def average_velocity(
    initial_position: float, final_position: float, time: float
) -> float:
    """Signed average velocity ``(x_f - x_i) / t`` (m/s).

    Args:
        initial_position: Position at the start, in m.
        final_position: Position at the end, in m.
        time: Elapsed time in s (strictly positive).
    """
    delta = require_finite(final_position, "final_position") - require_finite(
        initial_position, "initial_position"
    )
    return delta / require_positive(time, "time")


def instantaneous_velocity(
    position: Callable[[float], float],
    time: float,
    step: float = 1e-5,
) -> float:
    """Numerical derivative of a position function at ``time`` (m/s).

    Uses the central difference ``(x(t+h) - x(t-h)) / (2h)``, whose truncation
    error is O(h**2). Rounding error grows as ``eps / h`` (eps ~ 2.2e-16), so
    very small steps do not improve accuracy. The default ``h = 1e-5`` gives
    roughly 1e-9 to 1e-10 relative accuracy for smooth functions of order one.
    This is an approximation, not a symbolic derivative.

    Args:
        position: Function ``t -> x(t)`` returning metres for ``t`` in seconds.
        time: Instant at which the velocity is evaluated, in s.
        step: Finite-difference half-width ``h`` in s (strictly positive).

    Raises:
        InvalidInputError: If ``position`` is not callable, or arguments invalid.
    """
    if not callable(position):
        raise InvalidInputError("position must be a callable t -> x(t).")
    t = require_finite(time, "time")
    h = require_positive(step, "step")
    ahead = require_finite(position(t + h), "position(time + step)")
    behind = require_finite(position(t - h), "position(time - step)")
    return (ahead - behind) / (2.0 * h)


def acceleration(velocity_change: float, time: float) -> float:
    """Average acceleration ``delta_v / delta_t`` (m/s**2).

    Args:
        velocity_change: Signed change of velocity in m/s.
        time: Elapsed time in s (strictly positive).
    """
    return require_finite(velocity_change, "velocity_change") / require_positive(
        time, "time"
    )


def uniform_motion_position(
    initial_position: float, velocity: float, time: float
) -> float:
    """Position in uniform rectilinear motion: ``x = x0 + v*t`` (m)."""
    return require_finite(initial_position, "initial_position") + require_finite(
        velocity, "velocity"
    ) * require_non_negative(time, "time")


def accelerated_position(
    initial_position: float,
    initial_velocity: float,
    acceleration: float,
    time: float,
) -> float:
    """Position under constant acceleration: ``x = x0 + v0*t + a*t**2/2`` (m)."""
    x0 = require_finite(initial_position, "initial_position")
    v0 = require_finite(initial_velocity, "initial_velocity")
    a = require_finite(acceleration, "acceleration")
    t = require_non_negative(time, "time")
    return x0 + v0 * t + 0.5 * a * t * t


def accelerated_velocity(
    initial_velocity: float, acceleration: float, time: float
) -> float:
    """Velocity under constant acceleration: ``v = v0 + a*t`` (m/s)."""
    return require_finite(initial_velocity, "initial_velocity") + require_finite(
        acceleration, "acceleration"
    ) * require_non_negative(time, "time")


def speed_from_displacement(
    initial_velocity: float,
    acceleration: float,
    displacement: float,
) -> float:
    """Final **speed** (magnitude) from ``v**2 = v0**2 + 2*a*dx`` (m/s).

    The relation gives only ``v**2``, so the sign of the final velocity is not
    determined; this function returns ``abs(v)``.

    Raises:
        PhysicalInconsistencyError: If ``v0**2 + 2*a*dx < 0``, i.e. the body
            never reaches that displacement under this acceleration.
    """
    v0 = require_finite(initial_velocity, "initial_velocity")
    a = require_finite(acceleration, "acceleration")
    dx = require_finite(displacement, "displacement")
    v_squared = v0 * v0 + 2.0 * a * dx
    if v_squared < 0.0:
        raise PhysicalInconsistencyError(
            "v0**2 + 2*a*dx is negative: the displacement is never reached."
        )
    return math.sqrt(v_squared)


def free_fall_distance(time: float, gravity: float = STANDARD_GRAVITY) -> float:
    """Distance fallen from rest in vacuum: ``d = g*t**2/2`` (m).

    Assumes constant ``g`` and no air resistance.
    """
    t = require_non_negative(time, "time")
    return 0.5 * require_positive(gravity, "gravity") * t * t


def free_fall_velocity(
    time: float,
    initial_velocity: float = 0.0,
    gravity: float = STANDARD_GRAVITY,
) -> float:
    """Downward speed after ``time`` of free fall: ``v = v0 + g*t`` (m/s).

    ``initial_velocity`` is the initial **downward** speed (positive = down).
    """
    return require_finite(initial_velocity, "initial_velocity") + require_positive(
        gravity, "gravity"
    ) * require_non_negative(time, "time")


def free_fall_time(height: float, gravity: float = STANDARD_GRAVITY) -> float:
    """Time to fall a height ``h`` from rest: ``t = sqrt(2*h/g)`` (s)."""
    h = require_non_negative(height, "height")
    return math.sqrt(2.0 * h / require_positive(gravity, "gravity"))


def free_fall_impact_speed(height: float, gravity: float = STANDARD_GRAVITY) -> float:
    """Speed after falling ``h`` from rest: ``v = sqrt(2*g*h)`` (m/s)."""
    h = require_non_negative(height, "height")
    return math.sqrt(2.0 * require_positive(gravity, "gravity") * h)


def angular_velocity_from_period(period: float) -> float:
    """Angular velocity of uniform circular motion: ``omega = 2*pi/T`` (rad/s)."""
    return 2.0 * math.pi / require_positive(period, "period")


def tangential_speed(angular_velocity: float, radius: float) -> float:
    """Speed on a circle: ``v = omega * r`` (m/s), non-negative.

    ``angular_velocity`` is in rad/s; its sign (rotation sense) is discarded.
    """
    omega = require_finite(angular_velocity, "angular_velocity")
    return abs(omega) * require_positive(radius, "radius")


def centripetal_acceleration(speed: float, radius: float) -> float:
    """Centripetal acceleration ``a_c = v**2 / r`` (m/s**2), toward the centre."""
    v = require_finite(speed, "speed")
    return v * v / require_positive(radius, "radius")
