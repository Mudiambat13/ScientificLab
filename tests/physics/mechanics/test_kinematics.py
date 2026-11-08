import math

import pytest

from scientificlab.constants import STANDARD_GRAVITY
from scientificlab.physics.mechanics import (
    accelerated_position,
    accelerated_velocity,
    acceleration,
    angular_velocity_from_period,
    average_speed,
    average_velocity,
    centripetal_acceleration,
    distance_traveled,
    free_fall_distance,
    free_fall_impact_speed,
    free_fall_time,
    free_fall_velocity,
    instantaneous_velocity,
    speed_from_displacement,
    tangential_speed,
    uniform_motion_position,
)
from scientificlab.utils.exceptions import (
    InvalidInputError,
    PhysicalInconsistencyError,
)


def test_standard_gravity_value():
    assert STANDARD_GRAVITY == 9.80665


def test_distance_and_average_speed():
    assert distance_traveled(speed=20, time=3) == 60.0
    assert average_speed(distance=100, time=10) == 10.0


def test_average_velocity_is_signed():
    assert average_velocity(10, 4, 2) == -3.0


def test_acceleration():
    assert acceleration(velocity_change=-6, time=3) == -2.0


def test_uniform_motion():
    assert uniform_motion_position(5, 2, 10) == 25.0


def test_uniformly_accelerated_motion():
    assert accelerated_position(0, 0, 2, 3) == 9.0
    assert accelerated_position(1, 2, -1, 4) == pytest.approx(1 + 8 - 8)
    assert accelerated_velocity(0, 2, 3) == 6.0


def test_speed_from_displacement_matches_velocity_time_relation():
    a, t = 2.0, 3.0
    dx = accelerated_position(0, 0, a, t)
    assert speed_from_displacement(0, a, dx) == pytest.approx(
        accelerated_velocity(0, a, t)
    )


def test_speed_from_displacement_unreachable_displacement():
    with pytest.raises(PhysicalInconsistencyError):
        speed_from_displacement(1, -1, 5)


def test_instantaneous_velocity_of_quadratic():
    assert instantaneous_velocity(lambda t: t * t, 3.0) == pytest.approx(6.0, rel=1e-6)


def test_instantaneous_velocity_of_sine():
    assert instantaneous_velocity(math.sin, 0.0) == pytest.approx(1.0, rel=1e-8)


def test_instantaneous_velocity_rejects_bad_arguments():
    with pytest.raises(InvalidInputError):
        instantaneous_velocity(5, 1.0)  # type: ignore[arg-type]
    with pytest.raises(InvalidInputError):
        instantaneous_velocity(lambda t: t, 1.0, step=0)
    with pytest.raises(InvalidInputError):
        instantaneous_velocity(lambda t: float("nan"), 1.0)


def test_free_fall_reference_values():
    # g/2 metres are fallen in exactly one second.
    assert free_fall_distance(1.0) == pytest.approx(STANDARD_GRAVITY / 2)
    assert free_fall_time(STANDARD_GRAVITY / 2) == pytest.approx(1.0)
    assert free_fall_velocity(1.0) == pytest.approx(STANDARD_GRAVITY)


def test_free_fall_round_trip():
    height = 20.0
    t = free_fall_time(height)
    assert free_fall_distance(t) == pytest.approx(height)
    assert free_fall_velocity(t) == pytest.approx(free_fall_impact_speed(height))


def test_free_fall_other_gravity():
    # Moon, g = 1.62 m/s**2: t = sqrt(2*10/1.62)
    assert free_fall_time(10, gravity=1.62) == pytest.approx(math.sqrt(20 / 1.62))


def test_circular_motion():
    omega = angular_velocity_from_period(2 * math.pi)
    assert omega == pytest.approx(1.0)
    assert tangential_speed(omega, 5) == pytest.approx(5.0)
    assert centripetal_acceleration(10, 5) == 20.0


def test_centripetal_acceleration_equals_omega_squared_r():
    omega, radius = 3.0, 0.5
    v = tangential_speed(omega, radius)
    assert centripetal_acceleration(v, radius) == pytest.approx(omega**2 * radius)


def test_tangential_speed_discards_rotation_sense():
    assert tangential_speed(-2.0, 3.0) == 6.0


@pytest.mark.parametrize(
    "call",
    [
        lambda: average_speed(10, 0),
        lambda: average_speed(-1, 5),
        lambda: acceleration(1, 0),
        lambda: distance_traveled(1, -1),
        lambda: free_fall_time(-1),
        lambda: free_fall_distance(1, gravity=0),
        lambda: angular_velocity_from_period(0),
        lambda: centripetal_acceleration(1, 0),
        lambda: accelerated_velocity(float("nan"), 1, 1),
        lambda: average_velocity(0, float("inf"), 1),
    ],
)
def test_invalid_inputs_raise(call):
    with pytest.raises(InvalidInputError):
        call()
