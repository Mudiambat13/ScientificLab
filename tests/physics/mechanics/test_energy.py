import pytest

from scientificlab.physics.mechanics import (
    average_power,
    free_fall_impact_speed,
    kinetic_energy,
    mechanical_energy,
    potential_energy,
    power_from_force,
    work,
)
from scientificlab.utils.exceptions import InvalidInputError


def test_kinetic_energy_reference_example():
    assert kinetic_energy(mass=10, velocity=5) == 125.0  # 125 J


def test_kinetic_energy_ignores_velocity_sign():
    assert kinetic_energy(2, -3) == kinetic_energy(2, 3) == 9.0


def test_potential_energy():
    assert potential_energy(mass=2, height=10) == pytest.approx(2 * 9.80665 * 10)
    assert potential_energy(2, -1, gravity=10) == -20.0


def test_mechanical_energy():
    assert mechanical_energy(2, 3, 4) == pytest.approx(9 + 2 * 9.80665 * 4)


def test_energy_conservation_in_free_fall():
    mass, height = 3.0, 12.0
    speed = free_fall_impact_speed(height)
    assert kinetic_energy(mass, speed) == pytest.approx(potential_energy(mass, height))
    assert mechanical_energy(mass, speed, 0.0) == pytest.approx(
        mechanical_energy(mass, 0.0, height)
    )


def test_work_at_various_angles():
    assert work(10, 5) == 50.0
    assert work(10, 5, 60) == pytest.approx(25.0)
    assert work(10, 5, 90) == pytest.approx(0.0, abs=1e-12)
    assert work(10, 5, 180) == pytest.approx(-50.0)


def test_work_energy_theorem():
    # Constant force F on mass m over distance d from rest: W = delta(E_k).
    mass, force, distance = 4.0, 10.0, 8.0
    acceleration = force / mass
    final_speed = (2 * acceleration * distance) ** 0.5
    assert work(force, distance) == pytest.approx(kinetic_energy(mass, final_speed))


def test_power():
    assert average_power(energy_transferred=1000, time=10) == 100.0
    assert power_from_force(force=50, velocity=2) == 100.0
    assert power_from_force(50, 2, 60) == pytest.approx(50.0)


def test_power_matches_work_over_time():
    force, speed, duration = 20.0, 3.0, 5.0
    assert power_from_force(force, speed) == pytest.approx(
        average_power(work(force, speed * duration), duration)
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda: kinetic_energy(0, 5),
        lambda: kinetic_energy(-1, 5),
        lambda: kinetic_energy(1, float("inf")),
        lambda: potential_energy(1, float("nan")),
        lambda: potential_energy(1, 1, gravity=0),
        lambda: work(-1, 5),
        lambda: work(1, -5),
        lambda: work(1, 5, float("nan")),
        lambda: average_power(10, 0),
        lambda: power_from_force(1, -1),
    ],
)
def test_invalid_inputs_raise(call):
    with pytest.raises(InvalidInputError):
        call()
