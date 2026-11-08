import pytest

from scientificlab.physics.mechanics import (
    acceleration_from_force,
    centripetal_force,
    friction_force,
    momentum,
    net_force,
    weight,
)
from scientificlab.utils.exceptions import InvalidInputError


def test_newtons_second_law():
    assert net_force(mass=2, acceleration=3) == 6.0
    assert acceleration_from_force(force=12, mass=4) == 3.0


def test_second_law_round_trip():
    assert acceleration_from_force(net_force(7.5, -2.0), 7.5) == pytest.approx(-2.0)


def test_weight_of_one_kilogram_is_standard_gravity():
    assert weight(1) == pytest.approx(9.80665)


def test_weight_with_custom_gravity():
    assert weight(10, gravity=9.81) == pytest.approx(98.1)


def test_friction_force():
    assert friction_force(0.3, 100) == pytest.approx(30.0)
    assert friction_force(0.0, 100) == 0.0


def test_friction_on_horizontal_floor_uses_weight_as_normal_force():
    assert friction_force(0.5, weight(20)) == pytest.approx(0.5 * 20 * 9.80665)


def test_centripetal_force():
    assert centripetal_force(mass=2, speed=10, radius=5) == 40.0


def test_centripetal_force_matches_second_law():
    m, v, r = 3.0, 4.0, 2.0
    assert centripetal_force(m, v, r) == pytest.approx(net_force(m, v * v / r))


def test_momentum_is_signed():
    assert momentum(2, 3) == 6.0
    assert momentum(2, -3) == -6.0


@pytest.mark.parametrize(
    "call",
    [
        lambda: net_force(0, 1),
        lambda: net_force(-1, 1),
        lambda: acceleration_from_force(1, 0),
        lambda: weight(-5),
        lambda: weight(5, gravity=-1),
        lambda: friction_force(-0.1, 10),
        lambda: friction_force(0.1, -10),
        lambda: centripetal_force(1, 1, 0),
        lambda: momentum(float("nan"), 1),
        lambda: momentum(1, "fast"),
    ],
)
def test_invalid_inputs_raise(call):
    with pytest.raises(InvalidInputError):
        call()
