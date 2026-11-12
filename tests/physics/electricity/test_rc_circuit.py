import math

import pytest

from scientificlab.physics.electricity import (
    rc_current,
    rc_discharge_voltage,
    rc_time_constant,
    rc_time_to_voltage,
    rc_voltage,
)
from scientificlab.utils.exceptions import (
    InvalidInputError,
    PhysicalInconsistencyError,
)

R, C = 1_000.0, 1e-6  # tau = 1 ms


def test_time_constant():
    assert rc_time_constant(R, C) == pytest.approx(1e-3)


def test_charging_reference_values():
    tau = rc_time_constant(R, C)
    assert rc_voltage(5.0, R, C, 0.0) == pytest.approx(0.0)
    assert rc_voltage(5.0, R, C, tau) == pytest.approx(5.0 * 0.6321205588285577)
    assert rc_voltage(5.0, R, C, 5 * tau) == pytest.approx(5.0 * (1 - math.exp(-5)))


def test_fully_charged_after_many_time_constants():
    assert rc_voltage(5.0, R, C, 1.0) == pytest.approx(5.0, abs=1e-12)


def test_discharge_reference_values():
    tau = rc_time_constant(R, C)
    assert rc_discharge_voltage(5.0, R, C, tau) == pytest.approx(
        5.0 * 0.36787944117144233
    )
    assert rc_discharge_voltage(5.0, R, C, 0.0) == pytest.approx(5.0)


def test_discharge_is_charging_towards_zero():
    assert rc_discharge_voltage(5.0, R, C, 2e-3) == pytest.approx(
        rc_voltage(0.0, R, C, 2e-3, initial_voltage=5.0)
    )


def test_partially_charged_capacitor_continues_to_source_voltage():
    assert rc_voltage(10.0, R, C, 1e-3, initial_voltage=4.0) == pytest.approx(
        10.0 + (4.0 - 10.0) * math.exp(-1)
    )


def test_initial_current_is_source_over_resistance():
    assert rc_current(5.0, R, C, 0.0) == pytest.approx(5e-3)


def test_current_decays_exponentially_and_changes_sign_when_discharging():
    assert rc_current(5.0, R, C, 1e-3) == pytest.approx(5e-3 * math.exp(-1))
    assert rc_current(0.0, R, C, 0.0, initial_voltage=5.0) == pytest.approx(-5e-3)


def test_kirchhoff_voltage_law():
    # Source voltage = resistor drop + capacitor voltage at every instant.
    for t in (0.0, 3e-4, 1e-3, 4e-3):
        drop = rc_current(5.0, R, C, t) * R
        assert drop + rc_voltage(5.0, R, C, t) == pytest.approx(5.0)


def test_time_to_reach_ninety_percent():
    assert rc_time_to_voltage(5.0, R, C, 4.5) == pytest.approx(1e-3 * math.log(10))


def test_time_to_voltage_round_trip():
    t = rc_time_to_voltage(12.0, 470.0, 22e-6, 7.5, initial_voltage=1.0)
    assert rc_voltage(12.0, 470.0, 22e-6, t, initial_voltage=1.0) == pytest.approx(7.5)


def test_discharge_time():
    assert rc_time_to_voltage(0.0, R, C, 1.0, initial_voltage=5.0) == pytest.approx(
        1e-3 * math.log(5)
    )


def test_target_equal_to_initial_voltage_takes_no_time():
    assert rc_time_to_voltage(5.0, R, C, 2.0, initial_voltage=2.0) == 0.0


@pytest.mark.parametrize("target", [5.0, 5.1, -0.1])
def test_unreachable_target_voltages_are_inconsistent(target):
    with pytest.raises(PhysicalInconsistencyError):
        rc_time_to_voltage(5.0, R, C, target)


def test_target_before_the_initial_voltage_is_inconsistent():
    with pytest.raises(PhysicalInconsistencyError):
        rc_time_to_voltage(5.0, R, C, 1.0, initial_voltage=2.0)


def test_capacitor_already_at_source_voltage():
    with pytest.raises(PhysicalInconsistencyError):
        rc_time_to_voltage(5.0, R, C, 5.0, initial_voltage=5.0)


@pytest.mark.parametrize(
    "call",
    [
        lambda: rc_time_constant(0, C),
        lambda: rc_time_constant(R, -1e-6),
        lambda: rc_voltage(5, R, C, -1),
        lambda: rc_voltage(float("nan"), R, C, 1),
        lambda: rc_current(5, R, C, float("inf")),
        lambda: rc_discharge_voltage(5, R, 0, 1),
    ],
)
def test_invalid_inputs(call):
    with pytest.raises(InvalidInputError):
        call()
