import pytest

from scientificlab.physics.mechanics import kinetic_energy as kinetic_energy_si
from scientificlab.units import Q_
from scientificlab.units import physics as up
from scientificlab.units.physics import with_units
from scientificlab.utils.exceptions import (
    IncompatibleUnitsError,
    InvalidInputError,
    UnknownUnitError,
)


def test_kinetic_energy_mixed_units_matches_si():
    # 2 g at 36 km/h = 0.002 kg at 10 m/s -> 0.5 * 0.002 * 100 = 0.1 J
    result = up.kinetic_energy(Q_(2, "g"), Q_(36, "km/h"))
    assert result.magnitude == pytest.approx(0.1, rel=1e-12)
    assert str(result.units) == "J"
    assert result.magnitude == pytest.approx(kinetic_energy_si(0.002, 10.0))


def test_plain_numbers_are_si():
    assert up.kinetic_energy(10, 5).magnitude == 125.0
    assert up.average_speed(distance=100, time=20).magnitude == 5.0


def test_result_can_be_converted():
    energy = up.kinetic_energy(Q_(1, "kg"), Q_(2, "m/s")).to("kJ")
    assert energy.magnitude == pytest.approx(0.002)


def test_wrong_dimension_is_refused():
    with pytest.raises(IncompatibleUnitsError, match="mass"):
        up.kinetic_energy(Q_(2, "s"), 3)
    with pytest.raises(IncompatibleUnitsError):
        up.average_speed(Q_(100, "m"), Q_(5, "m"))


def test_bad_argument_types():
    with pytest.raises(InvalidInputError):
        up.kinetic_energy("2 kg", 3)
    with pytest.raises(InvalidInputError):
        up.kinetic_energy(True, 3)


def test_physics_validation_still_applies():
    with pytest.raises(InvalidInputError):
        up.kinetic_energy(Q_(-1, "kg"), 3)


def test_free_fall_time_with_default_gravity():
    assert up.free_fall_time(Q_(100, "cm")).magnitude == pytest.approx(0.4515, rel=1e-3)


def test_heat_energy_temperature_difference():
    # Water: 1 kg, 4186 J/(kg K), +10 K -> 41 860 J
    assert up.heat_energy(1, 4186, Q_(10, "K")).magnitude == pytest.approx(41860.0)
    assert up.heat_energy(1, 4186, Q_(10, "delta_degC")).magnitude == pytest.approx(
        41860.0
    )
    assert up.heat_energy(Q_(1000, "g"), 4186, 10).magnitude == pytest.approx(41860.0)


def test_absolute_celsius_not_accepted_as_difference():
    with pytest.raises(IncompatibleUnitsError, match="temperature_change"):
        up.heat_energy(1, 4186, Q_(20, "degC"))


def test_with_units_decorator_validates_at_definition():
    with pytest.raises(UnknownUnitError):
        with_units({"x": "bogus"}, "m")
    with pytest.raises(ValueError, match="no parameter"):
        with_units({"nope": "m"}, "m")(lambda x: x)  # type: ignore[arg-type,return-value]


def test_decorated_function_keeps_metadata():
    assert up.kinetic_energy.__name__ == "kinetic_energy"
    assert "Unit-aware" in (up.kinetic_energy.__doc__ or "")
