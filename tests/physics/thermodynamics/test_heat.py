import pytest

from scientificlab.physics.thermodynamics import (
    convert_temperature,
    heat_energy,
    latent_heat,
)
from scientificlab.utils.exceptions import (
    InvalidInputError,
    PhysicalInconsistencyError,
)


class TestHeat:
    def test_reference_example_water(self):
        assert (
            heat_energy(mass=2, specific_heat=4186, temperature_change=10) == 83_720.0
        )

    def test_cooling_releases_heat(self):
        assert heat_energy(2, 4186, -10) == -83_720.0

    def test_no_temperature_change_exchanges_no_heat(self):
        assert heat_energy(2, 4186, 0) == 0.0

    def test_melting_ice(self):
        # Latent heat of fusion of water ~ 334 kJ/kg.
        assert latent_heat(2, 334_000) == pytest.approx(668_000.0)

    def test_freezing_releases_heat(self):
        assert latent_heat(2, -334_000) == pytest.approx(-668_000.0)

    @pytest.mark.parametrize(
        "call",
        [
            lambda: heat_energy(0, 4186, 10),
            lambda: heat_energy(-1, 4186, 10),
            lambda: heat_energy(1, 0, 10),
            lambda: heat_energy(1, 4186, float("nan")),
            lambda: latent_heat(0, 334_000),
            lambda: latent_heat(1, float("inf")),
        ],
    )
    def test_invalid_inputs(self, call):
        with pytest.raises(InvalidInputError):
            call()


class TestTemperatureConversion:
    @pytest.mark.parametrize(
        ("value", "from_unit", "to_unit", "expected"),
        [
            (0, "C", "K", 273.15),
            (100, "C", "F", 212.0),
            (0, "C", "F", 32.0),
            (-40, "C", "F", -40.0),
            (-40, "F", "C", -40.0),
            (32, "F", "C", 0.0),
            (373.15, "K", "C", 100.0),
            (0, "K", "F", -459.67),
            (300, "K", "K", 300.0),
        ],
    )
    def test_reference_values(self, value, from_unit, to_unit, expected):
        assert convert_temperature(value, from_unit, to_unit) == pytest.approx(
            expected, abs=1e-9
        )

    @pytest.mark.parametrize("unit", ["K", "C", "F"])
    def test_round_trips(self, unit):
        other = {"K": "C", "C": "F", "F": "K"}[unit]
        value = 310.0 if unit == "K" else 36.6
        assert convert_temperature(
            convert_temperature(value, unit, other), other, unit
        ) == pytest.approx(value)

    def test_absolute_zero_is_allowed(self):
        assert convert_temperature(-273.15, "C", "K") == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize(
        ("value", "unit"), [(-0.1, "K"), (-273.16, "C"), (-459.68, "F")]
    )
    def test_below_absolute_zero_is_inconsistent(self, value, unit):
        with pytest.raises(PhysicalInconsistencyError, match="absolute zero"):
            convert_temperature(value, unit, "K")

    @pytest.mark.parametrize(
        ("from_unit", "to_unit"), [("X", "K"), ("K", "kelvin"), ("c", "K")]
    )
    def test_unknown_units(self, from_unit, to_unit):
        with pytest.raises(InvalidInputError):
            convert_temperature(10, from_unit, to_unit)

    def test_non_finite_value(self):
        with pytest.raises(InvalidInputError):
            convert_temperature(float("nan"), "C", "K")
