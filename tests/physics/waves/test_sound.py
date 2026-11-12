import math

import pytest

from scientificlab.physics.waves import (
    doppler_frequency,
    intensity_from_level,
    sound_intensity_level,
    sound_speed_ideal_gas,
    sound_speed_in_air,
)
from scientificlab.utils.exceptions import (
    InvalidInputError,
    PhysicalInconsistencyError,
)


class TestSoundSpeed:
    def test_air_at_zero_celsius(self):
        assert sound_speed_in_air(0.0) == pytest.approx(331.32, abs=0.01)

    def test_air_at_room_temperature_matches_textbook_value(self):
        assert sound_speed_in_air(20.0) == pytest.approx(343.2, rel=1e-3)

    def test_speed_increases_with_temperature(self):
        assert sound_speed_in_air(30.0) > sound_speed_in_air(10.0)

    def test_helium_is_about_three_times_faster_than_air(self):
        helium = sound_speed_ideal_gas(5 / 3, 293.15, 0.0040026)
        assert helium == pytest.approx(1007.0, rel=0.01)
        assert helium > 2.5 * sound_speed_in_air(20.0)

    def test_speed_scales_with_the_square_root_of_temperature(self):
        ratio = sound_speed_ideal_gas(1.4, 1200.0, 0.029) / sound_speed_ideal_gas(
            1.4, 300.0, 0.029
        )
        assert ratio == pytest.approx(2.0)

    def test_below_absolute_zero_is_inconsistent(self):
        with pytest.raises(PhysicalInconsistencyError):
            sound_speed_in_air(-300.0)

    @pytest.mark.parametrize(
        "call",
        [
            lambda: sound_speed_ideal_gas(0, 300, 0.029),
            lambda: sound_speed_ideal_gas(1.4, -5, 0.029),
            lambda: sound_speed_ideal_gas(1.4, 300, 0),
            lambda: sound_speed_in_air(float("nan")),
        ],
    )
    def test_invalid_inputs(self, call):
        with pytest.raises(InvalidInputError):
            call()


class TestIntensityLevel:
    def test_reference_values(self):
        assert sound_intensity_level(1e-12) == pytest.approx(0.0, abs=1e-12)
        assert sound_intensity_level(1e-6) == pytest.approx(60.0)
        assert sound_intensity_level(1.0) == pytest.approx(120.0)

    def test_doubling_the_intensity_adds_three_decibels(self):
        delta = sound_intensity_level(2e-6) - sound_intensity_level(1e-6)
        assert delta == pytest.approx(10 * math.log10(2))
        assert delta == pytest.approx(3.0103, abs=1e-4)

    def test_round_trip(self):
        assert intensity_from_level(sound_intensity_level(3.7e-5)) == pytest.approx(
            3.7e-5
        )
        assert intensity_from_level(0.0) == pytest.approx(1e-12)
        assert intensity_from_level(60.0) == pytest.approx(1e-6)

    def test_negative_levels_are_below_the_reference(self):
        assert intensity_from_level(-10.0) == pytest.approx(1e-13)

    @pytest.mark.parametrize("intensity", [0, -1e-6, float("nan")])
    def test_invalid_intensity(self, intensity):
        with pytest.raises(InvalidInputError):
            sound_intensity_level(intensity)

    def test_invalid_level(self):
        with pytest.raises(InvalidInputError):
            intensity_from_level(float("nan"))

    def test_overflowing_level_is_reported(self):
        with pytest.raises(PhysicalInconsistencyError):
            intensity_from_level(1e4)


class TestDoppler:
    def test_approaching_source(self):
        assert doppler_frequency(440, 343, source_velocity=30) == pytest.approx(
            482.17252396
        )

    def test_approaching_observer(self):
        assert doppler_frequency(440, 343, observer_velocity=30) == pytest.approx(
            478.48396501
        )

    def test_receding_source(self):
        assert doppler_frequency(440, 343, source_velocity=-30) == pytest.approx(
            404.61126005
        )

    def test_no_relative_motion_leaves_the_frequency_unchanged(self):
        assert doppler_frequency(440, 343) == pytest.approx(440.0)

    def test_source_and_observer_moving_together_with_equal_velocity(self):
        # Moving toward each other at 10 m/s each.
        assert doppler_frequency(440, 343, 10, 10) == pytest.approx(440 * 353 / 333)

    def test_source_and_observer_cases_differ_at_high_speed(self):
        assert doppler_frequency(440, 343, source_velocity=100) != pytest.approx(
            doppler_frequency(440, 343, observer_velocity=100)
        )

    @pytest.mark.parametrize("source_velocity", [343.0, 400.0])
    def test_supersonic_source_is_inconsistent(self, source_velocity):
        with pytest.raises(PhysicalInconsistencyError, match="shock"):
            doppler_frequency(440, 343, source_velocity=source_velocity)

    @pytest.mark.parametrize("observer_velocity", [-343.0, -500.0])
    def test_observer_outrunning_the_wave_is_inconsistent(self, observer_velocity):
        with pytest.raises(PhysicalInconsistencyError):
            doppler_frequency(440, 343, observer_velocity=observer_velocity)

    @pytest.mark.parametrize(
        "call",
        [
            lambda: doppler_frequency(0, 343),
            lambda: doppler_frequency(440, 0),
            lambda: doppler_frequency(440, 343, source_velocity=float("nan")),
            lambda: doppler_frequency(440, 343, observer_velocity=float("inf")),
        ],
    )
    def test_invalid_inputs(self, call):
        with pytest.raises(InvalidInputError):
            call()
