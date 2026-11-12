import math

import pytest

from scientificlab.constants import SPEED_OF_LIGHT
from scientificlab.physics.waves import (
    frequency_from_period,
    period_from_frequency,
    string_wave_speed,
    wave_displacement,
    wave_relation,
)
from scientificlab.utils.exceptions import InvalidInputError


class TestWaveRelation:
    def test_concert_pitch_in_air(self):
        wave = wave_relation(speed=343, frequency=440)
        assert wave.wavelength == pytest.approx(343 / 440)
        assert wave.period == pytest.approx(1 / 440)
        assert wave.angular_frequency == pytest.approx(2 * math.pi * 440)
        assert wave.wavenumber == pytest.approx(2 * math.pi * 440 / 343)

    def test_green_light_frequency(self):
        wave = wave_relation(speed=SPEED_OF_LIGHT, wavelength=500e-9)
        assert wave.frequency == pytest.approx(5.99584916e14)

    def test_each_quantity_can_be_the_unknown(self):
        assert wave_relation(frequency=10, wavelength=3).speed == pytest.approx(30.0)
        assert wave_relation(speed=30, wavelength=3).frequency == pytest.approx(10.0)
        assert wave_relation(speed=30, frequency=10).wavelength == pytest.approx(3.0)

    def test_properties_are_mutually_consistent(self):
        wave = wave_relation(speed=1500, frequency=2500)
        assert wave.speed == pytest.approx(wave.frequency * wave.wavelength)
        assert wave.angular_frequency / wave.wavenumber == pytest.approx(wave.speed)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"speed": 343},
            {"speed": 343, "frequency": 440, "wavelength": 0.78},
            {"speed": 0, "frequency": 440},
            {"speed": 343, "frequency": -1},
            {"speed": 343, "wavelength": float("nan")},
        ],
    )
    def test_invalid_arguments(self, kwargs):
        with pytest.raises(InvalidInputError):
            wave_relation(**kwargs)


def test_period_and_frequency_are_inverse():
    assert period_from_frequency(50.0) == pytest.approx(0.02)
    assert frequency_from_period(0.02) == pytest.approx(50.0)
    assert frequency_from_period(period_from_frequency(123.4)) == pytest.approx(123.4)
    with pytest.raises(InvalidInputError):
        period_from_frequency(0)
    with pytest.raises(InvalidInputError):
        frequency_from_period(-1)


def test_string_wave_speed():
    assert string_wave_speed(100, 0.01) == pytest.approx(100.0)
    assert string_wave_speed(400, 0.01) == pytest.approx(200.0)  # speed ~ sqrt(tension)
    with pytest.raises(InvalidInputError):
        string_wave_speed(0, 0.01)
    with pytest.raises(InvalidInputError):
        string_wave_speed(100, -0.01)


class TestDisplacement:
    def test_quarter_wavelength_is_a_crest(self):
        assert wave_displacement(2.0, 1.0, 1.0, 0.25, 0.0) == pytest.approx(2.0)

    def test_origin_at_time_zero(self):
        assert wave_displacement(2.0, 1.0, 1.0, 0.0, 0.0) == pytest.approx(
            0.0, abs=1e-15
        )

    def test_periodic_in_time_and_space(self):
        y = wave_displacement(1.5, 0.8, 0.02, 0.3, 0.007)
        assert wave_displacement(1.5, 0.8, 0.02, 0.3, 0.007 + 0.02) == pytest.approx(y)
        assert wave_displacement(1.5, 0.8, 0.02, 0.3 + 0.8, 0.007) == pytest.approx(y)

    def test_wave_travels_in_the_positive_direction_at_lambda_over_period(self):
        wavelength, period = 0.8, 0.02
        speed = wavelength / period
        y = wave_displacement(1.0, wavelength, period, 0.1, 0.0)
        moved = wave_displacement(1.0, wavelength, period, 0.1 + speed * 0.005, 0.005)
        assert moved == pytest.approx(y)

    def test_initial_phase_shifts_the_wave(self):
        assert wave_displacement(
            1.0, 1.0, 1.0, 0.0, 0.0, phase=math.pi / 2
        ) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "call",
        [
            lambda: wave_displacement(1.0, 0, 1.0, 0, 0),
            lambda: wave_displacement(1.0, 1.0, -1.0, 0, 0),
            lambda: wave_displacement(float("nan"), 1.0, 1.0, 0, 0),
            lambda: wave_displacement(1.0, 1.0, 1.0, float("inf"), 0),
        ],
    )
    def test_invalid_inputs(self, call):
        with pytest.raises(InvalidInputError):
            call()
