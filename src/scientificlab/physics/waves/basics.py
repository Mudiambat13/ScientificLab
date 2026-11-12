"""Wave fundamentals (SI units: Hz, s, m, m/s, rad).

Model: monochromatic, linear, non-dispersive waves (speed independent of
frequency), travelling in the +x direction.
"""

import math
from dataclasses import dataclass

from scientificlab.utils.validation import (
    require_finite,
    require_n_given,
    require_positive,
)


@dataclass(frozen=True)
class WaveProperties:
    """Consistent description of a sinusoidal wave.

    Attributes:
        speed: Propagation speed ``v`` in m/s.
        frequency: Frequency ``f`` in Hz.
        wavelength: Wavelength ``lambda`` in m.
        period: Period ``T = 1/f`` in s.
        angular_frequency: ``omega = 2*pi*f`` in rad/s.
        wavenumber: ``k = 2*pi/lambda`` in rad/m.
    """

    speed: float
    frequency: float
    wavelength: float
    period: float
    angular_frequency: float
    wavenumber: float


def period_from_frequency(frequency: float) -> float:
    """Period ``T = 1 / f`` (s)."""
    return 1.0 / require_positive(frequency, "frequency")


def frequency_from_period(period: float) -> float:
    """Frequency ``f = 1 / T`` (Hz)."""
    return 1.0 / require_positive(period, "period")


def wave_relation(
    speed: float | None = None,
    frequency: float | None = None,
    wavelength: float | None = None,
) -> WaveProperties:
    """Solve ``v = f * lambda`` for the missing quantity.

    Exactly two of the three arguments must be given; all must be strictly
    positive.

    Raises:
        InvalidInputError: If not exactly two values are given or a value is
            not a strictly positive finite number.
    """
    require_n_given(
        {"speed": speed, "frequency": frequency, "wavelength": wavelength}, 2
    )
    v = None if speed is None else require_positive(speed, "speed")
    f = None if frequency is None else require_positive(frequency, "frequency")
    wl = None if wavelength is None else require_positive(wavelength, "wavelength")
    if v is None:
        assert f is not None and wl is not None
        v = f * wl
    elif f is None:
        assert wl is not None
        f = v / wl
    else:
        wl = v / f
    assert wl is not None
    return WaveProperties(
        speed=v,
        frequency=f,
        wavelength=wl,
        period=1.0 / f,
        angular_frequency=2.0 * math.pi * f,
        wavenumber=2.0 * math.pi / wl,
    )


def string_wave_speed(tension: float, linear_density: float) -> float:
    """Transverse wave speed on a stretched string: ``v = sqrt(T / mu)`` (m/s).

    Valid for small displacements of a perfectly flexible, uniform string.

    Args:
        tension: Tension in N (> 0).
        linear_density: Mass per unit length ``mu`` in kg/m (> 0).
    """
    return math.sqrt(
        require_positive(tension, "tension")
        / require_positive(linear_density, "linear_density")
    )


def wave_displacement(
    amplitude: float,
    wavelength: float,
    period: float,
    position: float,
    time: float,
    phase: float = 0.0,
) -> float:
    """Displacement of a travelling wave: ``y = A sin(2 pi (x/lambda - t/T) + phi)``.

    The wave moves in the +x direction at speed ``lambda / T``.

    Args:
        amplitude: Amplitude ``A`` in the displacement unit (>= 0 is not
            enforced; a negative value shifts the phase by pi).
        wavelength: Wavelength in m (> 0).
        period: Period in s (> 0).
        position: Position ``x`` in m.
        time: Time ``t`` in s.
        phase: Initial phase ``phi`` in rad.
    """
    a = require_finite(amplitude, "amplitude")
    wl = require_positive(wavelength, "wavelength")
    t_period = require_positive(period, "period")
    x = require_finite(position, "position")
    t = require_finite(time, "time")
    phi = require_finite(phase, "phase")
    return a * math.sin(2.0 * math.pi * (x / wl - t / t_period) + phi)
