"""Sound waves: speed, intensity level and the Doppler effect (SI units).

Model: plane acoustic waves of small amplitude in a homogeneous ideal gas.
"""

import math

from scientificlab.constants import GAS_CONSTANT, ZERO_CELSIUS
from scientificlab.utils.exceptions import PhysicalInconsistencyError
from scientificlab.utils.validation import require_finite, require_positive

REFERENCE_INTENSITY: float = 1e-12
"""Reference intensity ``I_0`` in W/m**2 (conventional hearing threshold at 1 kHz)."""

AIR_MOLAR_MASS: float = 0.0289644
"""Molar mass of dry air in kg/mol (U.S. Standard Atmosphere, 1976)."""

AIR_HEAT_CAPACITY_RATIO: float = 1.4
"""Heat capacity ratio of air ``gamma`` (diatomic ideal gas, near room temperature)."""


def sound_speed_ideal_gas(
    heat_capacity_ratio: float,
    temperature: float,
    molar_mass: float,
) -> float:
    """Speed of sound in an ideal gas: ``v = sqrt(gamma * R * T / M)`` (m/s).

    Args:
        heat_capacity_ratio: ``gamma = C_p/C_v`` (> 0).
        temperature: Absolute temperature in K (> 0).
        molar_mass: Molar mass in kg/mol (> 0).
    """
    return math.sqrt(
        require_positive(heat_capacity_ratio, "heat_capacity_ratio")
        * GAS_CONSTANT
        * require_positive(temperature, "temperature")
        / require_positive(molar_mass, "molar_mass")
    )


def sound_speed_in_air(temperature_celsius: float) -> float:
    """Speed of sound in dry air at a given temperature (m/s).

    Uses the ideal-gas formula with ``gamma = 1.4`` and the molar mass of dry air.
    Accurate to about 0.2 % near room temperature; humidity and the variation
    of gamma with temperature are ignored.

    Raises:
        PhysicalInconsistencyError: If the temperature is below absolute zero.
    """
    kelvin = require_finite(temperature_celsius, "temperature_celsius") + ZERO_CELSIUS
    if kelvin <= 0.0:
        raise PhysicalInconsistencyError("Temperature must be above absolute zero.")
    return sound_speed_ideal_gas(AIR_HEAT_CAPACITY_RATIO, kelvin, AIR_MOLAR_MASS)


def sound_intensity_level(intensity: float) -> float:
    """Sound intensity level ``L = 10 log10(I / I_0)`` in decibels.

    Args:
        intensity: Intensity in W/m**2 (> 0), with ``I_0 = 1e-12`` W/m**2.
    """
    return 10.0 * math.log10(
        require_positive(intensity, "intensity") / REFERENCE_INTENSITY
    )


def intensity_from_level(level_db: float) -> float:
    """Intensity in W/m**2 from a level in dB: ``I = I_0 * 10**(L/10)``.

    Raises:
        InvalidInputError: If the level is not finite.
        PhysicalInconsistencyError: If the intensity overflows the float range.
    """
    level = require_finite(level_db, "level_db")
    try:
        return REFERENCE_INTENSITY * float(10.0 ** (level / 10.0))
    except OverflowError as error:
        raise PhysicalInconsistencyError(f"A level of {level} dB overflows.") from error


def doppler_frequency(
    source_frequency: float,
    wave_speed: float,
    source_velocity: float = 0.0,
    observer_velocity: float = 0.0,
) -> float:
    """Observed frequency ``f' = f * (v + v_o) / (v - v_s)`` (Hz).

    Velocities are measured along the line joining source and observer, with
    the medium at rest:

    * ``source_velocity > 0``: the source moves **toward** the observer
      (frequency rises); ``< 0`` away.
    * ``observer_velocity > 0``: the observer moves **toward** the source
      (frequency rises); ``< 0`` away.

    Args:
        source_frequency: Emitted frequency in Hz (> 0).
        wave_speed: Speed of sound in the medium in m/s (> 0).
        source_velocity: Source velocity in m/s.
        observer_velocity: Observer velocity in m/s.

    Raises:
        PhysicalInconsistencyError: If the source reaches or exceeds the wave
            speed (a shock wave forms and the formula is invalid), or the
            observer recedes at least as fast as the wave (no wave reaches it).
    """
    f = require_positive(source_frequency, "source_frequency")
    v = require_positive(wave_speed, "wave_speed")
    v_s = require_finite(source_velocity, "source_velocity")
    v_o = require_finite(observer_velocity, "observer_velocity")
    if v_s >= v:
        raise PhysicalInconsistencyError(
            "The source moves at or above the wave speed: the Doppler formula "
            "does not apply (shock wave)."
        )
    if v + v_o <= 0.0:
        raise PhysicalInconsistencyError(
            "The observer recedes at least as fast as the wave: it never receives it."
        )
    return f * (v + v_o) / (v - v_s)
