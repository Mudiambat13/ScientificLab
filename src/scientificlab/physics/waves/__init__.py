"""Waves: frequency, wavelength, mechanical and sound waves, superposition."""

from scientificlab.physics.waves.basics import (
    WaveProperties,
    frequency_from_period,
    period_from_frequency,
    string_wave_speed,
    wave_displacement,
    wave_relation,
)
from scientificlab.physics.waves.sound import (
    doppler_frequency,
    intensity_from_level,
    sound_intensity_level,
    sound_speed_ideal_gas,
    sound_speed_in_air,
)
from scientificlab.physics.waves.superposition import (
    SuperpositionResult,
    beat_frequency,
    superpose,
)

__all__ = [
    "SuperpositionResult",
    "WaveProperties",
    "beat_frequency",
    "doppler_frequency",
    "frequency_from_period",
    "intensity_from_level",
    "period_from_frequency",
    "sound_intensity_level",
    "sound_speed_ideal_gas",
    "sound_speed_in_air",
    "string_wave_speed",
    "superpose",
    "wave_displacement",
    "wave_relation",
]
