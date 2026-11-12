"""Superposition of waves (linear media: displacements add).

Sinusoidal waves of the **same frequency** combine into a sinusoid of the same
frequency whose amplitude and phase follow from the phasor sum.
"""

import cmath
import math
from collections.abc import Sequence
from dataclasses import dataclass

from scientificlab.utils.exceptions import InvalidInputError
from scientificlab.utils.validation import (
    require_finite,
    require_non_negative,
    require_positive,
)

_NEGLIGIBLE = 1e-12


@dataclass(frozen=True)
class SuperpositionResult:
    """Resultant of several same-frequency sinusoids.

    Attributes:
        amplitude: Amplitude of the resultant (same unit as the inputs).
        phase: Phase of the resultant in rad, in ``(-pi, pi]``, or ``None`` when
            the waves cancel (amplitude negligible), where the phase is undefined.
    """

    amplitude: float
    phase: float | None


def superpose(
    amplitudes: Sequence[float], phases: Sequence[float]
) -> SuperpositionResult:
    """Sum ``A_k * sin(w*t + phi_k)`` over waves of identical frequency.

    The resultant is ``A * sin(w*t + phi)`` with ``A * exp(i*phi) = sum_k A_k *
    exp(i*phi_k)``. Constructive interference gives ``sum(A_k)`` and total
    destructive interference 0. Cancellation is judged against the sum of the
    amplitudes with a relative tolerance of 1e-12.

    Args:
        amplitudes: Amplitudes (>= 0), at least one.
        phases: Phases in rad, one per amplitude.

    Raises:
        InvalidInputError: If the sequences are empty, differ in length, or
            contain invalid values.
    """
    if len(amplitudes) == 0 or len(amplitudes) != len(phases):
        raise InvalidInputError(
            "amplitudes and phases must be non-empty and equally long."
        )
    values = [
        require_non_negative(a, f"amplitudes[{i}]") for i, a in enumerate(amplitudes)
    ]
    angles = [require_finite(p, f"phases[{i}]") for i, p in enumerate(phases)]
    total = sum(a * cmath.exp(1j * p) for a, p in zip(values, angles, strict=True))
    amplitude = abs(total)
    if amplitude <= _NEGLIGIBLE * math.fsum(values):
        return SuperpositionResult(0.0, None)
    return SuperpositionResult(amplitude, cmath.phase(total))


def beat_frequency(frequency_1: float, frequency_2: float) -> float:
    """Beat frequency of two close tones: ``|f_1 - f_2|`` (Hz).

    The amplitude envelope of the sum of two waves of similar amplitude
    oscillates at this rate (each loudness maximum is one beat).
    """
    return abs(
        require_positive(frequency_1, "frequency_1")
        - require_positive(frequency_2, "frequency_2")
    )
