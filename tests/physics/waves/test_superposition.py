import math

import numpy as np
import pytest

from scientificlab.physics.waves import beat_frequency, superpose
from scientificlab.utils.exceptions import InvalidInputError


class TestSuperpose:
    def test_constructive_interference_adds_amplitudes(self):
        result = superpose([1.0, 1.0], [0.0, 0.0])
        assert result.amplitude == pytest.approx(2.0)
        assert result.phase == pytest.approx(0.0)

    def test_destructive_interference_cancels(self):
        result = superpose([1.0, 1.0], [0.0, math.pi])
        assert result.amplitude == pytest.approx(0.0, abs=1e-12)
        assert result.phase is None

    def test_quarter_period_offset(self):
        result = superpose([1.0, 1.0], [0.0, math.pi / 2])
        assert result.amplitude == pytest.approx(math.sqrt(2.0))
        assert result.phase == pytest.approx(math.pi / 4)

    def test_unequal_amplitudes_in_antiphase(self):
        result = superpose([3.0, 1.0], [0.0, math.pi])
        assert result.amplitude == pytest.approx(2.0)
        assert result.phase == pytest.approx(0.0)

    def test_three_symmetric_phases_cancel(self):
        result = superpose([1, 1, 1], [0.0, 2 * math.pi / 3, 4 * math.pi / 3])
        assert result.phase is None

    def test_matches_direct_summation_of_the_waves(self):
        amplitudes = [1.2, 0.7, 2.0]
        phases = [0.3, 1.9, -0.8]
        result = superpose(amplitudes, phases)
        t = np.linspace(0.0, 4 * math.pi, 200)
        direct = sum(a * np.sin(t + p) for a, p in zip(amplitudes, phases, strict=True))
        assert result.phase is not None
        assert direct == pytest.approx(result.amplitude * np.sin(t + result.phase))

    def test_amplitude_never_exceeds_the_sum_of_amplitudes(self):
        result = superpose([1.0, 2.0, 0.5], [0.4, 2.2, 5.0])
        assert result.amplitude <= 3.5 + 1e-12

    def test_a_single_wave_is_unchanged(self):
        result = superpose([2.5], [0.7])
        assert result.amplitude == pytest.approx(2.5)
        assert result.phase == pytest.approx(0.7)

    @pytest.mark.parametrize(
        ("amplitudes", "phases"),
        [
            ([], []),
            ([1.0, 1.0], [0.0]),
            ([-1.0, 1.0], [0.0, 0.0]),
            ([1.0, float("nan")], [0.0, 0.0]),
            ([1.0, 1.0], [0.0, float("inf")]),
        ],
    )
    def test_invalid_inputs(self, amplitudes, phases):
        with pytest.raises(InvalidInputError):
            superpose(amplitudes, phases)


def test_beat_frequency():
    assert beat_frequency(440, 444) == pytest.approx(4.0)
    assert beat_frequency(444, 440) == pytest.approx(4.0)
    assert beat_frequency(440, 440) == 0.0
    with pytest.raises(InvalidInputError):
        beat_frequency(0, 440)
