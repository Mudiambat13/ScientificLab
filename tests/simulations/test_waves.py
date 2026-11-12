import numpy as np
import pytest

from scientificlab.simulations import simulate_string_wave
from scientificlab.utils.exceptions import InvalidInputError

L, C = 1.0, 100.0


def mode(n):
    return lambda x: np.sin(n * np.pi * x / L)


def test_standing_wave_is_exact_at_courant_one():
    w = simulate_string_wave(L, C, mode(3), duration=0.02, nx=401, courant=1.0)
    exact = (
        np.sin(3 * np.pi * w.x / L)[None, :]
        * np.cos(3 * np.pi * C * w.time / L)[:, None]
    )
    assert np.max(np.abs(w.displacement - exact)) < 1e-12


@pytest.mark.parametrize("courant", [0.5, 0.9])
def test_standing_wave_second_order_accuracy(courant):
    errors = []
    for nx in (101, 201):
        w = simulate_string_wave(L, C, mode(2), duration=0.015, nx=nx, courant=courant)
        exact = (
            np.sin(2 * np.pi * w.x / L)[None, :]
            * np.cos(2 * np.pi * C * w.time / L)[:, None]
        )
        errors.append(np.max(np.abs(w.displacement - exact)))
    assert errors[1] < 2e-3
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.25)  # O(dx^2)


def test_energy_is_conserved():
    def pluck(x):
        return np.exp(-(((x - 0.3) / 0.05) ** 2))

    w = simulate_string_wave(L, C, pluck, duration=0.05, nx=401, courant=0.9)
    spread = (w.energy.max() - w.energy.min()) / w.energy.mean()
    assert spread < 1e-9
    assert w.energy.shape == w.energy_time.shape == (w.time.size - 1,)


def test_initial_velocity_standing_mode():
    # u(x,0)=0, u_t(x,0)=w sin(pi x/L)  ->  u = sin(pi x/L) sin(w t), w = pi c / L
    w_ = np.pi * C / L
    w = simulate_string_wave(
        L,
        C,
        lambda x: 0 * x,
        initial_velocity=lambda x: w_ * np.sin(np.pi * x / L),
        duration=0.01,
        nx=401,
        courant=1.0,
    )
    exact = np.sin(np.pi * w.x / L)[None, :] * np.sin(w_ * w.time)[:, None]
    assert np.max(np.abs(w.displacement - exact)) < 1e-4  # 2nd order, not exact


def test_pulse_reflects_with_sign_flip_and_returns():
    def pulse(x):
        return np.exp(-(((x - 0.5) / 0.04) ** 2))

    w = simulate_string_wave(L, C, pulse, duration=0.02, nx=401, courant=1.0)
    # After one full period 2L/c = 0.02 s the initial shape is restored.
    assert np.max(np.abs(w.displacement[-1] - w.displacement[0])) < 1e-10
    # After half a period the shape is inverted and mirrored (sign flip on reflection).
    half = w.displacement[w.displacement.shape[0] // 2]
    assert np.max(np.abs(half + w.displacement[0][::-1])) < 1e-10


def test_boundaries_stay_fixed():
    w = simulate_string_wave(L, C, mode(1), duration=0.05, nx=101)
    assert np.all(w.displacement[:, 0] == 0.0)
    assert np.all(w.displacement[:, -1] == 0.0)


def test_grid_and_time_metadata():
    w = simulate_string_wave(L, C, mode(1), duration=0.01, nx=101, courant=0.8)
    assert w.x[0] == 0.0 and w.x[-1] == L
    assert w.summary["dx"] == pytest.approx(L / 100)
    assert w.summary["dt"] == pytest.approx(0.8 * L / 100 / C)
    assert w.time[-1] <= 0.01 < w.time[-1] + w.summary["dt"]
    assert "dispersion" in " ".join(w.notes)


@pytest.mark.parametrize("courant", [1.01, 2.0, 0.0, -0.5])
def test_unstable_or_invalid_courant_is_refused(courant):
    with pytest.raises(InvalidInputError, match="courant"):
        simulate_string_wave(L, C, mode(1), duration=0.01, courant=courant)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"length": 0},
        {"wave_speed": -1},
        {"duration": 0},
        {"nx": 2},
        {"nx": True},
        {"duration": 1e-12},
        {"nx": 4001, "duration": 10.0},
    ],
)
def test_invalid_parameters(kwargs):
    base = {
        "length": L,
        "wave_speed": C,
        "initial_displacement": mode(1),
        "duration": 0.01,
    }
    with pytest.raises(InvalidInputError):
        simulate_string_wave(**{**base, **kwargs})


def test_initial_data_must_vanish_at_ends():
    with pytest.raises(InvalidInputError, match="vanish"):
        simulate_string_wave(L, C, lambda x: np.ones_like(x), duration=0.01)
    with pytest.raises(InvalidInputError, match="vanish"):
        simulate_string_wave(
            L, C, mode(1), initial_velocity=lambda x: np.ones_like(x), duration=0.01
        )


def test_bad_initial_functions():
    with pytest.raises(InvalidInputError, match="function"):
        simulate_string_wave(L, C, "sin(x)", duration=0.01)  # type: ignore[arg-type]
    with pytest.raises(InvalidInputError, match="shaped"):
        simulate_string_wave(L, C, lambda x: np.zeros(3), duration=0.01)
    with pytest.raises(InvalidInputError, match="finite"):
        simulate_string_wave(L, C, lambda x: x * np.nan, duration=0.01)
    with pytest.raises(InvalidInputError, match="evaluated"):
        simulate_string_wave(L, C, lambda x: x + "a", duration=0.01)  # type: ignore[operator]
