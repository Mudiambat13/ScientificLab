import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp, trapezoid

from scientificlab.simulations import (
    cooling_time_to_reach,
    simulate_cooling,
    simulate_harmonic_oscillator,
    simulate_rc_circuit,
)
from scientificlab.utils.exceptions import InvalidInputError, NoSolutionError


class TestOscillator:
    def test_undamped_period_and_cosine(self):
        r = simulate_harmonic_oscillator(1.0, 100.0, duration=1.0, points=1001)
        w0 = 10.0
        assert r.summary["regime"] == "undamped"
        assert r.summary["omega0"] == pytest.approx(w0)
        assert r.summary["period"] == pytest.approx(2 * math.pi / w0)
        assert r.series["x (m)"] == pytest.approx(np.cos(w0 * r.time), abs=1e-12)
        assert r.series["v (m/s)"] == pytest.approx(
            -w0 * np.sin(w0 * r.time), abs=1e-12
        )

    def test_undamped_energy_is_conserved(self):
        r = simulate_harmonic_oscillator(2.0, 50.0, initial_velocity=1.0)
        e = r.series["energy (J)"]
        assert e == pytest.approx(np.full_like(e, e[0]), rel=1e-12)

    @pytest.mark.parametrize(
        ("damping", "regime"),
        [
            (0.0, "undamped"),
            (2.0, "underdamped"),
            (20.0, "critically damped"),
            (60.0, "overdamped"),
        ],
    )
    def test_regimes_match_numerical_integration(self, damping, regime):
        m, k, x0, v0 = 1.0, 100.0, 0.3, -1.0
        r = simulate_harmonic_oscillator(
            m,
            k,
            damping=damping,
            initial_position=x0,
            initial_velocity=v0,
            duration=2.0,
            points=201,
        )
        assert r.summary["regime"] == regime
        sol = solve_ivp(
            lambda _t, s: [s[1], -(k * s[0] + damping * s[1]) / m],
            (0, 2.0),
            [x0, v0],
            t_eval=r.time,
            rtol=1e-12,
            atol=1e-13,
        )
        assert r.series["x (m)"] == pytest.approx(sol.y[0], abs=1e-9)
        assert r.series["v (m/s)"] == pytest.approx(sol.y[1], abs=1e-9)

    def test_damped_energy_never_increases(self):
        r = simulate_harmonic_oscillator(1.0, 100.0, damping=3.0)
        assert np.all(np.diff(r.series["energy (J)"]) <= 1e-12)

    def test_damping_ratio_and_initial_conditions_honoured(self):
        r = simulate_harmonic_oscillator(
            1.0, 100.0, damping=20.0, initial_position=0.5, initial_velocity=2.0
        )
        assert r.summary["damping_ratio"] == pytest.approx(1.0)
        assert r.series["x (m)"][0] == pytest.approx(0.5)
        assert r.series["v (m/s)"][0] == pytest.approx(2.0)

    def test_default_duration_is_five_periods(self):
        r = simulate_harmonic_oscillator(1.0, 100.0)
        assert r.time[-1] == pytest.approx(5 * 2 * math.pi / 10.0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"mass": 0, "stiffness": 1},
            {"mass": 1, "stiffness": -1},
            {"mass": 1, "stiffness": 1, "damping": -1},
            {"mass": 1, "stiffness": 1, "duration": 0},
            {"mass": 1, "stiffness": 1, "initial_position": float("nan")},
            {"mass": 1, "stiffness": 1, "points": 1},
        ],
    )
    def test_invalid(self, kwargs):
        with pytest.raises(InvalidInputError):
            simulate_harmonic_oscillator(**kwargs)


class TestRC:
    def test_charging_reference_values(self):
        r = simulate_rc_circuit(5.0, 1000.0, 1e-3, points=6)  # tau = 1 s
        assert r.summary["tau"] == pytest.approx(1.0)
        v = r.series["voltage (V)"]
        assert v[1] == pytest.approx(5 * (1 - math.exp(-1)))
        assert v[-1] == pytest.approx(5 * (1 - math.exp(-5)))

    def test_current_is_ohms_law(self):
        r = simulate_rc_circuit(5.0, 200.0, 1e-3, initial_voltage=1.0)
        i = r.series["current (A)"]
        v = r.series["voltage (V)"]
        assert i == pytest.approx((5.0 - v) / 200.0)
        assert i[0] == pytest.approx(4.0 / 200.0)

    def test_energy_conservation(self):
        r = simulate_rc_circuit(12.0, 470.0, 2.2e-3, initial_voltage=3.0)
        s = r.series
        balance = s["stored energy (J)"] - s["stored energy (J)"][0]
        assert s["supplied energy (J)"] == pytest.approx(
            balance + s["dissipated energy (J)"], rel=1e-12, abs=1e-15
        )

    def test_dissipated_energy_matches_numerical_integral(self):
        r = simulate_rc_circuit(10.0, 100.0, 1e-3, points=20001)
        i = r.series["current (A)"]
        numeric = trapezoid(i**2 * 100.0, r.time)
        assert r.series["dissipated energy (J)"][-1] == pytest.approx(numeric, rel=1e-6)

    def test_charging_dissipates_half_the_supplied_energy(self):
        r = simulate_rc_circuit(10.0, 100.0, 1e-3, duration=100 * 0.1)
        assert r.series["dissipated energy (J)"][-1] == pytest.approx(
            0.5 * r.series["supplied energy (J)"][-1], rel=1e-9
        )

    def test_discharge(self):
        r = simulate_rc_circuit(0.0, 100.0, 1e-3, initial_voltage=6.0)
        assert r.series["voltage (V)"][0] == 6.0
        assert r.series["voltage (V)"][-1] == pytest.approx(6 * math.exp(-5))
        assert np.all(r.series["current (A)"] < 0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"resistance": 0},
            {"capacitance": -1},
            {"source_voltage": float("inf")},
            {"duration": -1},
            {"points": 0},
        ],
    )
    def test_invalid(self, kwargs):
        base = {"source_voltage": 5.0, "resistance": 100.0, "capacitance": 1e-3}
        with pytest.raises(InvalidInputError):
            simulate_rc_circuit(**{**base, **kwargs})


class TestCooling:
    def test_reference_values(self):
        r = simulate_cooling(363.15, 293.15, 0.01, duration=100.0, points=101)
        assert r.series["temperature (K)"][0] == pytest.approx(363.15)
        assert r.series["temperature (K)"][-1] == pytest.approx(
            293.15 + 70.0 * math.exp(-1.0)
        )
        assert r.summary["time_constant"] == pytest.approx(100.0)
        assert r.summary["half_life"] == pytest.approx(math.log(2) * 100.0)

    def test_half_life_halves_the_excess(self):
        r = simulate_cooling(400.0, 300.0, 0.02)
        k = 0.02
        t_half = r.summary["half_life"]
        temp = 300.0 + 100.0 * math.exp(-k * t_half)
        assert temp == pytest.approx(350.0)

    def test_never_crosses_ambient_heating_and_cooling(self):
        cool = simulate_cooling(350.0, 300.0, 0.1, duration=1000.0)
        assert np.all(cool.series["temperature (K)"] >= 300.0)
        warm = simulate_cooling(250.0, 300.0, 0.1, duration=1000.0)
        assert np.all(warm.series["temperature (K)"] <= 300.0)

    def test_time_to_reach_inverts_the_curve(self):
        t = cooling_time_to_reach(363.15, 293.15, 0.01, 323.15)
        assert 293.15 + 70.0 * math.exp(-0.01 * t) == pytest.approx(323.15)

    def test_unreachable_targets(self):
        with pytest.raises(NoSolutionError, match="asymptotically"):
            cooling_time_to_reach(350.0, 300.0, 0.1, 300.0)
        with pytest.raises(NoSolutionError, match="asymptotically"):
            cooling_time_to_reach(350.0, 300.0, 0.1, 290.0)
        with pytest.raises(NoSolutionError, match="other side"):
            cooling_time_to_reach(350.0, 300.0, 0.1, 360.0)
        with pytest.raises(InvalidInputError, match="already"):
            cooling_time_to_reach(300.0, 300.0, 0.1, 310.0)

    @pytest.mark.parametrize(
        "args",
        [
            (0.0, 300.0, 0.1),
            (-5.0, 300.0, 0.1),
            (350.0, 0.0, 0.1),
            (350.0, 300.0, 0.0),
            (350.0, 300.0, -1.0),
            (float("nan"), 300.0, 0.1),
        ],
    )
    def test_invalid(self, args):
        with pytest.raises(InvalidInputError):
            simulate_cooling(*args)
        with pytest.raises(InvalidInputError):
            cooling_time_to_reach(*args, 320.0)
