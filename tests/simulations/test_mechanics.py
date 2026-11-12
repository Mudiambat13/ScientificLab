import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from scientificlab.constants import STANDARD_GRAVITY as G
from scientificlab.simulations import simulate_free_fall, simulate_projectile
from scientificlab.utils.exceptions import InvalidInputError


class TestProjectileVacuum:
    def test_range_formula_at_45_degrees(self):
        r = simulate_projectile(20.0, 45.0)
        assert r.method == "analytic"
        assert r.summary["range"] == pytest.approx(400.0 / G, rel=1e-12)
        assert r.summary["max_height"] == pytest.approx(
            (20 * math.sin(math.radians(45))) ** 2 / (2 * G)
        )
        assert r.summary["flight_time"] == pytest.approx(
            2 * 20 * math.sin(math.radians(45)) / G
        )

    def test_complementary_angles_same_range(self):
        a = simulate_projectile(30.0, 30.0).summary["range"]
        b = simulate_projectile(30.0, 60.0).summary["range"]
        assert a == pytest.approx(b, rel=1e-12)

    def test_energy_conservation_and_symmetric_impact_speed(self):
        r = simulate_projectile(25.0, 40.0)
        assert r.summary["impact_speed"] == pytest.approx(25.0)

    def test_launch_from_height(self):
        r = simulate_projectile(10.0, 0.0, height=20.0)
        t = math.sqrt(2 * 20.0 / G)
        assert r.summary["flight_time"] == pytest.approx(t)
        assert r.summary["range"] == pytest.approx(10.0 * t)
        assert r.summary["max_height"] == 20.0

    def test_vertical_throw(self):
        r = simulate_projectile(10.0, 90.0)
        assert r.summary["range"] == pytest.approx(0.0, abs=1e-12)
        assert r.summary["max_height"] == pytest.approx(100 / (2 * G))

    def test_series_shapes_and_endpoints(self):
        r = simulate_projectile(20.0, 45.0, points=11)
        assert r.time.shape == (11,)
        assert all(v.shape == (11,) for v in r.series.values())
        assert r.series["y (m)"][0] == 0.0
        assert r.series["y (m)"][-1] == 0.0
        assert np.all(np.diff(r.time) > 0)

    def test_custom_gravity(self):
        moon = simulate_projectile(20.0, 45.0, gravity=1.62)
        assert moon.summary["range"] == pytest.approx(400 / 1.62)


class TestProjectileDrag:
    def test_drag_reduces_range_height_and_impact_speed(self):
        vac = simulate_projectile(20.0, 45.0)
        air = simulate_projectile(20.0, 45.0, drag=0.01)
        assert "numeric" in air.method
        assert air.summary["range"] < vac.summary["range"]
        assert air.summary["max_height"] < vac.summary["max_height"]
        assert air.summary["flight_time"] < vac.summary["flight_time"]
        assert air.summary["impact_speed"] < 20.0

    def test_vanishing_drag_converges_to_vacuum(self):
        vac = simulate_projectile(20.0, 45.0)
        air = simulate_projectile(20.0, 45.0, drag=1e-9)
        assert air.summary["range"] == pytest.approx(vac.summary["range"], rel=1e-6)
        assert air.summary["max_height"] == pytest.approx(
            vac.summary["max_height"], rel=1e-6
        )

    def test_matches_independent_integration(self):
        c = 0.02
        r = simulate_projectile(30.0, 50.0, drag=c, height=5.0)

        def rhs(_t, s):
            v = math.hypot(s[2], s[3])
            return [s[2], s[3], -c * v * s[2], -G - c * v * s[3]]

        th = math.radians(50.0)
        sol = solve_ivp(
            rhs,
            (0, r.summary["flight_time"]),
            [0, 5.0, 30 * math.cos(th), 30 * math.sin(th)],
            rtol=1e-12,
            atol=1e-12,
        )
        assert sol.y[0, -1] == pytest.approx(r.summary["range"], rel=1e-7)
        assert sol.y[1, -1] == pytest.approx(0.0, abs=1e-5)

    def test_vertical_throw_with_drag_apex(self):
        # Up-throw with quadratic drag: apex = ln(1 + c v0^2 / g) / (2 c)
        c, v0 = 0.05, 20.0
        r = simulate_projectile(v0, 90.0, drag=c)
        expected = math.log(1 + c * v0**2 / G) / (2 * c)
        assert r.summary["max_height"] == pytest.approx(expected, rel=1e-7)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"speed": 0, "angle_degrees": 45},
        {"speed": -1, "angle_degrees": 45},
        {"speed": 10, "angle_degrees": -1},
        {"speed": 10, "angle_degrees": 91},
        {"speed": 10, "angle_degrees": float("nan")},
        {"speed": 10, "angle_degrees": 45, "height": -1},
        {"speed": 10, "angle_degrees": 45, "gravity": 0},
        {"speed": 10, "angle_degrees": 45, "drag": -0.1},
        {"speed": 10, "angle_degrees": 0},  # never leaves the ground
        {"speed": 10, "angle_degrees": 45, "points": 1},
    ],
)
def test_projectile_invalid_inputs(kwargs):
    with pytest.raises(InvalidInputError):
        simulate_projectile(**kwargs)


class TestFreeFall:
    def test_vacuum(self):
        r = simulate_free_fall(100.0)
        assert r.summary["fall_time"] == pytest.approx(math.sqrt(200 / G))
        assert r.summary["impact_speed"] == pytest.approx(math.sqrt(2 * G * 100))
        assert r.series["y (m)"][0] == 100.0
        assert r.series["y (m)"][-1] == pytest.approx(0.0, abs=1e-9)

    def test_one_second_fall_is_half_g(self):
        r = simulate_free_fall(1000.0)
        t = r.time
        y = r.series["y (m)"]
        i = int(np.argmin(np.abs(t - 1.0)))
        assert 1000.0 - y[i] == pytest.approx(0.5 * G * t[i] ** 2)

    def test_drag_matches_numerical_integration(self):
        c, h = 0.01, 500.0
        r = simulate_free_fall(h, drag=c)

        def rhs(_t, s):
            return [s[1], G - c * s[1] ** 2]  # s = (distance fallen, speed)

        sol = solve_ivp(
            rhs, (0, r.summary["fall_time"]), [0.0, 0.0], rtol=1e-12, atol=1e-12
        )
        assert sol.y[0, -1] == pytest.approx(h, rel=1e-8)
        assert sol.y[1, -1] == pytest.approx(r.summary["impact_speed"], rel=1e-8)

    def test_terminal_speed_is_not_exceeded(self):
        r = simulate_free_fall(10_000.0, drag=0.01)
        vt = math.sqrt(G / 0.01)
        assert r.summary["terminal_speed"] == pytest.approx(vt)
        assert np.max(r.series["v (m/s)"]) <= vt
        assert r.summary["impact_speed"] == pytest.approx(vt, rel=1e-9)

    def test_drag_slows_the_fall_and_small_drag_is_vacuum(self):
        vac = simulate_free_fall(100.0)
        assert (
            simulate_free_fall(100.0, drag=0.01).summary["fall_time"]
            > vac.summary["fall_time"]
        )
        assert simulate_free_fall(100.0, drag=1e-10).summary[
            "fall_time"
        ] == pytest.approx(vac.summary["fall_time"], rel=1e-7)

    def test_huge_drag_height_ratio_does_not_overflow(self):
        r = simulate_free_fall(1e6, drag=10.0)  # c*h = 1e7
        assert np.all(np.isfinite(r.series["y (m)"]))
        assert math.isfinite(r.summary["fall_time"])

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"height": 0},
            {"height": -5},
            {"height": 1, "gravity": -1},
            {"height": 1, "drag": -1},
            {"height": float("inf")},
        ],
    )
    def test_invalid(self, kwargs):
        with pytest.raises(InvalidInputError):
            simulate_free_fall(**kwargs)
