import math

import numpy as np
import pytest
from matplotlib.figure import Figure

from scientificlab.utils.exceptions import ExpressionParseError, InvalidInputError
from scientificlab.visualization import (
    plot_data,
    plot_function,
    plot_functions,
    plot_isotherms,
    plot_position_vs_time,
    plot_rc_charging,
    save_figure,
)
from scientificlab.visualization._common import linspace, sample


def line_data(fig, index=0):
    return fig.axes[0].lines[index].get_data()


class TestFunctions:
    def test_returns_figure_with_correct_data(self):
        fig = plot_function("x**2", (-2, 2), points=5)
        assert isinstance(fig, Figure)
        x, y = line_data(fig)
        assert list(x) == [-2, -1, 0, 1, 2]
        assert list(y) == [4, 1, 0, 1, 4]

    def test_callable_labels_titles(self):
        fig = plot_functions(
            [math.sin, lambda x: x],
            (0, 1),
            labels=["sin", "id"],
            title="T",
            xlabel="t (s)",
            ylabel="y (m)",
        )
        ax = fig.axes[0]
        assert ax.get_title() == "T"
        assert ax.get_xlabel() == "t (s)"
        assert [t.get_text() for t in ax.get_legend().get_texts()] == ["sin", "id"]

    def test_scalar_only_callable_is_supported(self):
        _, y = line_data(plot_function(lambda x: math.sqrt(x), (0, 4), points=3))
        assert list(y) == pytest.approx([0.0, math.sqrt(2), 2.0])

    def test_undefined_points_are_gaps_not_lines(self):
        _, y = line_data(plot_function("sqrt(x)", (-1, 1), points=5))
        assert np.isnan(y[0]) and np.isnan(y[1])
        assert y[-1] == 1.0

    def test_pole_is_cut(self):
        _, y = line_data(plot_function("1/x", (-1, 1), points=100))
        assert np.isnan(y[49])  # the sign flip across x = 0 is broken

    def test_continuous_zero_crossing_is_not_cut(self):
        _, y = line_data(plot_function("sin(x)", (-3, 3), points=100))
        assert not np.any(np.isnan(y))

    def test_everywhere_undefined(self):
        with pytest.raises(InvalidInputError, match="not finite"):
            plot_function("log(x)", (-5, -1))

    def test_y_limits(self):
        fig = plot_function("x", (0, 1), y_limits=(-2, 2))
        assert fig.axes[0].get_ylim() == (-2, 2)
        with pytest.raises(InvalidInputError):
            plot_function("x", (0, 1), y_limits=(2, -2))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"x_range": (1, 1)},
            {"x_range": (2, 1)},
            {"x_range": (0, float("inf"))},
            {"x_range": (0, 1, 2)},
            {"x_range": (0, 1), "points": 1},
        ],
    )
    def test_bad_range_or_points(self, kwargs):
        with pytest.raises(InvalidInputError):
            plot_function("x", **kwargs)

    def test_bad_function_arguments(self):
        with pytest.raises(InvalidInputError):
            plot_functions([], (0, 1))
        with pytest.raises(InvalidInputError):
            plot_functions(["x", "x**2"], (0, 1), labels=["only one"])
        with pytest.raises(InvalidInputError, match="x only"):
            plot_function("a*x", (0, 1))
        with pytest.raises(ExpressionParseError):
            plot_function("x +* 1", (0, 1))

    def test_complex_values_become_gaps(self):
        xs = np.array([-1.0, 1.0])
        y = sample(lambda x: np.sqrt(x + 0j) * 1.0, xs)
        assert np.isnan(y[0]) and y[1] == 1.0

    def test_linspace_validation(self):
        assert list(linspace((0, 1), 3)) == [0.0, 0.5, 1.0]


class TestData:
    def test_plot_data_with_error_bars(self):
        fig = plot_data([1, 2, 3], [2, 4, 5], yerr=[0.1, 0.1, 0.2], label="m")
        assert isinstance(fig, Figure)
        assert fig.axes[0].get_legend() is not None

    def test_plot_data_connected(self):
        fig = plot_data([1, 2], [3, 4], connect=True)
        assert fig.axes[0].lines[0].get_linestyle() == "-"

    @pytest.mark.parametrize(
        ("x", "y", "yerr"),
        [
            ([], [], None),
            ([1, 2], [1], None),
            ([1, float("nan")], [1, 2], None),
            ([1, 2], [1, 2], [-1, 1]),
            ([1, 2], [1, 2], [1]),
            ([[1]], [[1]], None),
        ],
    )
    def test_bad_data(self, x, y, yerr):
        with pytest.raises(InvalidInputError):
            plot_data(x, y, yerr=yerr)


class TestPhysicsPlots:
    def test_rc_charging_curve(self):
        fig = plot_rc_charging(5.0, 1000.0, 1e-3, points=6)  # tau = 1 s, 0..5 s
        t, v = line_data(fig)
        assert t[-1] == pytest.approx(5.0)
        assert v[0] == 0.0
        assert v[1] == pytest.approx(5 * (1 - math.exp(-1)))  # t = tau
        assert v[-1] == pytest.approx(5 * (1 - math.exp(-5)))

    def test_rc_invalid(self):
        with pytest.raises(InvalidInputError):
            plot_rc_charging(5, -1, 1e-3)
        with pytest.raises(InvalidInputError):
            plot_rc_charging(5, 1, 1e-3, duration_in_tau=0)
        with pytest.raises(InvalidInputError):
            plot_rc_charging(5, 1, 1e-3, points=1)

    def test_position_curve(self):
        _t, x = line_data(plot_position_vs_time(0.0, 0.0, 2.0, 4.0, points=5))
        assert list(x) == pytest.approx([0.0, 1.0, 4.0, 9.0, 16.0])  # a t^2 / 2

    def test_position_invalid(self):
        with pytest.raises(InvalidInputError):
            plot_position_vs_time(0, 0, 1, 0)
        with pytest.raises(InvalidInputError):
            plot_position_vs_time(0, 0, 1, 1, points=1)
        with pytest.raises(InvalidInputError):
            plot_position_vs_time(0, 0, float("nan"), 1)

    def test_isotherm_obeys_boyle(self):
        fig = plot_isotherms(1.0, [300.0, 600.0], (0.01, 0.1), points=4)
        v, p300 = line_data(fig, 0)
        _, p600 = line_data(fig, 1)
        assert p300[0] * v[0] == pytest.approx(8.314462618 * 300.0, rel=1e-9)
        assert p300 * v == pytest.approx(np.full(4, p300[0] * v[0]))
        assert p600 == pytest.approx(2 * p300)  # same V, double T

    def test_isotherm_invalid(self):
        with pytest.raises(InvalidInputError):
            plot_isotherms(1, [], (0.01, 0.1))
        with pytest.raises(InvalidInputError):
            plot_isotherms(1, [300], (0.1, 0.01))
        with pytest.raises(InvalidInputError):
            plot_isotherms(1, [300], (0.01, 0.1), points=1)
        with pytest.raises(InvalidInputError):
            plot_isotherms(1, [-5], (0.01, 0.1))


class TestExport:
    @pytest.mark.parametrize(
        ("name", "magic"),
        [("f.png", b"\x89PNG"), ("f.svg", b"<?xml"), ("f.pdf", b"%PDF")],
    )
    def test_export_formats(self, tmp_path, name, magic):
        path = tmp_path / name
        plot_function("x", (0, 1), save_path=path)
        assert path.read_bytes().startswith(magic)

    def test_upper_case_extension_and_string_path(self, tmp_path):
        target = str(tmp_path / "F.PNG")
        plot_function("x", (0, 1), save_path=target)
        assert (tmp_path / "F.PNG").stat().st_size > 0

    def test_unsupported_extension_writes_nothing(self, tmp_path):
        with pytest.raises(InvalidInputError, match="Unsupported format"):
            plot_function("x", (0, 1), save_path=tmp_path / "f.jpg")
        assert list(tmp_path.iterdir()) == []

    def test_missing_directory(self, tmp_path):
        with pytest.raises(InvalidInputError, match="does not exist"):
            plot_function("x", (0, 1), save_path=tmp_path / "nope" / "f.png")

    def test_save_figure_returns_path(self, tmp_path):
        fig = plot_function("x", (0, 1))
        assert save_figure(fig, tmp_path / "g.png") == tmp_path / "g.png"

    def test_no_pyplot_state_is_created(self):
        import sys

        plot_function("x", (0, 1))
        pyplot = sys.modules.get("matplotlib.pyplot")
        assert pyplot is None or not pyplot.get_fignums()


def test_show_uses_pyplot_lazily(monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(True))
    fig = plot_function("x", (0, 1), show=True)
    assert shown == [True]
    assert isinstance(fig, Figure)
    plt.close("all")


class TestSimulationPlots:
    def test_plot_simulation_panels(self, tmp_path):
        from scientificlab.simulations import simulate_harmonic_oscillator
        from scientificlab.visualization import plot_simulation

        r = simulate_harmonic_oscillator(1.0, 100.0, points=50)
        fig = plot_simulation(r, save_path=tmp_path / "s.png")
        assert len(fig.axes) == 3
        assert fig.axes[0].lines[0].get_ydata()[0] == pytest.approx(1.0)
        assert (tmp_path / "s.png").stat().st_size > 0
        assert len(plot_simulation(r, ["x (m)"]).axes) == 1

    def test_plot_simulation_errors(self):
        from scientificlab.simulations import simulate_cooling
        from scientificlab.visualization import plot_simulation

        r = simulate_cooling(350.0, 300.0, 0.1)
        with pytest.raises(InvalidInputError, match="Unknown"):
            plot_simulation(r, ["nope"])
        with pytest.raises(InvalidInputError):
            plot_simulation(r, [])

    def test_trajectory(self):
        from scientificlab.simulations import simulate_cooling, simulate_projectile
        from scientificlab.visualization import plot_trajectory

        r = simulate_projectile(20.0, 45.0, points=11)
        x, y = plot_trajectory(r).axes[0].lines[0].get_data()
        assert x[-1] == pytest.approx(r.summary["range"])
        assert max(y) == pytest.approx(r.summary["max_height"], rel=0.05)
        with pytest.raises(InvalidInputError):
            plot_trajectory(simulate_cooling(350.0, 300.0, 0.1))

    def test_wave_snapshots(self):
        from scientificlab.simulations import simulate_string_wave
        from scientificlab.visualization import plot_wave_snapshots

        w = simulate_string_wave(
            1.0, 100.0, lambda x: np.sin(np.pi * x), duration=0.01, nx=51
        )
        assert len(plot_wave_snapshots(w).axes[0].lines) == 5
        fig = plot_wave_snapshots(w, [0.0, 0.005])
        assert len(fig.axes[0].lines) == 2
        with pytest.raises(InvalidInputError):
            plot_wave_snapshots(w, [1.0])
        with pytest.raises(InvalidInputError):
            plot_wave_snapshots(w, [])


class TestStatisticsPlots:
    def test_histogram(self, tmp_path):
        from scientificlab.visualization import plot_histogram

        fig = plot_histogram(
            [1, 2, 2, 3, 3, 3, 4], bins=4, save_path=tmp_path / "h.png"
        )
        heights = [p.get_height() for p in fig.axes[0].patches]
        assert sum(heights) == 7
        assert (tmp_path / "h.png").stat().st_size > 0
        for bad in (0, True, 2.5):
            with pytest.raises(InvalidInputError):
                plot_histogram([1, 2, 3], bins=bad)
        with pytest.raises(InvalidInputError):
            plot_histogram([])

    def test_regression_plot(self):
        from scientificlab.visualization import plot_regression

        fig = plot_regression([1, 2, 3, 4], [2.1, 3.9, 6.2, 7.8])
        ax = fig.axes[0]
        assert len(ax.lines) == 2
        assert "R²" in ax.get_legend().get_texts()[1].get_text()
        with pytest.raises(InvalidInputError):
            plot_regression([1, 2], [1, 2])
        neg = plot_regression([1, 2, 3, 4], [-0.9, 1.1, 3.0, 5.1])
        assert " - " in neg.axes[0].get_legend().get_texts()[1].get_text()
