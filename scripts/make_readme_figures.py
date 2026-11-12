"""Regenerate the illustrations used in the README (docs/assets/*.png).

Every curve is computed by ScientificLab itself; nothing is drawn by hand.
Run from the repository root:  python scripts/make_readme_figures.py
"""

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from scientificlab.mathematics.statistics import linear_regression
from scientificlab.simulations import (
    simulate_harmonic_oscillator,
    simulate_projectile,
    simulate_string_wave,
)
from scientificlab.visualization import plot_functions

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets"
BLUE, ORANGE, GREEN, RED, GREY = "#1f5fd1", "#e8710a", "#17803d", "#c62828", "#6b7280"


def fresh(width: float, height: float, nrows: int = 1, ncols: int = 1):
    fig = Figure(
        figsize=(width, height), dpi=150, facecolor="white", layout="constrained"
    )
    axes = fig.subplots(nrows, ncols, squeeze=False)
    for ax in axes.flat:
        ax.set_facecolor("white")
        ax.grid(True, alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    return fig, axes


def save(fig: Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=150, facecolor="white")
    print("wrote", OUT / name)


def projectile() -> None:
    fig, axes = fresh(10, 4.2, 1, 2)
    left, right = axes[0]
    colors = ["#9fc2ff", "#6b9cff", BLUE, "#1746a0", "#0f2f6b"]
    for angle, color in zip((15, 30, 45, 60, 75), colors, strict=True):
        run = simulate_projectile(30, angle)
        left.plot(
            run.series["x (m)"],
            run.series["y (m)"],
            color=color,
            lw=2,
            label=f"{angle}°",
        )
    left.set(title="Projectile in vacuum, 30 m/s", xlabel="x (m)", ylabel="y (m)")
    left.legend(title="launch angle", frameon=False)
    left.set_aspect("equal", adjustable="datalim")
    vac = simulate_projectile(30, 45)
    for drag, color, label in (
        (0.0, BLUE, "vacuum"),
        (0.005, ORANGE, "drag c = 0.005 /m"),
        (0.02, RED, "drag c = 0.02 /m"),
    ):
        run = simulate_projectile(30, 45, drag=drag)
        right.plot(
            run.series["x (m)"],
            run.series["y (m)"],
            color=color,
            lw=2,
            label=f"{label}  (range {run.summary['range']:.0f} m)",
        )
    right.set(title="Effect of quadratic air drag, 45°", xlabel="x (m)", ylabel="y (m)")
    right.legend(frameon=False)
    right.set_aspect("equal", adjustable="datalim")
    assert vac.summary["range"] > 90
    save(fig, "projectile.png")


def oscillator() -> None:
    fig, axes = fresh(6.2, 4.2)
    ax = axes[0, 0]
    for damping, color, label in (
        (2.0, BLUE, "underdamped (ζ = 0.1)"),
        (20.0, GREEN, "critically damped (ζ = 1)"),
        (60.0, ORANGE, "overdamped (ζ = 3)"),
    ):
        run = simulate_harmonic_oscillator(1.0, 100.0, damping=damping, duration=2.5)
        ax.plot(run.time, run.series["x (m)"], color=color, lw=2, label=label)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set(
        title="Damped oscillator, m = 1 kg, k = 100 N/m",
        xlabel="time (s)",
        ylabel="position (m)",
    )
    ax.legend(frameon=False)
    save(fig, "oscillator.png")


def wave() -> None:
    def pulse(x):
        return np.exp(-(((x - 0.3) / 0.05) ** 2))

    res = simulate_string_wave(1.0, 100.0, pulse, duration=0.02, nx=401, courant=1.0)
    fig, axes = fresh(6.2, 4.2)
    ax = axes[0, 0]
    shades = ["#0f2f6b", GREEN, ORANGE, RED]
    for t_ms, color in zip((0, 2, 5, 8), shades, strict=True):
        k = int(np.argmin(np.abs(res.time - t_ms * 1e-3)))
        ax.plot(res.x, res.displacement[k], color=color, lw=2, label=f"t = {t_ms} ms")
    ax.set(
        title="Pulse on a string with fixed ends (reflection flips the sign)",
        xlabel="position (m)",
        ylabel="displacement (m)",
    )
    ax.title.set_fontsize(10)
    ax.legend(frameon=False, ncols=2)
    save(fig, "wave.png")


def functions() -> None:
    fig = plot_functions(
        ["sin(x)/x", "1/x", "tan(x)"],
        (-10, 10),
        y_limits=(-3, 3),
        title="Poles are left as gaps, never joined",
    )
    fig.set_facecolor("white")
    save(fig, "functions.png")


def regression() -> None:
    rng = np.random.default_rng(2026)
    x = np.linspace(0, 10, 25)
    y = 1.8 * x + 4.0 + rng.normal(0, 1.5, x.size)
    fit = linear_regression(x, y)
    fig, axes = fresh(6.2, 4.2)
    ax = axes[0, 0]
    ax.plot(x, y, "o", color=BLUE, label="measurements")
    ax.plot(
        x,
        fit.predict(x),
        color=ORANGE,
        lw=2.5,
        label=f"y = {fit.slope:.3f}x + {fit.intercept:.2f}   R² = {fit.r_squared:.3f}",
    )
    ax.set(
        title=f"Least squares: slope {fit.slope:.3f} ± {fit.slope_stderr:.3f}",
        xlabel="x",
        ylabel="y",
    )
    ax.legend(frameon=False)
    save(fig, "regression.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for build in (projectile, oscillator, wave, functions, regression):
        build()
