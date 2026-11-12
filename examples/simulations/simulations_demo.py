"""Simulations: projectile, free fall, oscillator, RC, cooling, string wave.

Run: python examples/simulations/simulations_demo.py   (writes PNG files here)
"""

from pathlib import Path

import numpy as np

from scientificlab.simulations import (
    cooling_time_to_reach,
    simulate_cooling,
    simulate_free_fall,
    simulate_harmonic_oscillator,
    simulate_projectile,
    simulate_rc_circuit,
    simulate_string_wave,
)
from scientificlab.visualization import (
    plot_simulation,
    plot_trajectory,
    plot_wave_snapshots,
)

out = Path(__file__).parent

vacuum = simulate_projectile(20, 45)
air = simulate_projectile(20, 45, drag=0.01)
print(f"range, vacuum: {vacuum.summary['range']:.2f} m")
print(f"range, with drag: {air.summary['range']:.2f} m")
plot_trajectory(air, save_path=out / "projectile.png")

fall = simulate_free_fall(1000, drag=0.01)
print(f"terminal speed {fall.summary['terminal_speed']:.1f} m/s")

osc = simulate_harmonic_oscillator(1.0, 100.0, damping=2.0)
print(osc.summary["regime"], f"zeta={osc.summary['damping_ratio']:.2f}")
plot_simulation(osc, save_path=out / "oscillator.png")

rc = simulate_rc_circuit(5.0, 1000.0, 1e-3)
print(f"tau = {rc.summary['tau']} s")

cool = simulate_cooling(363.15, 293.15, 0.01)
print(f"to 50 C: {cooling_time_to_reach(363.15, 293.15, 0.01, 323.15):.0f} s")

wave = simulate_string_wave(
    1.0, 100.0, lambda x: np.exp(-(((x - 0.3) / 0.05) ** 2)), duration=0.02
)
plot_wave_snapshots(wave, save_path=out / "wave.png")
print("figures written to", out)
