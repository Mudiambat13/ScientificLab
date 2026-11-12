# Simulations

| Function | Method |
| --- | --- |
| `simulate_projectile` | exact in vacuum; RK45 (rtol 1e-10) with quadratic drag |
| `simulate_free_fall` | exact (`tanh` solution with drag) |
| `simulate_harmonic_oscillator` | exact, all damping regimes |
| `simulate_rc_circuit` | exact, with energy balance |
| `simulate_cooling`, `cooling_time_to_reach` | Newton's law, exact |
| `simulate_string_wave` | explicit finite differences, Courant number checked |

```python
import numpy as np
from scientificlab.simulations import simulate_harmonic_oscillator, simulate_string_wave

osc = simulate_harmonic_oscillator(mass=1.0, stiffness=100.0, damping=2.0)
print(osc.summary["regime"], round(osc.summary["damping_ratio"], 2))  # underdamped 0.1

wave = simulate_string_wave(
    length=1.0,
    wave_speed=100.0,
    initial_displacement=lambda x: np.sin(3 * np.pi * x),
    duration=0.02,
    nx=401,
    courant=1.0,
)
print(wave.displacement.shape, wave.summary["courant"])  # (801, 401) 1.0
```

Each result carries its `method`, `notes` (model assumptions) and a `summary` of
derived quantities. The wave array has one row per time step. A Courant number above 1 is refused because the scheme would
be unstable.
