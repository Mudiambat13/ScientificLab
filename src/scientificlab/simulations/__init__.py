"""Numerical and analytic simulations of classical physical systems.

Every simulation states its method (``analytic`` or numeric), its model
assumptions and its validity limits in its docstring and in the ``notes`` of the
result. Results are plain data (NumPy arrays), independent of any plotting.
"""

from scientificlab.simulations._common import SimulationResult, WaveResult
from scientificlab.simulations.circuits import simulate_rc_circuit
from scientificlab.simulations.mechanics import simulate_free_fall, simulate_projectile
from scientificlab.simulations.oscillations import simulate_harmonic_oscillator
from scientificlab.simulations.thermal import cooling_time_to_reach, simulate_cooling
from scientificlab.simulations.waves import simulate_string_wave

__all__ = [
    "SimulationResult",
    "WaveResult",
    "cooling_time_to_reach",
    "simulate_cooling",
    "simulate_free_fall",
    "simulate_harmonic_oscillator",
    "simulate_projectile",
    "simulate_rc_circuit",
    "simulate_string_wave",
]
