"""Thermodynamics: heat, temperature, ideal gases, work and engine efficiency."""

from scientificlab.physics.thermodynamics.efficiency import (
    carnot_efficiency,
    efficiency_from_heats,
    thermal_efficiency,
)
from scientificlab.physics.thermodynamics.gases import (
    IdealGasState,
    adiabatic_pressure,
    adiabatic_work,
    ideal_gas,
    isobaric_work,
    isothermal_work,
)
from scientificlab.physics.thermodynamics.heat import (
    convert_temperature,
    heat_energy,
    latent_heat,
)

__all__ = [
    "IdealGasState",
    "adiabatic_pressure",
    "adiabatic_work",
    "carnot_efficiency",
    "convert_temperature",
    "efficiency_from_heats",
    "heat_energy",
    "ideal_gas",
    "isobaric_work",
    "isothermal_work",
    "latent_heat",
    "thermal_efficiency",
]
