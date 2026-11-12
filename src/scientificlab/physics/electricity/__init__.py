"""Electricity: Ohm's law, power, resistor networks, capacitors and RC circuits."""

from scientificlab.physics.electricity.capacitors import (
    capacitance_from_charge,
    capacitor_charge,
    capacitor_energy,
    charge_from_current,
    parallel_plate_capacitance,
)
from scientificlab.physics.electricity.circuits import (
    OhmsLawResult,
    electric_energy,
    electric_power,
    ohms_law,
    parallel_resistance,
    series_resistance,
)
from scientificlab.physics.electricity.rc_circuit import (
    rc_current,
    rc_discharge_voltage,
    rc_time_constant,
    rc_time_to_voltage,
    rc_voltage,
)

__all__ = [
    "OhmsLawResult",
    "capacitance_from_charge",
    "capacitor_charge",
    "capacitor_energy",
    "charge_from_current",
    "electric_energy",
    "electric_power",
    "ohms_law",
    "parallel_plate_capacitance",
    "parallel_resistance",
    "rc_current",
    "rc_discharge_voltage",
    "rc_time_constant",
    "rc_time_to_voltage",
    "rc_voltage",
    "series_resistance",
]
