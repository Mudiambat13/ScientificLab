"""Electricity, thermodynamics, optics, waves and fluids (SI units).

Run: python examples/physics/physics_demo.py
"""

from scientificlab.physics.electricity import ohms_law, rc_time_to_voltage, rc_voltage
from scientificlab.physics.fluid_mechanics import (
    bernoulli_pressure,
    continuity_velocity,
    hydrostatic_pressure,
)
from scientificlab.physics.optics import snell_refraction, thin_lens_equation
from scientificlab.physics.thermodynamics import (
    heat_energy,
    ideal_gas,
    thermal_efficiency,
)
from scientificlab.physics.waves import doppler_frequency, wave_relation

result = ohms_law(voltage=12, resistance=4)
print(result.current, "A,", result.power, "W")  # 3.0 A, 36.0 W

tau = 1_000.0 * 1e-6  # R = 1 kOhm, C = 1 uF
print(rc_voltage(5.0, 1_000.0, 1e-6, tau))  # 63.2 % of 5 V = 3.16 V
print(rc_time_to_voltage(5.0, 1_000.0, 1e-6, 4.5))  # time to 90 %: 2.30 ms

print(heat_energy(mass=2, specific_heat=4186, temperature_change=10), "J")  # 83720.0
print(ideal_gas(pressure=101_325, amount=1, temperature=273.15).volume)  # 0.02241 m^3
print(thermal_efficiency(400, 1000, hot_temperature=500, cold_temperature=300))  # 0.4

glass = snell_refraction(1.5, 1.0, 45.0)  # glass -> air
print("total internal reflection:", glass.total_internal_reflection)  # True
image = thin_lens_equation(focal_length=0.1, object_distance=0.3)
print(image.image_distance, image.magnification)  # 0.15 m, -0.5 (real, inverted)

print(wave_relation(speed=343, frequency=440).wavelength)  # 0.7795 m
print(doppler_frequency(440, 343, source_velocity=30))  # 482.2 Hz

v2 = continuity_velocity(0.1, 2.0, 0.05)  # pipe narrows: 2 -> 4 m/s
print(bernoulli_pressure(1000.0, 200_000, 2.0, 0.0, v2, 0.0))  # 194000.0 Pa
print(hydrostatic_pressure(1000.0, 10.0, surface_pressure=101_325))  # 199391.5 Pa
