"""Mechanics examples (SI units). Run: python examples/physics/mechanics_demo.py"""

from scientificlab.physics.mechanics import (
    free_fall_impact_speed,
    free_fall_time,
    kinetic_energy,
    potential_energy,
    weight,
)

print(f"{kinetic_energy(mass=10, velocity=5)} J")  # 125.0 J
print(f"{weight(70):.1f} N")  # weight of a 70 kg person
height = 20.0
print(f"fall time from {height} m: {free_fall_time(height):.3f} s")
speed = free_fall_impact_speed(height)
print(f"impact speed: {speed:.3f} m/s")
# Energy conservation: E_k at impact == E_p at the top.
print(kinetic_energy(1.0, speed), potential_energy(1.0, height))
