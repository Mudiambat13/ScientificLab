# Physics

Modules: `mechanics`, `electricity`, `thermodynamics`, `optics`, `waves`,
`fluid_mechanics`. All use SI units and validate inputs.

```python
from scientificlab.physics.mechanics import free_fall_time, weight
from scientificlab.physics.electricity import ohms_law
from scientificlab.physics.thermodynamics import ideal_gas, carnot_efficiency

print(round(free_fall_time(height=100), 3))  # 4.516 s
print(round(weight(mass=70), 3))  # 686.465 N
print(carnot_efficiency(500.0, 300.0))  # 0.4  (kelvin)
gas = ideal_gas(volume=0.0224, amount=1.0, temperature=273.15)
print(round(gas.pressure))  # 101388 Pa
```

Over-determined or under-determined calls raise `InvalidInputError`; a result
that contradicts physics (for instance an efficiency above the Carnot limit)
raises `PhysicalInconsistencyError`.
