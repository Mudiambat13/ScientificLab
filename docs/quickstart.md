# Quick start

Every example on this page is executed by the test suite.

## Algebra

```python
from scientificlab.mathematics.algebra import solve_quadratic

solution = solve_quadratic(a=1, b=-5, c=6)
print(solution.roots)  # (2.0, 3.0)
```

## Physics (plain floats, SI units)

```python
from scientificlab.physics.mechanics import kinetic_energy

print(kinetic_energy(mass=10, velocity=5))  # 125.0 (joules)
```

## Units

```python
from scientificlab.units import Q_, convert
from scientificlab.units import physics as up

print(convert(100, "km/h", "m/s"))  # 27.77777777777778 m/s
print(up.kinetic_energy(Q_(2, "g"), Q_(36, "km/h")))  # 0.1 J
```

## Equations

```python
from scientificlab.solvers import solve_equation

result = solve_equation("v = v0 + a*t", "t", values={"v": 20, "v0": 0, "a": 2})
print(result.solutions, result.exact, result.complete, result.verified)
# (10,) True True True
```

## Simulation and plot

```python
from scientificlab.simulations import simulate_projectile
from scientificlab.visualization import plot_trajectory

shot = simulate_projectile(speed=20, angle_degrees=45, drag=0.01)
print(round(shot.summary["range"], 2))  # 31.33 (metres, with air drag)
figure = plot_trajectory(shot)  # a Matplotlib Figure
figure.savefig("trajectory.png")  # or .svg / .pdf
```

## Errors are explicit

```python
from scientificlab.utils.exceptions import IncompatibleUnitsError
from scientificlab.units import convert

try:
    convert(1, "m", "s")
except IncompatibleUnitsError as error:
    print(error)  # Cannot convert m to s: different dimensions.
```
