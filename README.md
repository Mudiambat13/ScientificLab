# ScientificLab

**Mathematical & Physics Computing Toolkit**

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](CHANGELOG.md)

> ScientificLab is an open-source Python library for mathematical computation, equation solving, numerical analysis, physics calculations, scientific simulations, and data visualization.

ScientificLab favours **scientific accuracy over feature count**: every function performs a real calculation, validates its inputs, documents its units, sign conventions and limits, and is tested against known reference values. Inconsistent results are never returned silently.

> **Status: Phase 1 (Foundation) - v0.1.0.** Algebra and classical mechanics are implemented. The remaining modules are planned (see [Roadmap](#roadmap)); their packages exist but contain no code yet.
> The PyPI and CI badges will be added when the package is published (Phase 6).

## Features

**Mathematics - algebra** (`scientificlab.mathematics.algebra`)

| Function | Description |
| --- | --- |
| `solve_linear(a, b)` | Solves `a*x + b = 0`; reports no / infinitely many solutions explicitly |
| `solve_quadratic(a, b, c)` | Real and complex roots, numerically stable, with discriminant and root nature |
| `polynomial_roots(coefficients)` | Roots of any polynomial (numerical, via companion matrix) |
| `expand_expression`, `factor_expression`, `simplify_expression` | Exact symbolic manipulation (SymPy) |
| `solve_system(equations)` | Exact solutions of linear and nonlinear systems (SymPy) |

**Physics - classical mechanics** (`scientificlab.physics.mechanics`, SI units)

- Kinematics: distance, average speed/velocity, numerical instantaneous velocity, acceleration, uniform and uniformly accelerated motion, free fall, uniform circular motion
- Dynamics: Newton's second law, weight, friction, centripetal force, momentum
- Energy and work: kinetic, potential and mechanical energy, work of a force, power

## Installation

```bash
git clone <repository-url>
cd ScientificLab
pip install -e ".[dev]"
```

Requires Python 3.11+. Runtime dependencies: NumPy and SymPy.

## Quick start

```python
from scientificlab.mathematics.algebra import solve_quadratic
from scientificlab.physics.mechanics import kinetic_energy

result = solve_quadratic(a=1, b=-5, c=6)
print(result.roots)  # (2.0, 3.0)

result = solve_quadratic(a=1, b=2, c=5)
print(result.roots)  # ((-1-2j), (-1+2j))
print(result.nature)  # complex_conjugate

energy = kinetic_energy(mass=10, velocity=5)
print(energy)  # 125.0  (joules)
```

```python
from scientificlab.mathematics.algebra import factor_expression, solve_system

print(factor_expression("x**2 - 1"))  # (x - 1)*(x + 1)
print(solve_system(["x + y = 3", "x - y = 1"]).solutions)
# ({'x': 2, 'y': 1},)
```

Physics functions take and return plain floats in **SI units** (documented in each docstring). Automatic unit handling with Pint arrives in Phase 4. Invalid inputs raise explicit exceptions from `scientificlab.utils.exceptions`.

## Architecture

```
src/scientificlab/
├── mathematics/   algebra (done) | calculus | linear_algebra | differential_equations
│                  numerical_methods | statistics
├── physics/       mechanics (done) | electricity | thermodynamics | optics | waves
│                  fluid_mechanics
├── units/  constants/  solvers/  visualization/  utils/
tests/  examples/  notebooks/  docs/  scripts/
```

Each concern is separated: validation (`utils.validation`), computation (module functions), and - from Phase 4 - display (`visualization`). New modules plug into this layout without restructuring.

## Technologies

NumPy (numerics), SymPy (exact symbolic computation), Pytest, Ruff (lint + format), MyPy (strict typing), GitHub Actions (CI). SciPy, Matplotlib, Pint, Pandas and Pydantic will be added only when the phase that needs them lands.

## Tests and quality

```bash
pytest              # run the test suite
ruff check .        # lint
ruff format --check .
mypy                # strict static typing
```

Tests use known reference values (Vieta's relations, free fall of `g/2` metres in one second, energy conservation, the work-energy theorem) and cover invalid inputs, zero/negative values, complex roots and numerical precision.

## Roadmap

- [x] **Phase 1 - Foundation:** architecture, tooling, algebra, mechanics
- [ ] **Phase 2 - Mathematics core:** calculus, linear algebra, ODEs, numerical methods
- [ ] **Phase 3 - Physics core:** electricity, thermodynamics, optics, waves, fluids
- [ ] **Phase 4 - Scientific tools:** CODATA constants, Pint units, equation-solving engine, plots
- [ ] **Phase 5 - Simulations:** projectile, oscillator, RC circuit, cooling, wave propagation
- [ ] **Phase 6 - Publication:** documentation site, CI, PyPI release

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issues and pull requests are welcome.

## License

[MIT](LICENSE)
