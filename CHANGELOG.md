# Changelog

All notable changes are documented here. The project follows
[Semantic Versioning](https://semver.org/) and the
[Keep a Changelog](https://keepachangelog.com/) format.

## [0.1.0] - 2026-11-08

Phase 1 - Foundation.

### Added
- Project architecture (`src/` layout), `pyproject.toml` (Hatchling), MIT license.
- Quality tooling: Pytest, Ruff (lint + format), MyPy (strict), GitHub Actions CI.
- `scientificlab.utils`: explicit exception hierarchy and input validation.
- `scientificlab.constants.STANDARD_GRAVITY` (exact, 9.80665 m/s**2).
- `scientificlab.mathematics.algebra`: `solve_linear`, `solve_quadratic`
  (stable, real and complex), `polynomial_roots`, `expand_expression`,
  `factor_expression`, `simplify_expression`, `solve_system`.
- `scientificlab.physics.mechanics`: kinematics, dynamics, energy and work
  (SI units).
- Documentation: README, CONTRIBUTING, CLAUDE.md.

### Notes
- Project named **ScientificLab** (the specification's working name was PhysMathLab).
- Only NumPy and SymPy are declared as dependencies; the other libraries of the
  specification are introduced in the phases that use them.
