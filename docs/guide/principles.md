# Principles

## Exact, numeric, verified

ScientificLab never blurs these notions, and each function states which one it
returns:

| Word | Meaning |
| --- | --- |
| **exact / symbolic** | SymPy objects, no floating-point approximation (`calculus`, `solve_ode`, `EquationSolver`). |
| **numeric** | NumPy / SciPy floats with a documented tolerance (`linear_algebra`, `numerical_methods`, `simulate_*` with a numeric method). |
| **complete** | The solution set is guaranteed complete. Only polynomial and rational equations qualify. |
| **verified** | Every solution was substituted back into the equation and the residual is negligible. |

The solver never prints "solution steps": it cannot justify them, so it does not
invent them.

## Units and conventions

- Core physics functions take plain floats in **SI units**. Wrappers in
  `scientificlab.units.physics` accept Pint quantities.
- Sign conventions are documented per function (for instance: work is positive
  when done *by* the gas; real images have positive distance in the lens equation).
- Temperatures are absolute (kelvin) unless a function says otherwise; a
  temperature below absolute zero is rejected.

## Errors

All library errors derive from `ScientificLabError`:
`InvalidInputError`, `NoSolutionError`, `InfiniteSolutionsError`,
`ExpressionParseError`, `PhysicalInconsistencyError`, `SolverError`,
`ConvergenceError`, `SingularMatrixError`, and `UnitError` with its children.
`IllConditionedWarning` is a warning, not an error. See the
[exceptions reference](../reference/utils.md).

## Security

Functions that take equations or expressions as **text** use SymPy's parser,
which relies on `eval`. Never pass untrusted text to them. See the
[security policy](../security.md).
