# Equation solver

```python
from scientificlab.solvers import EquationSolver, solve_equation

print(solve_equation("x**2 - 4 = 0").solutions)  # (-2, 2)

result = solve_equation("exp(x) = 2")
print(result.solutions, result.complete)  # (log(2),) False

numeric = solve_equation("cos(x) = x", initial_guess=0.7)
print(round(numeric.values[0], 12), numeric.method)  # 0.739085133215 numeric

system = EquationSolver().solve_system(["x + y = 3", "x - y = 1"])
print(system.solutions)  # ({'x': 2, 'y': 1},)
```

Reading a `SolveResult`:

| Field | Meaning |
| --- | --- |
| `exact` | Symbolic result with no floating-point approximation (false if you passed floats). |
| `complete` | The solution set is guaranteed complete (polynomial and rational equations only). |
| `verified` | Each solution was substituted back and the residual is negligible. |
| `warnings` | Plain-language caveats. |

Periodic or infinite solution sets raise `InfiniteSolutionsError`; equations
without a closed form raise `SolverError` unless you pass `initial_guess`, which
runs a numerical search for **one** root (`exact=False`, `complete=False`).
