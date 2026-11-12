# Mathematics

```python
from scientificlab.mathematics.algebra import solve_quadratic, solve_system
from scientificlab.mathematics.calculus import differentiate, definite_integral
from scientificlab.mathematics.linear_algebra import determinant, solve_linear_system
from scientificlab.mathematics.numerical_methods import newton_raphson

print(solve_quadratic(1, 2, 5).roots)  # ((-1-2j), (-1+2j)): complex roots shown
print(differentiate("x**3", "x"))  # 3*x**2   (exact, symbolic)
print(definite_integral("sin(x)", "x", 0, "pi"))  # 2       (exact)
print(round(determinant([[1, 2], [3, 4]]), 12))  # -2.0   (numeric)

root = newton_raphson(lambda x: x**2 - 2, 1.0)
print(round(root.root, 12), root.residual < 1e-12)  # 1.414213562373 True
```

- **Algebra, calculus, ODEs (symbolic)** return SymPy results, exact.
- **Linear algebra, numerical methods** return floats; ill-conditioned systems
  raise `IllConditionedWarning`, singular ones `SingularMatrixError`.
- `solve_ode` verifies every symbolic solution by substitution; numeric ODEs use
  SciPy (`solve_ode_numeric`).

## Statistics

```python
from scientificlab.mathematics.statistics import (
    confidence_interval_mean,
    describe,
    linear_regression,
    normal_cdf,
    normal_quantile,
    standard_deviation,
)

data = [2, 4, 4, 4, 5, 5, 7, 9]
print(standard_deviation(data, ddof=0))  # 2.0   (population)
print(round(standard_deviation(data), 6))  # 2.13809   (sample, ddof=1 by default)
summary = describe(data)
print(summary.median, summary.q1, summary.q3)  # 4.5 4.0 5.5

fit = linear_regression([1, 2, 3, 4, 5], [2.1, 3.9, 6.2, 7.8, 10.1])
print(round(fit.slope, 4), round(fit.intercept, 4), round(fit.r_squared, 5))
# 1.99 0.05 0.99731
print(fit.predict([6]))  # [11.99]

ci = confidence_interval_mean(data)  # Student t, 95 % by default
print(round(ci.lower, 3), round(ci.upper, 3))  # 3.213 6.787
print(round(normal_cdf(1.96), 4), round(normal_quantile(0.975), 4))  # 0.975 1.96
```

Sample statistics use `ddof=1` unless stated; quantiles follow the common
"type 7" definition (NumPy's default). Regression needs at least three points,
and constant data (no variance) raises `InvalidInputError` instead of returning a
meaningless fit.

See the [API reference](../reference/mathematics.md) for every function.
