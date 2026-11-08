"""Algebra examples. Run: python examples/mathematics/algebra_demo.py"""

from scientificlab.mathematics.algebra import (
    factor_expression,
    polynomial_roots,
    solve_quadratic,
    solve_system,
)

print(solve_quadratic(a=1, b=-5, c=6).roots)  # (2.0, 3.0)
print(solve_quadratic(a=1, b=2, c=5).roots)  # complex pair -1 -/+ 2j
# Numerical roots: approximately (1.0, 2.0, 3.0), within ~1e-14.
print(polynomial_roots([1, -6, 11, -6]).roots)
print(factor_expression("x**2 - 1"))  # (x - 1)*(x + 1)
print(solve_system(["x + y = 3", "x - y = 1"]).solutions)  # x=2, y=1
