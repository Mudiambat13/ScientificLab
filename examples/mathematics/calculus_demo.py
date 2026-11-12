"""Calculus, ODE and numerical-methods examples.

Run: python examples/mathematics/calculus_demo.py
"""

import math

from scientificlab.mathematics.calculus import (
    definite_integral,
    differentiate,
    limit,
    taylor_series,
)
from scientificlab.mathematics.differential_equations import solve_ode
from scientificlab.mathematics.linear_algebra import eigen_decomposition
from scientificlab.mathematics.numerical_methods import (
    adaptive_integral,
    bisection,
    newton_raphson,
)

print(differentiate("x**3 + 2*x**2 + 5", variable="x"))  # 3*x**2 + 4*x
print(definite_integral("sin(x)", "x", 0, "pi"))  # 2 (exact)
print(limit("sin(x)/x", "x", 0))  # 1
print(taylor_series("sin(x)", "x", 0, 5))  # x**5/120 - x**3/6 + x

ode = solve_ode("y'' + y = 0", initial_conditions={"y(0)": 0, "y'(0)": 1})
print(ode.solution, "verified:", ode.verified)  # sin(x) verified: True

print(newton_raphson(lambda x: x * x - 2, 1.0, lambda x: 2 * x).root)
print(bisection(lambda x: math.cos(x) - x, 0.0, 1.0).root)  # 0.739085...
print(adaptive_integral(lambda x: x**-0.5, 0.0, 1.0).value)  # ~2.0 (numerical)
print(eigen_decomposition([[2, 1], [1, 2]]).values)  # [1. 3.]
