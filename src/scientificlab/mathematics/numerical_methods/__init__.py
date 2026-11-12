"""Numerical methods: root finding, interpolation, quadrature, finite differences.

Every method documents its parameters, convergence conditions and limits, and
raises ``ConvergenceError`` rather than returning an unreliable value.
"""

from scientificlab.mathematics.numerical_methods.differentiation import (
    numerical_derivative,
)
from scientificlab.mathematics.numerical_methods.interpolation import (
    PolynomialInterpolant,
    linear_interpolation,
    polynomial_interpolation,
)
from scientificlab.mathematics.numerical_methods.quadrature import (
    IntegrationResult,
    adaptive_integral,
    simpson_rule,
    trapezoid_rule,
)
from scientificlab.mathematics.numerical_methods.root_finding import (
    RootResult,
    bisection,
    newton_raphson,
    secant,
)

__all__ = [
    "IntegrationResult",
    "PolynomialInterpolant",
    "RootResult",
    "adaptive_integral",
    "bisection",
    "linear_interpolation",
    "newton_raphson",
    "numerical_derivative",
    "polynomial_interpolation",
    "secant",
    "simpson_rule",
    "trapezoid_rule",
]
