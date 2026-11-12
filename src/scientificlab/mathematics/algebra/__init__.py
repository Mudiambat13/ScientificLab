"""Algebra: equations, polynomials, symbolic expressions, and systems."""

from scientificlab.mathematics.algebra.equations import (
    LinearSolution,
    QuadraticSolution,
    RootNature,
    solve_linear,
    solve_quadratic,
)
from scientificlab.mathematics.algebra.expressions import (
    expand_expression,
    factor_expression,
    parse_equation,
    parse_expression,
    parse_symbol,
    simplify_expression,
)
from scientificlab.mathematics.algebra.polynomials import (
    PolynomialRoots,
    polynomial_roots,
)
from scientificlab.mathematics.algebra.systems import SystemSolution, solve_system

__all__ = [
    "LinearSolution",
    "PolynomialRoots",
    "QuadraticSolution",
    "RootNature",
    "SystemSolution",
    "expand_expression",
    "factor_expression",
    "parse_equation",
    "parse_expression",
    "parse_symbol",
    "polynomial_roots",
    "simplify_expression",
    "solve_linear",
    "solve_quadratic",
    "solve_system",
]
