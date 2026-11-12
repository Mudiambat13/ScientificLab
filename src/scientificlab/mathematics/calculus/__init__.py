"""Calculus: derivatives, integrals, limits, Taylor series and series sums.

All functions return exact symbolic results (SymPy), never numerical
approximations. For numerical differentiation and integration see
``scientificlab.mathematics.numerical_methods``.
"""

from scientificlab.mathematics.calculus.differentiation import (
    differentiate,
    partial_derivative,
)
from scientificlab.mathematics.calculus.integration import (
    definite_integral,
    indefinite_integral,
)
from scientificlab.mathematics.calculus.limits_series import (
    limit,
    series_sum,
    taylor_series,
)

__all__ = [
    "definite_integral",
    "differentiate",
    "indefinite_integral",
    "limit",
    "partial_derivative",
    "series_sum",
    "taylor_series",
]
