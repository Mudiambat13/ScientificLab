"""Numerical integration: trapezoid, Simpson and adaptive quadrature."""

import math
import warnings
from dataclasses import dataclass

from scipy.integrate import IntegrationWarning, quad

from scientificlab.mathematics.numerical_methods._common import (
    ScalarFunction,
    evaluate,
    require_callable,
)
from scientificlab.utils.exceptions import ConvergenceError, InvalidInputError
from scientificlab.utils.validation import require_finite, require_positive


def _require_subdivisions(n: int, minimum: int, name: str = "n") -> int:
    if isinstance(n, bool) or not isinstance(n, int) or n < minimum:
        raise InvalidInputError(f"{name} must be an integer >= {minimum}, got {n!r}.")
    return n


def trapezoid_rule(
    function: ScalarFunction, a: float, b: float, n: int = 1000
) -> float:
    """Composite trapezoid rule with ``n`` equal subintervals.

    The error behaves like ``-(b - a) * h**2 * f''(xi) / 12`` with ``h = (b-a)/n``:
    it falls by 4 when ``n`` doubles, and the rule is exact for linear
    functions. Requires ``f`` to be finite and continuous on ``[a, b]``;
    singularities or rapid oscillations need much larger ``n`` or
    :func:`adaptive_integral`.

    Args:
        function: Integrand ``f(x)``.
        a: Lower bound (finite).
        b: Upper bound (finite); ``b < a`` gives the negated integral.
        n: Number of subintervals (>= 1).

    Raises:
        InvalidInputError: On invalid arguments.
        ConvergenceError: If ``f`` is not finite at a sample point.
    """
    require_callable(function, "function")
    lower = require_finite(a, "a")
    upper = require_finite(b, "b")
    count = _require_subdivisions(n, 1)
    h = (upper - lower) / count
    total = 0.5 * (evaluate(function, lower) + evaluate(function, upper))
    total += math.fsum(evaluate(function, lower + k * h) for k in range(1, count))
    return total * h


def simpson_rule(function: ScalarFunction, a: float, b: float, n: int = 1000) -> float:
    """Composite Simpson rule with ``n`` (even) equal subintervals.

    The error behaves like ``-(b - a) * h**4 * f''''(xi) / 180``: it falls by 16
    when ``n`` doubles, and the rule is exact for polynomials up to degree 3.
    Requires ``f`` to be finite and four times differentiable for the stated
    rate; otherwise convergence is slower.

    Args:
        function: Integrand ``f(x)``.
        a: Lower bound (finite).
        b: Upper bound (finite); ``b < a`` gives the negated integral.
        n: Number of subintervals; must be even and >= 2.

    Raises:
        InvalidInputError: If ``n`` is odd or less than 2, or arguments invalid.
        ConvergenceError: If ``f`` is not finite at a sample point.
    """
    require_callable(function, "function")
    lower = require_finite(a, "a")
    upper = require_finite(b, "b")
    count = _require_subdivisions(n, 2)
    if count % 2 != 0:
        raise InvalidInputError(f"n must be even for Simpson's rule, got {n}.")
    h = (upper - lower) / count
    odd = math.fsum(evaluate(function, lower + k * h) for k in range(1, count, 2))
    even = math.fsum(evaluate(function, lower + k * h) for k in range(2, count, 2))
    ends = evaluate(function, lower) + evaluate(function, upper)
    return (ends + 4.0 * odd + 2.0 * even) * h / 3.0


@dataclass(frozen=True)
class IntegrationResult:
    """Result of an adaptive integration.

    Attributes:
        value: Approximate integral.
        error_estimate: Estimated absolute error (an estimate, not a bound).
    """

    value: float
    error_estimate: float


def adaptive_integral(
    function: ScalarFunction,
    a: float,
    b: float,
    *,
    abs_tol: float = 1.49e-8,
    rel_tol: float = 1.49e-8,
) -> IntegrationResult:
    """Adaptive Gauss-Kronrod quadrature (QUADPACK via ``scipy.integrate.quad``).

    Handles smooth integrands, mild singularities and infinite bounds
    (``float("inf")``). The returned error is an estimate of the absolute error.
    Integrals that diverge, or integrands with strong unresolved singularities
    or oscillations, raise instead of returning an unreliable number.

    Args:
        function: Integrand ``f(x)``.
        a: Lower bound (may be ``-inf``).
        b: Upper bound (may be ``inf``).
        abs_tol: Requested absolute accuracy.
        rel_tol: Requested relative accuracy.

    Raises:
        InvalidInputError: On NaN bounds or invalid tolerances.
        ConvergenceError: If QUADPACK reports a problem (non-convergence,
            roundoff, divergence) or the result is not finite.
    """
    require_callable(function, "function")
    if math.isnan(a) or math.isnan(b):
        raise InvalidInputError("Integration bounds must not be NaN.")
    absolute = require_positive(abs_tol, "abs_tol")
    relative = require_positive(rel_tol, "rel_tol")

    def integrand(x: float) -> float:
        return evaluate(function, x)

    with warnings.catch_warnings():
        warnings.simplefilter("error", IntegrationWarning)
        try:
            value, error = quad(integrand, a, b, epsabs=absolute, epsrel=relative)
        except IntegrationWarning as warning:
            raise ConvergenceError(
                f"Adaptive integration failed: {warning}"
            ) from warning
    if not (math.isfinite(value) and math.isfinite(error)):
        raise ConvergenceError("Adaptive integration produced a non-finite result.")
    return IntegrationResult(value=float(value), error_estimate=float(error))
