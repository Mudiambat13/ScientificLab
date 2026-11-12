"""Iterative root-finding: Newton-Raphson, bisection and secant methods.

All methods return floating-point approximations of a root of ``f`` and raise
:class:`ConvergenceError` instead of returning a value when they fail.
"""

import math
from dataclasses import dataclass

from scientificlab.mathematics.numerical_methods._common import (
    ScalarFunction,
    evaluate,
    require_callable,
    require_max_iterations,
)
from scientificlab.mathematics.numerical_methods.differentiation import (
    numerical_derivative,
)
from scientificlab.utils.exceptions import ConvergenceError, InvalidInputError
from scientificlab.utils.validation import require_finite, require_positive


@dataclass(frozen=True)
class RootResult:
    """Outcome of a successful root search.

    Attributes:
        root: Approximate root ``x*``.
        iterations: Number of iterations performed.
        residual: ``abs(f(root))``. Small residual and small step are different
            things: for ill-conditioned or multiple roots check both.
    """

    root: float
    iterations: int
    residual: float


def newton_raphson(
    function: ScalarFunction,
    x0: float,
    derivative: ScalarFunction | None = None,
    *,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> RootResult:
    """Find a root with Newton's method: ``x_{k+1} = x_k - f(x_k) / f'(x_k)``.

    Convergence conditions and limits:
        * Converges **quadratically** to a simple root (``f' != 0`` there) when
          ``f`` is twice continuously differentiable and ``x0`` is close enough.
        * Converges only **linearly** to a multiple root.
        * May diverge, cycle or jump to a different root if ``x0`` is far from
          the root or ``f'`` is near zero along the way.
        * No bracketing is used, so the root found is not guaranteed to be the
          one nearest to ``x0``. Use :func:`bisection` when a sign-changing
          interval is known.

    Args:
        function: ``f(x)``.
        x0: Starting guess.
        derivative: ``f'(x)``. If ``None``, it is approximated by central
            differences (slightly reducing the convergence rate).
        tol: Stops when ``|x_{k+1} - x_k| <= tol * (1 + |x_{k+1}|)``.
        max_iter: Maximum number of iterations.

    Raises:
        InvalidInputError: On invalid arguments.
        ConvergenceError: If ``f'(x_k) == 0``, an iterate is non-finite, or
            ``max_iter`` is exhausted.
    """
    require_callable(function, "function")
    if derivative is not None:
        require_callable(derivative, "derivative")
    x = require_finite(x0, "x0")
    tolerance = require_positive(tol, "tol")
    limit = require_max_iterations(max_iter)

    for iteration in range(1, limit + 1):
        fx = evaluate(function, x)
        if fx == 0.0:
            return RootResult(x, iteration - 1, 0.0)
        slope = (
            evaluate(derivative, x)
            if derivative is not None
            else numerical_derivative(function, x)
        )
        if slope == 0.0:
            raise ConvergenceError(
                f"Zero derivative at x = {x!r}; Newton step undefined."
            )
        x_next = x - fx / slope
        if not math.isfinite(x_next):
            raise ConvergenceError("Newton iteration diverged to a non-finite value.")
        if abs(x_next - x) <= tolerance * (1.0 + abs(x_next)):
            return RootResult(x_next, iteration, abs(evaluate(function, x_next)))
        x = x_next
    raise ConvergenceError(f"Newton-Raphson did not converge in {limit} iterations.")


def bisection(
    function: ScalarFunction,
    a: float,
    b: float,
    *,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> RootResult:
    """Find a root in ``[a, b]`` by repeated interval halving.

    Convergence conditions and limits:
        * ``f`` must be continuous on ``[a, b]`` and ``f(a)``, ``f(b)`` must
          have opposite signs (or one of them be zero). Convergence is then
          **guaranteed**, but only linear: one binary digit per iteration
          (about ``log2((b - a) / tol)`` iterations).
        * If several roots lie in the interval, one of them is found.
        * A sign change can also come from a pole (e.g. ``1/x`` on ``[-1, 1]``);
          the returned point is then not a root, so check ``residual``.
        * A ``tol`` below the floating-point spacing around the root cannot be
          met; the search then stops at machine resolution.

    Args:
        function: ``f(x)``.
        a: One end of the interval.
        b: The other end.
        tol: Stops when the half-width of the interval is ``<= tol``.
        max_iter: Maximum number of halvings.

    Raises:
        InvalidInputError: If ``a == b`` or ``f(a)`` and ``f(b)`` have the same
            sign, or on other invalid arguments.
        ConvergenceError: If ``max_iter`` is exhausted.
    """
    require_callable(function, "function")
    low = require_finite(a, "a")
    high = require_finite(b, "b")
    tolerance = require_positive(tol, "tol")
    limit = require_max_iterations(max_iter)
    if low == high:
        raise InvalidInputError("a and b must be different.")
    if low > high:
        low, high = high, low

    f_low = evaluate(function, low)
    f_high = evaluate(function, high)
    if f_low == 0.0:
        return RootResult(low, 0, 0.0)
    if f_high == 0.0:
        return RootResult(high, 0, 0.0)
    if (f_low > 0.0) == (f_high > 0.0):
        raise InvalidInputError(
            f"f(a) and f(b) must have opposite signs (got {f_low} and {f_high})."
        )

    for iteration in range(1, limit + 1):
        mid = low + (high - low) / 2.0
        f_mid = evaluate(function, mid)
        if f_mid == 0.0 or (high - low) / 2.0 <= tolerance or mid in (low, high):
            return RootResult(mid, iteration, abs(f_mid))
        if (f_mid > 0.0) == (f_low > 0.0):
            low, f_low = mid, f_mid
        else:
            high = mid
    raise ConvergenceError(f"Bisection did not reach tol={tol} in {limit} iterations.")


def secant(
    function: ScalarFunction,
    x0: float,
    x1: float,
    *,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> RootResult:
    """Find a root with the secant method (Newton without derivative).

    Each step replaces ``f'`` with the slope through the last two iterates.

    Convergence conditions and limits:
        * Converges **superlinearly** (order ~1.618) to a simple root for
          starting points close enough to it and a smooth ``f``.
        * Not bracketed: it can diverge or leave the interval ``[x0, x1]``.
        * Fails if two successive values of ``f`` are equal (flat secant).

    Args:
        function: ``f(x)``.
        x0: First starting point.
        x1: Second starting point (different from ``x0``).
        tol: Stops when ``|x_{k+1} - x_k| <= tol * (1 + |x_{k+1}|)``.
        max_iter: Maximum number of iterations.

    Raises:
        InvalidInputError: If ``x0 == x1`` or arguments are invalid.
        ConvergenceError: On a flat secant, divergence, or ``max_iter`` exhausted.
    """
    require_callable(function, "function")
    previous = require_finite(x0, "x0")
    current = require_finite(x1, "x1")
    tolerance = require_positive(tol, "tol")
    limit = require_max_iterations(max_iter)
    if previous == current:
        raise InvalidInputError("x0 and x1 must be different.")

    f_previous = evaluate(function, previous)
    f_current = evaluate(function, current)
    for iteration in range(1, limit + 1):
        if f_current == 0.0:
            return RootResult(current, iteration - 1, 0.0)
        if f_current == f_previous:
            raise ConvergenceError("Flat secant: f has equal values at both iterates.")
        x_next = current - f_current * (current - previous) / (f_current - f_previous)
        if not math.isfinite(x_next):
            raise ConvergenceError("Secant iteration diverged to a non-finite value.")
        if abs(x_next - current) <= tolerance * (1.0 + abs(x_next)):
            return RootResult(x_next, iteration, abs(evaluate(function, x_next)))
        previous, f_previous = current, f_current
        current, f_current = x_next, evaluate(function, x_next)
    raise ConvergenceError(f"Secant method did not converge in {limit} iterations.")
