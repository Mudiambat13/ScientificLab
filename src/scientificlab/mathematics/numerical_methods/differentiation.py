"""Numerical approximation of derivatives by finite differences."""

import sys

from scientificlab.mathematics.numerical_methods._common import (
    ScalarFunction,
    evaluate,
    require_callable,
)
from scientificlab.utils.exceptions import InvalidInputError
from scientificlab.utils.validation import require_finite, require_positive

_EPS = sys.float_info.epsilon
_SCHEMES = ("central", "forward", "backward")


def numerical_derivative(
    function: ScalarFunction,
    x: float,
    *,
    step: float | None = None,
    scheme: str = "central",
) -> float:
    """Approximate ``f'(x)`` with a finite-difference formula.

    Schemes (``h`` is the step):

    * ``"central"``: ``(f(x+h) - f(x-h)) / (2h)``, truncation error O(h**2).
    * ``"forward"``: ``(f(x+h) - f(x)) / h``, error O(h).
    * ``"backward"``: ``(f(x) - f(x-h)) / h``, error O(h).

    The total error is the truncation error plus a rounding error of order
    ``eps * |f| / h`` (eps ~ 2.2e-16). Making ``h`` smaller therefore stops
    helping quickly. The default steps balance both terms for smooth functions
    of order one: ``eps**(1/3) * max(1, |x|)`` for central differences (about
    6e-6, giving ~1e-10 accuracy) and ``eps**(1/2) * max(1, |x|)`` for one-sided
    differences (about 1.5e-8, giving ~1e-8 accuracy). The derivative must exist
    and ``f`` be smooth on ``[x - h, x + h]``; near kinks, discontinuities or
    steep oscillations the result is unreliable.

    Args:
        function: ``f(x) -> float``.
        x: Evaluation point.
        step: Step ``h > 0``; ``None`` selects the default described above.
        scheme: ``"central"``, ``"forward"`` or ``"backward"``.

    Raises:
        InvalidInputError: On invalid arguments.
        ConvergenceError: If ``f`` returns a non-finite value at an evaluated point.
    """
    require_callable(function, "function")
    if scheme not in _SCHEMES:
        raise InvalidInputError(f"scheme must be one of {_SCHEMES}, got {scheme!r}.")
    point = require_finite(x, "x")
    scale = max(1.0, abs(point))
    h: float
    if step is None:
        exponent = 1.0 / 3.0 if scheme == "central" else 0.5
        h = float(_EPS**exponent) * scale
    else:
        h = require_positive(step, "step")

    if scheme == "central":
        return (evaluate(function, point + h) - evaluate(function, point - h)) / (
            2.0 * h
        )
    if scheme == "forward":
        return (evaluate(function, point + h) - evaluate(function, point)) / h
    return (evaluate(function, point) - evaluate(function, point - h)) / h
