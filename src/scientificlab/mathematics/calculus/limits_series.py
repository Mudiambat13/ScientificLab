"""Limits, Taylor expansions and series sums (SymPy backend, exact results)."""

import sympy as sp

from scientificlab.mathematics.algebra.expressions import (
    parse_expression,
    parse_symbol,
)
from scientificlab.mathematics.calculus._common import (
    reject_undefined,
    reject_unevaluated,
    require_order,
    to_sympy_value,
)
from scientificlab.utils.exceptions import InvalidInputError, SolverError

_DIRECTIONS = ("+", "-", "+-")


def limit(
    expression: str | sp.Expr,
    variable: str,
    point: float | str | sp.Expr,
    direction: str = "+-",
) -> sp.Expr:
    """Limit of an expression as ``variable`` approaches ``point``.

    Args:
        expression: Expression, e.g. ``"sin(x)/x"``.
        variable: Name of the variable.
        point: Number or text (``"0"``, ``"oo"``, ``"-oo"``, ``"pi"``).
        direction: ``"+"`` (from the right), ``"-"`` (from the left) or ``"+-"``
            (two-sided; the default).

    Returns:
        The exact limit, which may be ``oo`` or ``-oo`` when the expression
        diverges to infinity in a definite direction.

    Raises:
        InvalidInputError: If ``direction`` is not one of ``+``, ``-``, ``+-``.
        SolverError: If SymPy cannot evaluate the limit, or it does not exist
            (e.g. ``1/x`` two-sided at 0, or an oscillating function).
    """
    if direction not in _DIRECTIONS:
        raise InvalidInputError(f"direction must be one of {_DIRECTIONS}.")
    result = sp.limit(
        parse_expression(expression),
        parse_symbol(variable),
        to_sympy_value(point, "point"),
        direction,
    )
    reject_unevaluated(result, sp.Limit, "limit")
    if result.has(sp.nan, sp.zoo) or isinstance(result, sp.AccumBounds):
        raise SolverError(f"The limit does not exist or is undefined (got {result}).")
    return result


def taylor_series(
    expression: str | sp.Expr,
    variable: str = "x",
    point: float | str | sp.Expr = 0,
    order: int = 5,
) -> sp.Expr:
    """Taylor polynomial of ``expression`` around ``point``, up to degree ``order``.

    The returned polynomial in ``(variable - point)`` contains every term of
    degree ``<= order`` and **no remainder term**: it is an approximation valid
    near ``point`` only, with error of order ``(variable - point)**(order + 1)``
    for functions analytic at ``point``. Expansions of functions that are not
    analytic there (e.g. ``1/x`` or ``sqrt(x)`` at 0) may contain negative or
    fractional powers (Laurent/Puiseux terms).

    Args:
        expression: Expression to expand.
        variable: Expansion variable.
        point: Expansion point (``0`` gives a Maclaurin series).
        order: Highest degree kept (non-negative integer).
    """
    n = require_order(order)
    series = sp.series(
        parse_expression(expression),
        parse_symbol(variable),
        to_sympy_value(point, "point"),
        n + 1,
    )
    return sp.expand(series.removeO())


def series_sum(
    expression: str | sp.Expr,
    variable: str,
    start: float | str | sp.Expr,
    end: float | str | sp.Expr,
) -> sp.Expr:
    """Exact sum of ``expression`` for ``variable`` from ``start`` to ``end``.

    ``end`` may be ``"oo"`` for an infinite series, e.g. the Basel problem
    ``series_sum("1/n**2", "n", 1, "oo")`` gives ``pi**2/6``.

    Raises:
        SolverError: If no closed form is found, or the series diverges.
    """
    result = sp.summation(
        parse_expression(expression),
        (
            parse_symbol(variable),
            to_sympy_value(start, "start"),
            to_sympy_value(end, "end"),
        ),
    )
    reject_unevaluated(result, sp.Sum, "series sum")
    return reject_undefined(result, "series")
