"""Exact symbolic integration (SymPy backend)."""

import sympy as sp

from scientificlab.mathematics.algebra.expressions import (
    parse_expression,
    parse_symbol,
)
from scientificlab.mathematics.calculus._common import (
    reject_undefined,
    reject_unevaluated,
    to_sympy_value,
)
from scientificlab.utils.exceptions import InvalidInputError


def indefinite_integral(expression: str | sp.Expr, variable: str = "x") -> sp.Expr:
    """Antiderivative of an expression (the constant of integration is omitted).

    Args:
        expression: Integrand, e.g. ``"3*x**2"``.
        variable: Integration variable.

    Returns:
        One exact antiderivative ``F`` with ``dF/dx = integrand``; any
        ``F + C`` is also valid.

    Raises:
        SolverError: If SymPy finds no closed form.
    """
    result = sp.integrate(parse_expression(expression), parse_symbol(variable))
    return reject_unevaluated(result, sp.Integral, "antiderivative")


def definite_integral(
    expression: str | sp.Expr,
    variable: str,
    lower: float | str | sp.Expr,
    upper: float | str | sp.Expr,
) -> sp.Expr:
    """Exact definite integral between two bounds.

    Bounds may be numbers or text such as ``"pi"``, ``"oo"`` (infinity) or
    ``"1/3"``. Python floats make the result floating-point; use integers or
    strings to keep it exact.

    Raises:
        InvalidInputError: If a bound is a boolean.
        SolverError: If no closed form is found, or the integral diverges or is
            undefined (e.g. ``1/x`` over ``[0, 1]``).
    """
    a = to_sympy_value(lower, "lower")
    b = to_sympy_value(upper, "upper")
    if a == b:
        return sp.Integer(0)
    symbol = parse_symbol(variable)
    expr = parse_expression(expression)
    if a.has(symbol) or b.has(symbol):
        raise InvalidInputError("Bounds must not contain the integration variable.")
    result = sp.integrate(expr, (symbol, a, b))
    reject_unevaluated(result, sp.Integral, "definite integral")
    return reject_undefined(result, "definite integral")
