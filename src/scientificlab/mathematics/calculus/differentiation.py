"""Exact symbolic differentiation (SymPy backend)."""

from collections.abc import Sequence

import sympy as sp

from scientificlab.mathematics.algebra.expressions import (
    parse_expression,
    parse_symbol,
)
from scientificlab.mathematics.calculus._common import require_order
from scientificlab.utils.exceptions import InvalidInputError


def differentiate(
    expression: str | sp.Expr,
    variable: str = "x",
    order: int = 1,
) -> sp.Expr:
    """Differentiate an expression exactly with respect to one variable.

    Args:
        expression: Expression such as ``"x**3 + 2*x**2 + 5"``.
        variable: Name of the differentiation variable.
        order: Derivative order (``0`` returns the expression unchanged).

    Returns:
        The exact symbolic derivative (not a numerical approximation).

    Raises:
        ExpressionParseError: If the expression or variable name is invalid.
        InvalidInputError: If ``order`` is not a non-negative integer.
    """
    expr = parse_expression(expression)
    symbol = parse_symbol(variable)
    n = require_order(order)
    if n == 0:
        return expr
    return sp.diff(expr, symbol, n)


def partial_derivative(
    expression: str | sp.Expr,
    variables: str | Sequence[str],
) -> sp.Expr:
    """Differentiate with respect to several variables in sequence.

    ``partial_derivative(f, "x")`` is df/dx; ``partial_derivative(f, ["x", "y"])``
    is the mixed derivative d2f/dx dy; repeat a name for higher orders, e.g.
    ``["x", "x"]`` for d2f/dx2. Other variables are held constant.

    Raises:
        InvalidInputError: If no variable is given.
        ExpressionParseError: If the expression or a variable name is invalid.
    """
    expr = parse_expression(expression)
    names = [variables] if isinstance(variables, str) else list(variables)
    if not names:
        raise InvalidInputError("At least one variable is required.")
    return sp.diff(expr, *(parse_symbol(name) for name in names))
