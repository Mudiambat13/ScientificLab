"""Symbolic manipulation of algebraic expressions (SymPy backend).

Results are exact symbolic objects, never floating-point approximations.

Security note:
    Expressions given as text are parsed with SymPy's parser, which relies on
    ``eval``. Do not pass untrusted input to these functions.
"""

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    parse_expr,
    standard_transformations,
)

from scientificlab.utils.exceptions import ExpressionParseError

_TRANSFORMATIONS = (*standard_transformations, convert_xor)


def parse_expression(expression: str | sp.Expr) -> sp.Expr:
    """Parse a textual expression into a SymPy expression.

    ``^`` is read as exponentiation (like ``**``). Multiplication must be
    explicit (``2*x``, not ``2x``).

    Raises:
        ExpressionParseError: If the text is not a valid expression.
    """
    if isinstance(expression, sp.Expr):
        return expression
    if not isinstance(expression, str) or not expression.strip():
        raise ExpressionParseError("Expression must be a non-empty string.")
    try:
        parsed = parse_expr(expression, transformations=_TRANSFORMATIONS)
    except Exception as error:  # SymPy raises many unrelated exception types.
        raise ExpressionParseError(
            f"Cannot parse expression {expression!r}: {error}"
        ) from error
    if not isinstance(parsed, sp.Expr):
        raise ExpressionParseError(
            f"{expression!r} is not an algebraic expression "
            f"(got {type(parsed).__name__})."
        )
    return parsed


def parse_symbol(name: str) -> sp.Symbol:
    """Return the SymPy symbol called ``name``.

    Raises:
        ExpressionParseError: If ``name`` is not a plain identifier.
    """
    if not isinstance(name, str) or not name.isidentifier():
        raise ExpressionParseError(f"{name!r} is not a valid variable name.")
    return sp.Symbol(name)


def expand_expression(expression: str | sp.Expr) -> sp.Expr:
    """Expand products and integer powers, e.g. ``(x+1)**2`` -> ``x**2 + 2*x + 1``."""
    return sp.expand(parse_expression(expression))


def factor_expression(expression: str | sp.Expr) -> sp.Expr:
    """Factor over the rationals, e.g. ``x**2 - 1`` -> ``(x - 1)*(x + 1)``."""
    return sp.factor(parse_expression(expression))


def simplify_expression(expression: str | sp.Expr) -> sp.Expr:
    """Simplify an expression with SymPy's heuristics.

    The result is mathematically equivalent to the input but "simplest" is a
    heuristic notion; the output form is not guaranteed canonical.
    """
    return sp.simplify(parse_expression(expression))


def parse_equation(text: str) -> sp.Expr:
    """Turn ``"lhs = rhs"`` into the expression ``lhs - rhs`` (equal to 0).

    A string without ``=`` is read as ``expression = 0``.

    Raises:
        ExpressionParseError: If the text is empty, uses ``==`` or several
            ``=``, or a side is not a valid expression.
    """
    if not isinstance(text, str) or not text.strip():
        raise ExpressionParseError("Each equation must be a non-empty string.")
    if "==" in text or text.count("=") > 1:
        raise ExpressionParseError(f"Use a single '=' in equation {text!r}.")
    if "=" in text:
        left, right = text.split("=")
        return parse_expression(left) - parse_expression(right)
    return parse_expression(text)
