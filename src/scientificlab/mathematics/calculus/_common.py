"""Internal helpers shared by the calculus modules."""

import sympy as sp

from scientificlab.mathematics.algebra.expressions import parse_expression
from scientificlab.utils.exceptions import InvalidInputError, SolverError

_UNDEFINED = (sp.oo, sp.zoo, sp.nan)


def to_sympy_value(value: float | str | sp.Expr, name: str) -> sp.Expr:
    """Convert a bound/point given as number, text (``"oo"``, ``"pi"``) or Expr.

    Python floats become SymPy ``Float`` objects, so results involving them are
    floating-point; pass integers or strings (``"1/3"``) to keep results exact.
    """
    if isinstance(value, bool):
        raise InvalidInputError(f"{name} must be a number or expression.")
    if isinstance(value, (int, float)):
        return sp.sympify(value)
    return parse_expression(value)


def require_order(order: int, name: str = "order") -> int:
    """Return ``order`` as an int, requiring a non-negative integer."""
    if isinstance(order, bool) or not isinstance(order, int) or order < 0:
        raise InvalidInputError(
            f"{name} must be a non-negative integer, got {order!r}."
        )
    return order


def reject_unevaluated(result: sp.Expr, kind: type, what: str) -> sp.Expr:
    """Raise if SymPy returned an unevaluated ``kind`` (no closed form found)."""
    if result.has(kind):
        raise SolverError(f"No closed form found for the {what}.")
    return result


def reject_undefined(result: sp.Expr, what: str) -> sp.Expr:
    """Raise if ``result`` is infinite or undefined (divergence, pole, nan)."""
    if result.has(*_UNDEFINED):
        raise SolverError(f"The {what} diverges or is undefined (got {result}).")
    return result
