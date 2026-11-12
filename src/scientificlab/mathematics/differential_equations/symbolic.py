"""Exact solution of first- and second-order ordinary differential equations.

Backed by SymPy's ``dsolve``. Results are exact symbolic expressions; every
solution is checked by substitution (and against the initial conditions) and
the outcome is reported in ``SymbolicODESolution.verified``.

Security note:
    Equations are parsed with SymPy's parser, which relies on ``eval``. Do not
    pass untrusted input.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    parse_expr,
    standard_transformations,
)

from scientificlab.mathematics.algebra.expressions import parse_expression
from scientificlab.utils.exceptions import (
    ExpressionParseError,
    InvalidInputError,
    SolverError,
)

_MAX_ORDER = 2
_TRANSFORMATIONS = (*standard_transformations, convert_xor)


@dataclass(frozen=True)
class SymbolicODESolution:
    """Closed-form solution(s) of an ODE.

    Attributes:
        equation: The equation as given.
        function: Name of the unknown function (e.g. ``"y"``).
        variable: Name of the independent variable (e.g. ``"x"``).
        order: Order of the equation (1 or 2).
        solutions: Right-hand sides of ``y(x) = ...``. Linear equations have one;
            some nonlinear equations have several branches. Unconstrained
            integration constants appear as ``C1``, ``C2``.
        verified: ``True`` if every solution satisfies the equation and all
            initial conditions after simplification. ``False`` means the check
            failed **or could not be completed**; treat the result with care.
    """

    equation: str
    function: str
    variable: str
    order: int
    solutions: tuple[sp.Expr, ...]
    verified: bool

    @property
    def solution(self) -> sp.Expr:
        """The unique solution.

        Raises:
            InvalidInputError: If the equation has several solution branches
                (use :attr:`solutions`).
        """
        if len(self.solutions) != 1:
            raise InvalidInputError(
                f"The equation has {len(self.solutions)} solution branches; "
                "use .solutions."
            )
        return self.solutions[0]


def _check_names(function: str, variable: str) -> None:
    for label, name in (("function", function), ("variable", variable)):
        if not name.isidentifier():
            raise ExpressionParseError(f"{name!r} is not a valid {label} name.")
    if function == variable:
        raise InvalidInputError("function and variable must have different names.")


def _rewrite(text: str, function: str, variable: str) -> str:
    """Rewrite ``y''`` and bare ``y`` into SymPy ``Derivative``/``y(x)`` calls."""
    with_derivatives = re.sub(
        rf"\b{function}('+)",
        lambda m: (
            f"Derivative({function}({variable}), ({variable}, {len(m.group(1))}))"
        ),
        text,
    )
    return re.sub(
        rf"\b{function}\b(?!\s*\()", f"{function}({variable})", with_derivatives
    )


def _parse_equation(
    equation: str, function: str, variable: str
) -> tuple[sp.Expr, sp.Function, sp.Symbol]:
    if not isinstance(equation, str) or not equation.strip():
        raise ExpressionParseError("The equation must be a non-empty string.")
    if "==" in equation or equation.count("=") > 1:
        raise ExpressionParseError(f"Use a single '=' in equation {equation!r}.")
    func = sp.Function(function)
    symbol = sp.Symbol(variable)
    local = {function: func, variable: symbol}
    sides = [_rewrite(s, function, variable) for s in equation.split("=")]
    try:
        parsed = [
            parse_expr(s, local_dict=local, transformations=_TRANSFORMATIONS)
            for s in sides
        ]
    except Exception as error:
        raise ExpressionParseError(
            f"Cannot parse equation {equation!r}: {error}"
        ) from error
    residual = parsed[0] - parsed[1] if len(parsed) == 2 else parsed[0]
    if not isinstance(residual, sp.Expr):
        raise ExpressionParseError(f"{equation!r} is not an ODE.")
    return residual, func, symbol


def _parse_conditions(
    conditions: Mapping[str, float | str],
    function: str,
    order: int,
) -> list[tuple[int, sp.Expr, sp.Expr]]:
    """Parse ``{"y(0)": 1, "y'(0)": 0}`` into ``(derivative order, point, value)``."""
    pattern = re.compile(rf"^{function}('*)\(\s*(.+?)\s*\)$")
    parsed: list[tuple[int, sp.Expr, sp.Expr]] = []
    for key, value in conditions.items():
        match = pattern.match(key.strip())
        if match is None:
            raise InvalidInputError(
                f"Initial condition {key!r} must look like '{function}(0)' or "
                f"'{function}'(0)'."
            )
        k = len(match.group(1))
        if k >= order:
            raise InvalidInputError(
                f"Condition {key!r} constrains derivative {k}, but the equation "
                f"has order {order} (allowed derivatives: 0..{order - 1})."
            )
        point = parse_expression(match.group(2))
        number = (
            sp.sympify(value)
            if isinstance(value, (int, float))
            else parse_expression(value)
        )
        parsed.append((k, point, number))
    if len({(k, p) for k, p, _ in parsed}) != len(parsed):
        raise InvalidInputError("Duplicate initial conditions.")
    return parsed


def solve_ode(
    equation: str,
    function: str = "y",
    variable: str = "x",
    initial_conditions: Mapping[str, float | str] | None = None,
) -> SymbolicODESolution:
    """Solve a first- or second-order ODE exactly.

    Derivatives are written with primes: ``"y' + 2*y = 0"``,
    ``"y'' + y = 0"``. The unknown function is written bare (``y``), not
    ``y(x)``. A single ``=`` separates the sides; without ``=`` the expression
    is taken to equal 0.

    Args:
        equation: The differential equation.
        function: Name of the unknown function.
        variable: Name of the independent variable.
        initial_conditions: Optional mapping such as ``{"y(0)": 1, "y'(0)": 0}``.
            Values are numbers or text (``"1/2"``). Use integers/strings to keep
            the result exact. Fewer conditions than the order leave free
            constants ``C1``, ``C2`` in the solution.

    Returns:
        A :class:`SymbolicODESolution`. Check ``verified``.

    Raises:
        ExpressionParseError: If the equation or a name cannot be parsed.
        InvalidInputError: If the equation has no derivative, has order > 2, or
            the initial conditions are malformed or exceed the order.
        SolverError: If SymPy cannot solve the equation or the initial
            conditions are inconsistent with it.
    """
    _check_names(function, variable)
    residual, func, symbol = _parse_equation(equation, function, variable)
    derivatives = residual.atoms(sp.Derivative)
    if not derivatives:
        raise InvalidInputError("The equation contains no derivative of the function.")
    order = max(d.derivative_count for d in derivatives)
    if order > _MAX_ORDER:
        raise InvalidInputError(
            f"Only first- and second-order equations are supported, got order {order}."
        )

    conditions = _parse_conditions(initial_conditions or {}, function, order)
    ics: dict[sp.Expr, sp.Expr] = {}
    for k, point, value in conditions:
        target = func(symbol) if k == 0 else sp.Derivative(func(symbol), (symbol, k))
        ics[target.subs(symbol, point)] = value

    try:
        raw = sp.dsolve(sp.Eq(residual, 0), func(symbol), ics=ics or None)
    except (NotImplementedError, ValueError, TypeError, NameError) as error:
        raise SolverError(f"Cannot solve {equation!r}: {error}") from error

    equalities = raw if isinstance(raw, list) else [raw]
    solutions = tuple(e.rhs for e in equalities)
    verified = _verify(residual, func, symbol, equalities, conditions)
    return SymbolicODESolution(
        equation=equation,
        function=function,
        variable=variable,
        order=order,
        solutions=solutions,
        verified=verified,
    )


def _verify(
    residual: sp.Expr,
    func: sp.Function,
    symbol: sp.Symbol,
    equalities: list[sp.Eq],
    conditions: list[tuple[int, sp.Expr, sp.Expr]],
) -> bool:
    """Substitute every solution into the ODE and the initial conditions."""
    for equality in equalities:
        try:
            satisfied, _ = sp.checkodesol(
                sp.Eq(residual, 0), equality, func=func(symbol)
            )
        except Exception:  # SymPy may fail on exotic expressions
            return False
        if satisfied is not True:
            return False
        for k, point, value in conditions:
            actual = (
                sp.diff(equality.rhs, symbol, k).subs(symbol, point)
                if k
                else (equality.rhs.subs(symbol, point))
            )
            if sp.simplify(actual - value) != 0:
                return False
    return True
