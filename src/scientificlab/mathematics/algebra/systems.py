"""Exact solving of systems of algebraic equations (SymPy backend)."""

from collections.abc import Sequence
from dataclasses import dataclass

import sympy as sp

from scientificlab.mathematics.algebra.expressions import parse_equation
from scientificlab.utils.exceptions import (
    InfiniteSolutionsError,
    InvalidInputError,
    NoSolutionError,
)


@dataclass(frozen=True)
class SystemSolution:
    """Solutions of a system of equations.

    Attributes:
        variables: Names of the unknowns, in the order they were solved for.
        solutions: One mapping ``variable name -> exact value`` per solution.
            Linear systems have exactly one; nonlinear systems may have several.
    """

    variables: tuple[str, ...]
    solutions: tuple[dict[str, sp.Expr], ...]


def solve_system(
    equations: Sequence[str],
    variables: Sequence[str] | None = None,
) -> SystemSolution:
    """Solve a system of equations exactly.

    Args:
        equations: Equations such as ``"x + y = 3"``. A string without ``=`` is
            read as ``expression = 0``.
        variables: Unknowns to solve for. Defaults to every symbol appearing in
            the equations, sorted alphabetically.

    Returns:
        A :class:`SystemSolution` with exact (symbolic/rational) values.

    Raises:
        InvalidInputError: If no equation is given.
        ExpressionParseError: If an equation cannot be parsed.
        NoSolutionError: If the system is inconsistent.
        InfiniteSolutionsError: If some unknowns stay free (underdetermined).
    """
    if not equations:
        raise InvalidInputError("At least one equation is required.")
    expressions = [parse_equation(e) for e in equations]

    if variables is None:
        symbols = sorted(set().union(*(e.free_symbols for e in expressions)), key=str)
    else:
        symbols = [sp.Symbol(v) for v in variables]
    if not symbols:
        raise InvalidInputError("The equations contain no variable to solve for.")

    raw = sp.solve(expressions, symbols, dict=True)
    if not raw:
        raise NoSolutionError("The system has no solution.")

    solutions: list[dict[str, sp.Expr]] = []
    for solution in raw:
        unsolved = [s for s in symbols if s not in solution]
        free = {s for value in solution.values() for s in value.free_symbols} & set(
            symbols
        )
        if unsolved or free:
            raise InfiniteSolutionsError(
                "The system is underdetermined: some unknowns remain free."
            )
        solutions.append({str(s): solution[s] for s in symbols})
    return SystemSolution(
        variables=tuple(str(s) for s in symbols),
        solutions=tuple(solutions),
    )
