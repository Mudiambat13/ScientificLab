"""Central equation-solving engine.

:class:`EquationSolver` solves a single equation or a system and is explicit
about *what kind* of answer it returns, so a result is never over-claimed:

* ``exact``: solutions are SymPy objects (no floating-point approximation).
  Inputs given as floats make the result inexact.
* ``complete``: ``True`` only when the solution set is guaranteed complete. This
  holds for polynomial and rational equations. For transcendental equations
  (``sin``, ``exp``, ``log``...) SymPy may return only some solutions, so the
  result is flagged incomplete.
* ``verified``: every solution was substituted back in the equation and the
  residual is negligible (relative tolerance ``1e-9``).

The engine never shows "solution steps": it does not produce justified
derivations, and inventing them would be misleading.

A numerical search is used **only** when the caller passes ``initial_guess``.
It finds a single root near the guess and is flagged ``exact=False``,
``complete=False``.

Security note:
    Equations are parsed with SymPy (``eval``-based). Do not pass untrusted text.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import sympy as sp

from scientificlab.mathematics.algebra.expressions import parse_equation
from scientificlab.mathematics.numerical_methods import newton_raphson
from scientificlab.utils.exceptions import (
    ConvergenceError,
    InfiniteSolutionsError,
    InvalidInputError,
    NoSolutionError,
    SolverError,
)

_VERIFY_TOL = 1e-9
_DOMAINS = {"real": sp.S.Reals, "complex": sp.S.Complexes}


@dataclass(frozen=True)
class SolveResult:
    """Outcome of :meth:`EquationSolver.solve`.

    Attributes:
        equation: The equation as given.
        variable: Name of the unknown that was solved for.
        solutions: Solutions as SymPy expressions (floats if ``exact`` is False).
        method: ``"symbolic"`` or ``"numeric"``.
        exact: No floating-point approximation is involved.
        complete: The solution set is guaranteed complete (see module doc).
        verified: Back-substitution residual is negligible for every solution.
        warnings: Human-readable caveats about the result.
    """

    equation: str
    variable: str
    solutions: tuple[sp.Expr, ...]
    method: str
    exact: bool
    complete: bool
    verified: bool
    warnings: tuple[str, ...] = field(default=())

    @property
    def values(self) -> tuple[float | complex, ...]:
        """Numerical value of each solution (fails if free symbols remain)."""
        out: list[float | complex] = []
        for s in self.solutions:
            number = complex(sp.N(s))
            out.append(number.real if abs(number.imag) == 0.0 else number)
        return tuple(out)


@dataclass(frozen=True)
class SystemResult:
    """Outcome of :meth:`EquationSolver.solve_system`.

    ``solutions`` holds one mapping ``variable -> SymPy value`` per solution;
    ``exact``, ``complete``, ``verified`` and ``warnings`` mean the same as in
    :class:`SolveResult`.
    """

    equations: tuple[str, ...]
    variables: tuple[str, ...]
    solutions: tuple[dict[str, sp.Expr], ...]
    method: str
    exact: bool
    complete: bool
    verified: bool
    warnings: tuple[str, ...] = field(default=())


def _symbols(values: Mapping[str, float] | None) -> dict[sp.Symbol, sp.Expr]:
    substitutions: dict[sp.Symbol, sp.Expr] = {}
    for name, number in (values or {}).items():
        if isinstance(number, bool) or not isinstance(number, int | float):
            raise InvalidInputError(f"Value for {name!r} must be a real number.")
        if not math.isfinite(number):
            raise InvalidInputError(f"Value for {name!r} must be finite.")
        substitutions[sp.Symbol(name)] = (
            sp.Integer(number) if isinstance(number, int) else sp.Float(number)
        )
    return substitutions


def _residual_ok(expr: sp.Expr, solution: sp.Expr, variable: sp.Symbol) -> bool:
    """Substitute ``solution`` in ``expr`` and check that it vanishes.

    With free symbols left the check is symbolic (``simplify(residual) == 0``);
    otherwise it is numeric with a tolerance relative to the size of the terms.
    """
    residual = expr.subs(variable, solution)
    if residual.free_symbols or solution.free_symbols:
        return bool(sp.simplify(residual) == 0)
    value = abs(complex(sp.N(residual, 30)))
    terms = [
        abs(complex(sp.N(term.subs(variable, solution), 30)))
        for term in sp.Add.make_args(sp.expand(expr))
    ]
    return value <= _VERIFY_TOL * max([1.0, *terms])


def _is_algebraic(expr: sp.Expr, variable: sp.Symbol) -> bool:
    """Polynomial or rational in ``variable`` (solution set provably complete)."""
    numerator, denominator = sp.fraction(sp.together(expr))
    return bool(
        numerator.is_polynomial(variable) and denominator.is_polynomial(variable)
    )


class EquationSolver:
    """Solve equations and systems with explicit exactness guarantees.

    Args:
        domain: ``"real"`` (default) keeps only real solutions; ``"complex"``
            returns all complex solutions of algebraic equations.

    Raises:
        InvalidInputError: If ``domain`` is not ``"real"`` or ``"complex"``.
    """

    def __init__(self, domain: str = "real") -> None:
        """Create a solver working in the given number domain."""
        if domain not in _DOMAINS:
            raise InvalidInputError("domain must be 'real' or 'complex'.")
        self.domain = domain

    # ------------------------------------------------------------------ single
    def solve(
        self,
        equation: str,
        variable: str | None = None,
        *,
        values: Mapping[str, float] | None = None,
        initial_guess: float | None = None,
    ) -> SolveResult:
        """Solve one equation for ``variable``.

        Args:
            equation: e.g. ``"x**2 - 4 = 0"`` or ``"v = v0 + a*t"``.
            variable: Unknown to solve for. May be omitted if, after applying
                ``values``, exactly one symbol remains.
            values: Known numbers (SI) substituted before solving, e.g.
                ``{"v": 20, "v0": 0, "a": 2}``. Passing floats makes the result
                inexact (``exact=False``).
            initial_guess: Enables a numerical search from this starting point
                (single root, ``exact=False``, ``complete=False``). Required
                when no closed-form solution set can be found.

        Raises:
            ExpressionParseError: If the equation cannot be parsed.
            InvalidInputError: If the unknown is missing, ambiguous or listed
                in ``values``.
            NoSolutionError: If the equation has no solution in the domain.
            InfiniteSolutionsError: If every value is a solution (identity) or
                the solution set is infinite (e.g. periodic).
            SolverError: If no closed form is found and no ``initial_guess``
                was given.
            ConvergenceError: If the numerical search fails.
        """
        expr = parse_equation(equation)
        substitutions = _symbols(values)
        var = self._unknown(expr, variable, substitutions)
        expr = expr.subs(substitutions)
        if var not in expr.free_symbols:
            if sp.simplify(expr) == 0:
                raise InfiniteSolutionsError(
                    f"After substitution the equation does not depend on {var}: "
                    "it holds for every value (identity)."
                )
            raise NoSolutionError(
                f"After substitution the equation does not depend on {var} and is "
                f"false ({sp.simplify(expr)} != 0)."
            )

        if initial_guess is not None:
            return self._numeric(equation, expr, var, initial_guess)

        # With free symbols left, "real" cannot be enforced (their values are
        # unknown), so solve over the complexes and say so.
        symbolic = bool(expr.free_symbols - {var})
        domain = sp.S.Complexes if symbolic else _DOMAINS[self.domain]
        found = sp.solveset(expr, var, domain=domain)
        if found is sp.S.EmptySet:
            raise NoSolutionError(f"No {self.domain} solution for {equation!r}.")
        if found == domain:
            raise InfiniteSolutionsError(f"{equation!r} holds for every value.")
        if not isinstance(found, sp.FiniteSet):
            if not isinstance(found, sp.ConditionSet):
                raise InfiniteSolutionsError(
                    f"The solution set of {equation!r} is infinite ({found}); "
                    "pass initial_guess to find one solution numerically."
                )
            raise SolverError(
                f"No closed form found for {equation!r}. Pass initial_guess "
                "to search for one root numerically."
            )

        solutions = tuple(sorted(found, key=sp.default_sort_key))
        warnings: list[str] = []
        if symbolic and self.domain == "real":
            warnings.append(
                "Other symbols remain, so the real domain is not enforced: "
                "check solutions once they are given numerical values."
            )
        complete = _is_algebraic(expr, var)
        if not complete:
            warnings.append(
                "Non-algebraic equation: the solution set may be incomplete."
            )
        exact = not expr.has(sp.Float)
        if not exact:
            warnings.append("Inputs contain floats: solutions are approximate.")
        verified = all(_residual_ok(expr, s, var) for s in solutions)
        if not verified:
            warnings.append("At least one solution failed back-substitution.")
        return SolveResult(
            equation,
            str(var),
            solutions,
            "symbolic",
            exact,
            complete,
            verified,
            tuple(warnings),
        )

    def _unknown(
        self,
        expr: sp.Expr,
        variable: str | None,
        substitutions: Mapping[sp.Symbol, sp.Expr],
    ) -> sp.Symbol:
        if variable is not None:
            if not isinstance(variable, str) or not variable.isidentifier():
                raise InvalidInputError(f"{variable!r} is not a valid variable name.")
            symbol = sp.Symbol(variable)
            if symbol in substitutions:
                raise InvalidInputError(
                    f"{variable!r} is both the unknown and a given value."
                )
            return symbol
        remaining = sorted(expr.free_symbols - set(substitutions), key=str)
        if len(remaining) != 1:
            raise InvalidInputError(
                "Cannot infer the unknown: specify `variable` (free symbols: "
                f"{[str(s) for s in remaining]})."
            )
        return remaining[0]

    def _numeric(
        self, equation: str, expr: sp.Expr, var: sp.Symbol, guess: float
    ) -> SolveResult:
        if expr.free_symbols - {var}:
            raise InvalidInputError(
                "Numerical solving needs every other symbol given in `values`: "
                f"missing {sorted(str(s) for s in expr.free_symbols - {var})}."
            )
        func = sp.lambdify(var, expr, modules="math")
        deriv = sp.lambdify(var, sp.diff(expr, var), modules="math")
        result = newton_raphson(func, guess, derivative=deriv)
        root = sp.Float(result.root)
        verified = result.residual <= 1e-9 * max(1.0, abs(result.root))
        warnings = [
            "Numerical search: one root near the initial guess; others may exist.",
        ]
        if not verified:
            warnings.append("Residual is not negligible: check the result.")
        return SolveResult(
            equation,
            str(var),
            (root,),
            "numeric",
            False,
            False,
            verified,
            tuple(warnings),
        )

    # ------------------------------------------------------------------ system
    def solve_system(
        self,
        equations: Sequence[str],
        variables: Sequence[str] | None = None,
        *,
        values: Mapping[str, float] | None = None,
        initial_guess: Mapping[str, float] | None = None,
    ) -> SystemResult:
        """Solve a system of equations.

        Exact solving (SymPy) is the default. With ``initial_guess`` (one
        starting value per unknown) a numerical Newton search is used instead.

        Raises:
            InvalidInputError: Bad input, or the unknowns cannot be determined.
            NoSolutionError: If the system is inconsistent.
            InfiniteSolutionsError: If unknowns remain free.
            ConvergenceError: If the numerical search fails.
        """
        if not equations:
            raise InvalidInputError("At least one equation is required.")
        substitutions = _symbols(values)
        exprs = [parse_equation(e).subs(substitutions) for e in equations]
        if variables is None:
            names = sorted({str(s) for e in exprs for s in e.free_symbols})
        else:
            names = list(variables)
        if not names:
            raise InvalidInputError("The equations contain no unknown.")
        symbols = [sp.Symbol(n) for n in names]
        exact = not any(e.has(sp.Float) for e in exprs)
        warnings: list[str] = []

        if initial_guess is not None:
            missing = [n for n in names if n not in initial_guess]
            if missing:
                raise InvalidInputError(f"initial_guess lacks values for {missing}.")
            try:
                point = sp.nsolve(
                    exprs, symbols, [initial_guess[n] for n in names], prec=30
                )
            except (ValueError, ZeroDivisionError) as exc:
                raise ConvergenceError(f"Numerical system solve failed: {exc}") from exc
            solution = {n: sp.Float(point[i]) for i, n in enumerate(names)}
            verified = all(abs(complex(sp.N(e.subs(solution)))) <= 1e-9 for e in exprs)
            warnings.append("Numerical search: one solution near the initial guess.")
            return SystemResult(
                tuple(equations),
                tuple(names),
                (solution,),
                "numeric",
                False,
                False,
                verified,
                tuple(warnings),
            )

        raw = sp.solve(exprs, symbols, dict=True)
        if not raw:
            raise NoSolutionError("The system has no solution.")
        solutions: list[dict[str, sp.Expr]] = []
        for sol in raw:
            free = {s for v in sol.values() for s in v.free_symbols} & set(symbols)
            if [s for s in symbols if s not in sol] or free:
                raise InfiniteSolutionsError(
                    "The system is underdetermined: some unknowns remain free."
                )
            solutions.append({str(s): sol[s] for s in symbols})
        if self.domain == "real":
            real = [
                s for s in solutions if all(v.is_real is not False for v in s.values())
            ]
            if not real:
                raise NoSolutionError("The system has no real solution.")
            solutions = real
        polynomial = all(e.is_polynomial(*symbols) for e in exprs)
        if not polynomial:
            warnings.append(
                "Non-polynomial system: the solution set may be incomplete."
            )
        if not exact:
            warnings.append("Inputs contain floats: solutions are approximate.")
        verified = all(
            all(
                abs(complex(sp.N(e.subs({sp.Symbol(k): v for k, v in s.items()}), 30)))
                <= _VERIFY_TOL * max(1.0, *(abs(complex(sp.N(v))) for v in s.values()))
                for e in exprs
            )
            for s in solutions
        )
        if not verified:
            warnings.append("At least one solution failed back-substitution.")
        return SystemResult(
            tuple(equations),
            tuple(names),
            tuple(solutions),
            "symbolic",
            exact,
            polynomial,
            verified,
            tuple(warnings),
        )


def solve_equation(
    equation: str,
    variable: str | None = None,
    *,
    values: Mapping[str, float] | None = None,
    initial_guess: float | None = None,
    domain: str = "real",
) -> SolveResult:
    """Shortcut for ``EquationSolver(domain).solve(...)``."""
    return EquationSolver(domain).solve(
        equation, variable, values=values, initial_guess=initial_guess
    )
