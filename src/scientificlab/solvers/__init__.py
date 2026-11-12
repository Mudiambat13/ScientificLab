"""Central equation-solving engine (exact SymPy first, explicit numeric fallback)."""

from scientificlab.solvers.engine import (
    EquationSolver,
    SolveResult,
    SystemResult,
    solve_equation,
)

__all__ = ["EquationSolver", "SolveResult", "SystemResult", "solve_equation"]
