"""Numerical solution of ordinary differential equation initial-value problems.

Backed by ``scipy.integrate.solve_ivp``. Results are floating-point
approximations whose accuracy is controlled by ``rtol`` and ``atol``; they are
local-error estimates, not guaranteed global error bounds.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp

from scientificlab.utils.exceptions import ConvergenceError, InvalidInputError
from scientificlab.utils.validation import (
    as_vector,
    require_finite,
    require_non_negative,
    require_positive,
)

_METHODS = ("RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA")

RightHandSide = Callable[[float, npt.NDArray[np.float64]], ArrayLike]


@dataclass(frozen=True)
class NumericODESolution:
    """Numerical solution of ``dy/dt = f(t, y)``.

    Attributes:
        t: Times, shape ``(n_points,)``.
        y: States, shape ``(n_states, n_points)``; ``y[i, k]`` is component
            ``i`` at ``t[k]``.
        method: Integration method used.
        function_evaluations: Number of right-hand-side evaluations.
    """

    t: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    method: str
    function_evaluations: int


def solve_ode_numeric(
    rhs: RightHandSide,
    t_span: tuple[float, float],
    y0: ArrayLike,
    *,
    method: str = "RK45",
    rtol: float = 1e-8,
    atol: float = 1e-10,
    t_eval: ArrayLike | None = None,
) -> NumericODESolution:
    """Integrate the first-order system ``dy/dt = rhs(t, y)``, ``y(t0) = y0``.

    Method guidance: ``RK45`` (default), ``RK23`` and ``DOP853`` are explicit
    methods for non-stiff problems; ``Radau``, ``BDF`` and ``LSODA`` handle
    stiff problems. Explicit methods on stiff problems need tiny steps and may
    fail or be very slow.

    Args:
        rhs: Function ``(t, y) -> dy/dt`` where ``y`` is a 1-D array.
        t_span: ``(t0, t1)``; ``t1 < t0`` integrates backwards.
        y0: Initial state (vector).
        method: One of ``RK45, RK23, DOP853, Radau, BDF, LSODA``.
        rtol: Relative tolerance (> 0); SciPy raises values below ~2e-14.
        atol: Absolute tolerance (>= 0).
        t_eval: Optional times (inside ``t_span``) at which to store the solution.

    Raises:
        InvalidInputError: On invalid arguments or an ill-shaped ``rhs``.
        ConvergenceError: If the integrator fails or produces non-finite values.
    """
    if not callable(rhs):
        raise InvalidInputError("rhs must be a callable (t, y) -> dy/dt.")
    if method not in _METHODS:
        raise InvalidInputError(f"method must be one of {_METHODS}, got {method!r}.")
    if len(t_span) != 2:
        raise InvalidInputError("t_span must be a pair (t0, t1).")
    t0 = require_finite(t_span[0], "t_span[0]")
    t1 = require_finite(t_span[1], "t_span[1]")
    if t0 == t1:
        raise InvalidInputError("t_span must have distinct endpoints.")
    state0 = as_vector(y0, "y0")
    relative = require_positive(rtol, "rtol")
    absolute = require_non_negative(atol, "atol")
    times = None if t_eval is None else as_vector(t_eval, "t_eval")

    try:
        result = solve_ivp(
            rhs,
            (t0, t1),
            state0,
            method=method,
            rtol=relative,
            atol=absolute,
            t_eval=times,
        )
    except ValueError as error:
        raise InvalidInputError(f"Invalid ODE problem: {error}") from error

    if not result.success:
        raise ConvergenceError(f"ODE integration failed: {result.message}")
    if not np.all(np.isfinite(result.y)):
        raise ConvergenceError("ODE integration produced non-finite values.")
    return NumericODESolution(
        t=np.asarray(result.t, dtype=np.float64),
        y=np.asarray(result.y, dtype=np.float64),
        method=method,
        function_evaluations=int(result.nfev),
    )


def solve_second_order_numeric(
    acceleration: Callable[[float, float, float], float],
    t_span: tuple[float, float],
    position0: float,
    velocity0: float,
    *,
    method: str = "RK45",
    rtol: float = 1e-8,
    atol: float = 1e-10,
    t_eval: ArrayLike | None = None,
) -> NumericODESolution:
    """Integrate ``y'' = acceleration(t, y, y')`` with ``y(t0)``, ``y'(t0)`` given.

    The second-order equation is rewritten as the first-order system
    ``(y, v)' = (v, acceleration(t, y, v))``. In the result, ``y[0]`` is the
    position ``y(t)`` and ``y[1]`` the velocity ``y'(t)``.

    See :func:`solve_ode_numeric` for the other parameters and errors.
    """
    if not callable(acceleration):
        raise InvalidInputError("acceleration must be a callable (t, y, v) -> y''.")

    def system(t: float, state: npt.NDArray[np.float64]) -> ArrayLike:
        return [state[1], acceleration(t, float(state[0]), float(state[1]))]

    return solve_ode_numeric(
        system,
        t_span,
        [
            require_finite(position0, "position0"),
            require_finite(velocity0, "velocity0"),
        ],
        method=method,
        rtol=rtol,
        atol=atol,
        t_eval=t_eval,
    )
