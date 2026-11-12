"""Shared figure helpers (object-oriented Matplotlib API, no global state).

Figures are built with :class:`matplotlib.figure.Figure` directly instead of
``pyplot``, so importing this package never opens a window, never needs a GUI
backend and leaks no figures. ``pyplot`` is imported lazily, only for
``show=True``.
"""

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp
from matplotlib.figure import Figure

from scientificlab.mathematics.algebra.expressions import parse_expression
from scientificlab.utils.exceptions import InvalidInputError

SUPPORTED_FORMATS = (".png", ".svg", ".pdf")

ScalarFunction = Callable[[float], float]


def new_figure(
    title: str | None,
    xlabel: str | None,
    ylabel: str | None,
    figsize: tuple[float, float] = (7.0, 4.5),
) -> tuple[Figure, Any]:
    """Create a figure with one labelled, gridded axes."""
    fig = Figure(figsize=figsize, layout="constrained")
    ax = fig.add_subplot(1, 1, 1)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return fig, ax


def finish(fig: Figure, ax: Any, save_path: str | Path | None, show: bool) -> Figure:
    """Add a legend if needed, optionally save (png/svg/pdf) and show."""
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    if save_path is not None:
        save_figure(fig, save_path)
    if show:
        import matplotlib.pyplot as plt

        fig.set_canvas(plt.figure().canvas)
        plt.show()
    return fig


def save_figure(fig: Figure, path: str | Path, dpi: int = 150) -> Path:
    """Save ``fig`` as PNG, SVG or PDF according to the file extension.

    Raises:
        InvalidInputError: If the extension is not .png, .svg or .pdf, or the
            target directory does not exist.
    """
    target = Path(path)
    if target.suffix.lower() not in SUPPORTED_FORMATS:
        raise InvalidInputError(
            f"Unsupported format {target.suffix!r}; use one of {SUPPORTED_FORMATS}."
        )
    if not target.parent.exists():
        raise InvalidInputError(f"Directory {str(target.parent)!r} does not exist.")
    fig.savefig(target, dpi=dpi)
    return target


def as_callable(function: ScalarFunction | str) -> tuple[ScalarFunction, str]:
    """Return ``(callable, default label)`` from a callable or an expression in x."""
    if callable(function):
        return function, getattr(function, "__name__", "f")
    expr = parse_expression(function)
    extra = expr.free_symbols - {sp.Symbol("x")}
    if extra:
        raise InvalidInputError(
            f"Expression {function!r} must depend on x only "
            f"(found {sorted(map(str, extra))})."
        )
    return sp.lambdify(sp.Symbol("x"), expr, modules="numpy"), f"y = {function}"


def sample(
    function: ScalarFunction, xs: np.ndarray, name: str = "function"
) -> np.ndarray:
    """Evaluate ``function`` on ``xs``; non-finite or failing points become NaN.

    NaN leaves a visible gap in the curve (poles, domain limits) instead of a
    misleading line. If *no* point is finite the data cannot be plotted.

    Raises:
        InvalidInputError: If every evaluated point is non-finite.
    """
    ys = np.full(xs.shape, np.nan)
    with np.errstate(all="ignore"):
        try:
            vectorised: Callable[[Any], Any] = function
            values = np.asarray(vectorised(xs), dtype=np.complex128)
            if values.shape == xs.shape:
                ys = np.where(np.abs(values.imag) < 1e-12, values.real, np.nan)
        except Exception:  # not vectorised: fall back to point by point
            for i, x in enumerate(xs):
                try:
                    v = complex(function(float(x)))
                except (ArithmeticError, ValueError, TypeError):
                    continue
                if abs(v.imag) < 1e-12:
                    ys[i] = v.real
    ys[~np.isfinite(ys)] = np.nan
    if np.all(np.isnan(ys)):
        raise InvalidInputError(f"{name} is not finite anywhere on the interval.")
    return _break_at_poles(ys)


def _break_at_poles(ys: np.ndarray) -> np.ndarray:
    """Insert gaps where the curve flips sign across a huge jump (e.g. 1/x at 0).

    Heuristic: a jump between neighbours of opposite sign larger than 10 times
    the 90th percentile of ``|y|`` is treated as a pole, not as a real segment.
    Genuine steep-but-continuous curves can occasionally be cut; raise
    ``points`` to resolve them.
    """
    out = ys.copy()
    scale = np.nanpercentile(np.abs(ys), 90)
    if not np.isfinite(scale) or scale == 0.0:
        return out
    with np.errstate(invalid="ignore"):
        jump = (ys[:-1] * ys[1:] < 0) & (np.abs(ys[1:] - ys[:-1]) > 10 * scale)
    out[:-1][jump] = np.nan
    return out


def linspace(x_range: Sequence[float], points: int) -> np.ndarray:
    """Validated ``np.linspace`` over ``x_range = (xmin, xmax)``."""
    if len(x_range) != 2:
        raise InvalidInputError("x_range must be (xmin, xmax).")
    lo, hi = float(x_range[0]), float(x_range[1])
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo >= hi:
        raise InvalidInputError("x_range must be finite with xmin < xmax.")
    if isinstance(points, bool) or not isinstance(points, int) or points < 2:
        raise InvalidInputError("points must be an integer >= 2.")
    return np.linspace(lo, hi, points)
