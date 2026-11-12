"""Plots of mathematical functions and data."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from scientificlab.utils.exceptions import InvalidInputError
from scientificlab.visualization._common import (
    ScalarFunction,
    as_callable,
    finish,
    linspace,
    new_figure,
    sample,
)


def plot_functions(
    functions: Sequence[ScalarFunction | str],
    x_range: tuple[float, float],
    *,
    labels: Sequence[str] | None = None,
    points: int = 1000,
    y_limits: tuple[float, float] | None = None,
    title: str | None = None,
    xlabel: str | None = "x",
    ylabel: str | None = "y",
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Plot one or more functions of ``x`` on ``x_range``.

    Each function is a callable ``f(x)`` or an expression string in ``x``
    (``"sin(x)/x"``). Points where a function is undefined or not real (poles,
    ``sqrt`` of a negative number, ``log`` of a non-positive number) are left
    as gaps; they are **not** interpolated across. Sampling is uniform with
    ``points`` points, so very narrow features may be missed: increase
    ``points`` if in doubt. Poles (sign flip across a huge jump, e.g. ``1/x``
    at 0) are cut rather than joined by a vertical line; this is a heuristic.

    Args:
        functions: Non-empty sequence of callables or expression strings.
        x_range: ``(xmin, xmax)``.
        labels: Legend labels (default: function name or expression).
        points: Number of sample points (>= 2).
        y_limits: ``(ymin, ymax)`` to crop the view (useful near poles).
        title: Figure title.
        xlabel: X axis label (include units, e.g. ``"t (s)"``).
        ylabel: Y axis label.
        save_path: If given, saves to this ``.png``, ``.svg`` or ``.pdf`` file.
        show: Open an interactive window (needs a GUI backend).

    Returns:
        The Matplotlib ``Figure``.

    Raises:
        InvalidInputError: Bad range/points/labels, unsupported file format, or
            a function that is nowhere finite on the range.
        ExpressionParseError: If a string expression cannot be parsed.
    """
    if not functions:
        raise InvalidInputError("At least one function is required.")
    if labels is not None and len(labels) != len(functions):
        raise InvalidInputError("labels must have one entry per function.")
    xs = linspace(x_range, points)
    fig, ax = new_figure(title, xlabel, ylabel)
    for index, function in enumerate(functions):
        func, default = as_callable(function)
        ys = sample(func, xs, default)
        ax.plot(xs, ys, label=labels[index] if labels else default)
    if y_limits is not None:
        if len(y_limits) != 2 or not y_limits[0] < y_limits[1]:
            raise InvalidInputError("y_limits must be (ymin, ymax) with ymin < ymax.")
        ax.set_ylim(*y_limits)
    return finish(fig, ax, save_path, show)


def plot_function(
    function: ScalarFunction | str,
    x_range: tuple[float, float],
    **options: object,
) -> Figure:
    """Plot a single function; see :func:`plot_functions` for all options."""
    return plot_functions([function], x_range, **options)  # type: ignore[arg-type]


def plot_data(
    x: Sequence[float],
    y: Sequence[float],
    *,
    yerr: Sequence[float] | None = None,
    label: str | None = None,
    connect: bool = False,
    title: str | None = None,
    xlabel: str | None = "x",
    ylabel: str | None = "y",
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Plot measured/discrete data as markers, optionally with error bars.

    Raises:
        InvalidInputError: If lengths differ, data is empty or non-finite, or
            ``yerr`` is negative.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.ndim != 1 or xa.shape != ya.shape or xa.size == 0:
        raise InvalidInputError(
            "x and y must be non-empty 1-D sequences of equal length."
        )
    if not (np.all(np.isfinite(xa)) and np.all(np.isfinite(ya))):
        raise InvalidInputError("x and y must contain only finite values.")
    err = None
    if yerr is not None:
        err = np.asarray(yerr, dtype=float)
        if err.shape != ya.shape or np.any(err < 0) or not np.all(np.isfinite(err)):
            raise InvalidInputError(
                "yerr must be finite, non-negative, same length as y."
            )
    fig, ax = new_figure(title, xlabel, ylabel)
    style = "o-" if connect else "o"
    if err is not None:
        ax.errorbar(xa, ya, yerr=err, fmt=style, capsize=3, label=label)
    else:
        ax.plot(xa, ya, style, label=label)
    return finish(fig, ax, save_path, show)
