"""Plots for samples and linear fits."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from scientificlab.mathematics.statistics import LinearFit, linear_regression
from scientificlab.utils.exceptions import InvalidInputError
from scientificlab.utils.validation import as_vector
from scientificlab.visualization._common import finish, new_figure


def plot_histogram(
    data: Sequence[float],
    *,
    bins: int = 10,
    title: str | None = None,
    xlabel: str | None = "value",
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Histogram of a sample, with the mean marked.

    Raises:
        InvalidInputError: If the data are invalid or ``bins < 1``.
    """
    values = as_vector(data, "data")
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise InvalidInputError("bins must be an integer >= 1.")
    fig, ax = new_figure(title, xlabel, "count")
    ax.hist(values, bins=bins, edgecolor="white")
    ax.axvline(float(np.mean(values)), color="black", ls="--", label="mean")
    return finish(fig, ax, save_path, show)


def plot_regression(
    x: Sequence[float],
    y: Sequence[float],
    fit: LinearFit | None = None,
    *,
    title: str | None = None,
    xlabel: str | None = "x",
    ylabel: str | None = "y",
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Scatter plot of the data with the least-squares line and its equation.

    Args:
        x: Independent variable.
        y: Dependent variable.
        fit: A fit from :func:`linear_regression`; computed if omitted.
        title: Figure title.
        xlabel: X axis label.
        ylabel: Y axis label.
        save_path: Optional ``.png``, ``.svg`` or ``.pdf`` file.
        show: Open an interactive window.

    Raises:
        InvalidInputError: If the data cannot be fitted (see
            :func:`linear_regression`).
    """
    xs, ys = as_vector(x, "x"), as_vector(y, "y")
    result = fit if fit is not None else linear_regression(xs, ys)
    fig, ax = new_figure(title, xlabel, ylabel)
    ax.plot(xs, ys, "o", label="data")
    line_x = np.array([np.min(xs), np.max(xs)])
    sign = "+" if result.intercept >= 0 else "-"
    ax.plot(
        line_x,
        result.predict(line_x),
        label=(
            f"y = {result.slope:.4g} x {sign} {abs(result.intercept):.4g} "
            f"(R² = {result.r_squared:.4f})"
        ),
    )
    return finish(fig, ax, save_path, show)
