"""Ordinary least-squares straight-line fit."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike

from scientificlab.utils.exceptions import InvalidInputError
from scientificlab.utils.validation import as_vector


@dataclass(frozen=True)
class LinearFit:
    """Result of :func:`linear_regression` for the model ``y = slope*x + intercept``.

    Attributes:
        slope: Fitted slope.
        intercept: Fitted intercept.
        r_squared: Coefficient of determination ``1 - SSE/SST`` in ``[0, 1]``.
        slope_stderr: Standard error of the slope.
        intercept_stderr: Standard error of the intercept.
        residual_std: Residual standard deviation ``sqrt(SSE / (n - 2))``, in
            the units of ``y``.
        n: Number of points.
    """

    slope: float
    intercept: float
    r_squared: float
    slope_stderr: float
    intercept_stderr: float
    residual_std: float
    n: int

    def predict(self, x: ArrayLike) -> npt.NDArray[np.float64]:
        """Evaluate the fitted line at ``x`` (no extrapolation warning is made)."""
        return self.slope * as_vector(x, "x") + self.intercept


def linear_regression(x: ArrayLike, y: ArrayLike) -> LinearFit:
    """Least-squares fit of ``y = slope * x + intercept``.

    The errors are assumed to be in ``y`` only, independent, with constant
    variance (the standard standard-error formulas rely on this). Weighted fits
    and errors in ``x`` are not handled.

    Args:
        x: Independent variable, at least 3 values, not all equal.
        y: Dependent variable, same length, not all equal.

    Raises:
        InvalidInputError: If lengths differ, there are fewer than 3 points
            (the standard errors need ``n - 2 >= 1`` degrees of freedom), ``x``
            is constant (vertical line) or ``y`` is constant (``R**2`` is
            undefined).
    """
    xs = as_vector(x, "x")
    ys = as_vector(y, "y")
    if xs.size != ys.size:
        raise InvalidInputError(
            f"x and y must have the same length, got {xs.size} and {ys.size}."
        )
    n = xs.size
    if n < 3:
        raise InvalidInputError("linear_regression needs at least 3 points.")
    dx, dy = xs - xs.mean(), ys - ys.mean()
    sxx, sst = float(np.sum(dx * dx)), float(np.sum(dy * dy))
    if sxx == 0.0:
        raise InvalidInputError("x is constant: the slope is undefined.")
    if sst == 0.0:
        raise InvalidInputError("y is constant: R-squared is undefined.")
    slope = float(np.sum(dx * dy) / sxx)
    intercept = float(ys.mean() - slope * xs.mean())
    residuals = ys - (slope * xs + intercept)
    sse = float(np.sum(residuals * residuals))
    s = float(np.sqrt(sse / (n - 2)))
    return LinearFit(
        slope=slope,
        intercept=intercept,
        r_squared=float(np.clip(1.0 - sse / sst, 0.0, 1.0)),
        slope_stderr=s / float(np.sqrt(sxx)),
        intercept_stderr=s * float(np.sqrt(1.0 / n + xs.mean() ** 2 / sxx)),
        residual_std=s,
        n=int(n),
    )
