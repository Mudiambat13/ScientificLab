"""Linear and polynomial interpolation of tabulated data."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike

from scientificlab.utils.exceptions import InvalidInputError
from scientificlab.utils.validation import as_vector, require_finite


def _check_data(
    xs: ArrayLike, ys: ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    x_values = as_vector(xs, "xs")
    y_values = as_vector(ys, "ys")
    if x_values.shape != y_values.shape:
        raise InvalidInputError(
            "xs and ys must have the same length, "
            f"got {x_values.size} and {y_values.size}."
        )
    return x_values, y_values


def linear_interpolation(
    xs: ArrayLike,
    ys: ArrayLike,
    x: float,
    *,
    extrapolate: bool = False,
) -> float:
    """Piecewise-linear interpolation through the points ``(xs[i], ys[i])``.

    Within a segment the error is ``O(h**2)`` for a smooth underlying function
    (``h`` = segment length); the interpolant is continuous but not smooth.

    Args:
        xs: Strictly increasing abscissae (at least two).
        ys: Values at ``xs``.
        x: Query point.
        extrapolate: If ``False`` (default), a query outside ``[xs[0], xs[-1]]``
            raises; if ``True`` the first/last segment is extended linearly,
            which is unreliable far from the data.

    Raises:
        InvalidInputError: On invalid data, non-increasing ``xs``, or an
            out-of-range query when ``extrapolate`` is ``False``.
    """
    x_values, y_values = _check_data(xs, ys)
    if x_values.size < 2:
        raise InvalidInputError("At least two points are required.")
    if np.any(np.diff(x_values) <= 0.0):
        raise InvalidInputError("xs must be strictly increasing.")
    query = require_finite(x, "x")
    if not extrapolate and not x_values[0] <= query <= x_values[-1]:
        raise InvalidInputError(
            f"x = {query} is outside the data range [{x_values[0]}, {x_values[-1]}]; "
            "pass extrapolate=True to extend the end segments."
        )
    index = int(np.clip(np.searchsorted(x_values, query) - 1, 0, x_values.size - 2))
    x_left, x_right = x_values[index], x_values[index + 1]
    fraction = (query - x_left) / (x_right - x_left)
    return float(y_values[index] + fraction * (y_values[index + 1] - y_values[index]))


@dataclass(frozen=True)
class PolynomialInterpolant:
    """Interpolating polynomial in Newton form (divided differences).

    Attributes:
        nodes: Interpolation nodes ``x_0..x_n``.
        coefficients: Divided differences ``c_k = f[x_0..x_k]``; the polynomial
            is ``c_0 + c_1 (x - x_0) + c_2 (x - x_0)(x - x_1) + ...``.
    """

    nodes: npt.NDArray[np.float64]
    coefficients: npt.NDArray[np.float64]

    @property
    def degree(self) -> int:
        """Degree of the polynomial (number of nodes minus one)."""
        return int(self.nodes.size - 1)

    def __call__(self, x: float | ArrayLike) -> float | npt.NDArray[np.float64]:
        """Evaluate the polynomial (scalar in, float out; array in, array out)."""
        points = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(points)):
            raise InvalidInputError("x must contain only finite values.")
        result = np.full_like(points, self.coefficients[-1], dtype=np.float64)
        for k in range(self.nodes.size - 2, -1, -1):
            result = result * (points - self.nodes[k]) + self.coefficients[k]
        return float(result) if result.ndim == 0 else result


def polynomial_interpolation(xs: ArrayLike, ys: ArrayLike) -> PolynomialInterpolant:
    """Build the unique polynomial of degree ``n`` through ``n + 1`` points.

    Limits: the polynomial reproduces the data exactly at the nodes, but between
    nodes the error can be large. With many equally spaced nodes (degree above
    about 10 to 15) the Runge phenomenon produces wild oscillations near the
    ends, and the divided differences become ill-conditioned. Extrapolating
    outside the node range is unreliable. For many points prefer
    :func:`linear_interpolation` or splines.

    Args:
        xs: Distinct nodes (at least one; order is irrelevant).
        ys: Values at the nodes.

    Returns:
        A callable :class:`PolynomialInterpolant`.

    Raises:
        InvalidInputError: On invalid data or duplicate nodes.
    """
    x_values, y_values = _check_data(xs, ys)
    if np.unique(x_values).size != x_values.size:
        raise InvalidInputError("Interpolation nodes must be distinct.")
    coefficients = y_values.copy()
    for order in range(1, x_values.size):
        coefficients[order:] = (coefficients[order:] - coefficients[order - 1 : -1]) / (
            x_values[order:] - x_values[:-order]
        )
    return PolynomialInterpolant(nodes=x_values, coefficients=coefficients)
