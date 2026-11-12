"""Descriptive statistics on one or two samples (NumPy, numeric results).

All functions validate their input (finite real numbers, non-empty, 1-D) and
raise :class:`InvalidInputError` when a statistic is undefined for the data
(for example the sample variance of a single value, or the correlation with a
constant sample) instead of returning ``nan``.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike

from scientificlab.utils.exceptions import InvalidInputError
from scientificlab.utils.validation import as_vector


def _ddof(ddof: int, n: int) -> int:
    if isinstance(ddof, bool) or not isinstance(ddof, int) or ddof < 0:
        raise InvalidInputError("ddof must be a non-negative integer.")
    if n - ddof < 1:
        raise InvalidInputError(
            f"Need more than {ddof} value(s) for ddof={ddof}, got {n}."
        )
    return ddof


def mean(data: ArrayLike) -> float:
    """Arithmetic mean.

    Raises:
        InvalidInputError: If ``data`` is empty, not 1-D or not finite.
    """
    return float(np.mean(as_vector(data, "data")))


def median(data: ArrayLike) -> float:
    """Median (average of the two middle values for an even count).

    Raises:
        InvalidInputError: If ``data`` is empty, not 1-D or not finite.
    """
    return float(np.median(as_vector(data, "data")))


def variance(data: ArrayLike, ddof: int = 1) -> float:
    """Variance with divisor ``n - ddof``.

    Args:
        data: The sample.
        ddof: ``1`` (default) gives the unbiased **sample** variance; ``0`` the
            **population** variance.

    Raises:
        InvalidInputError: If ``n - ddof < 1`` (for example a single value with
            ``ddof=1``) or the data are invalid.
    """
    values = as_vector(data, "data")
    return float(np.var(values, ddof=_ddof(ddof, values.size)))


def standard_deviation(data: ArrayLike, ddof: int = 1) -> float:
    """Square root of :func:`variance` (sample standard deviation by default)."""
    values = as_vector(data, "data")
    return float(np.std(values, ddof=_ddof(ddof, values.size)))


def quantile(data: ArrayLike, q: float) -> float:
    """Quantile of probability ``q`` in ``[0, 1]`` by linear interpolation.

    This is the "type 7" definition (NumPy and R default): the value at
    position ``q * (n - 1)`` of the sorted data, interpolated linearly.

    Raises:
        InvalidInputError: If ``q`` is outside ``[0, 1]`` or the data are invalid.
    """
    values = as_vector(data, "data")
    if isinstance(q, bool) or not 0.0 <= q <= 1.0:
        raise InvalidInputError(f"q must be between 0 and 1, got {q!r}.")
    return float(np.quantile(values, q))


@dataclass(frozen=True)
class Summary:
    """Result of :func:`describe`.

    Attributes:
        count: Number of values.
        mean: Arithmetic mean.
        median: Median.
        standard_deviation: Sample standard deviation (``ddof=1``).
        minimum: Smallest value.
        maximum: Largest value.
        q1: First quartile (type 7).
        q3: Third quartile (type 7).
    """

    count: int
    mean: float
    median: float
    standard_deviation: float
    minimum: float
    maximum: float
    q1: float
    q3: float

    @property
    def interquartile_range(self) -> float:
        """``q3 - q1``."""
        return self.q3 - self.q1


def describe(data: ArrayLike) -> Summary:
    """Summary statistics of a sample (at least two values).

    Raises:
        InvalidInputError: If there are fewer than two values or the data are
            invalid.
    """
    values = as_vector(data, "data")
    if values.size < 2:
        raise InvalidInputError("describe needs at least 2 values.")
    q1, q3 = np.quantile(values, [0.25, 0.75])
    return Summary(
        count=int(values.size),
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        standard_deviation=float(np.std(values, ddof=1)),
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
        q1=float(q1),
        q3=float(q3),
    )


def _pair(
    x: ArrayLike, y: ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    xs = as_vector(x, "x")
    ys = as_vector(y, "y")
    if xs.size != ys.size:
        raise InvalidInputError(
            f"x and y must have the same length, got {xs.size} and {ys.size}."
        )
    return xs, ys


def covariance(x: ArrayLike, y: ArrayLike, ddof: int = 1) -> float:
    """Covariance of two samples with divisor ``n - ddof``.

    Raises:
        InvalidInputError: If lengths differ or ``n - ddof < 1``.
    """
    xs, ys = _pair(x, y)
    d = _ddof(ddof, xs.size)
    return float(np.sum((xs - xs.mean()) * (ys - ys.mean())) / (xs.size - d))


def correlation(x: ArrayLike, y: ArrayLike) -> float:
    """Pearson correlation coefficient, in ``[-1, 1]``.

    It measures **linear** association only, and says nothing about causation.

    Raises:
        InvalidInputError: If lengths differ, there are fewer than two points, or
            one sample is constant (the coefficient is then undefined).
    """
    xs, ys = _pair(x, y)
    if xs.size < 2:
        raise InvalidInputError("correlation needs at least 2 points.")
    dx, dy = xs - xs.mean(), ys - ys.mean()
    sxx, syy = float(np.sum(dx * dx)), float(np.sum(dy * dy))
    if sxx == 0.0 or syy == 0.0:
        raise InvalidInputError(
            "The correlation is undefined when one of the samples is constant."
        )
    return float(np.clip(np.sum(dx * dy) / np.sqrt(sxx * syy), -1.0, 1.0))
