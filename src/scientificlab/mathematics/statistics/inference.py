"""Confidence intervals and the normal distribution."""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import special, stats

from scientificlab.utils.exceptions import InvalidInputError
from scientificlab.utils.validation import as_vector, require_finite, require_positive


@dataclass(frozen=True)
class ConfidenceInterval:
    """A two-sided confidence interval for a mean.

    Attributes:
        mean: Sample mean.
        lower: Lower bound.
        upper: Upper bound.
        margin: Half-width ``t * s / sqrt(n)``.
        level: Confidence level, e.g. 0.95.
        n: Sample size.
    """

    mean: float
    lower: float
    upper: float
    margin: float
    level: float
    n: int


def confidence_interval_mean(
    data: ArrayLike, level: float = 0.95
) -> ConfidenceInterval:
    """Student-t confidence interval for the mean of a sample.

    Valid when the data are independent and roughly normally distributed (or the
    sample is large). The interval is for the **mean**, not for individual
    values.

    Args:
        data: At least two values.
        level: Confidence level strictly between 0 and 1.

    Raises:
        InvalidInputError: If ``level`` is not in ``(0, 1)``, there are fewer than
            two values, or the data are invalid.
    """
    values = as_vector(data, "data")
    if isinstance(level, bool) or not 0.0 < level < 1.0:
        raise InvalidInputError(
            f"level must be strictly between 0 and 1, got {level!r}."
        )
    n = values.size
    if n < 2:
        raise InvalidInputError("confidence_interval_mean needs at least 2 values.")
    centre = float(np.mean(values))
    sem = float(np.std(values, ddof=1)) / math.sqrt(n)
    margin = float(stats.t.ppf(0.5 + level / 2.0, df=n - 1)) * sem
    return ConfidenceInterval(
        centre, centre - margin, centre + margin, margin, level, int(n)
    )


def normal_pdf(x: float, mean: float = 0.0, std: float = 1.0) -> float:
    """Probability density of the normal distribution ``N(mean, std**2)``.

    Raises:
        InvalidInputError: If ``std <= 0`` or an argument is not finite.
    """
    z = (require_finite(x, "x") - require_finite(mean, "mean")) / require_positive(
        std, "std"
    )
    return math.exp(-0.5 * z * z) / (std * math.sqrt(2.0 * math.pi))


def normal_cdf(x: float, mean: float = 0.0, std: float = 1.0) -> float:
    """Cumulative distribution ``P(X <= x)`` of ``N(mean, std**2)``.

    Raises:
        InvalidInputError: If ``std <= 0`` or an argument is not finite.
    """
    z = (require_finite(x, "x") - require_finite(mean, "mean")) / require_positive(
        std, "std"
    )
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def normal_quantile(p: float, mean: float = 0.0, std: float = 1.0) -> float:
    """Inverse of :func:`normal_cdf`: the ``x`` with ``P(X <= x) = p``.

    Raises:
        InvalidInputError: If ``p`` is not strictly between 0 and 1 or ``std <= 0``.
    """
    if isinstance(p, bool) or not 0.0 < p < 1.0:
        raise InvalidInputError(f"p must be strictly between 0 and 1, got {p!r}.")
    return require_finite(mean, "mean") + require_positive(std, "std") * float(
        special.ndtri(p)
    )
