"""Statistics: descriptive measures, correlation, linear regression, intervals.

All results are numeric (NumPy / SciPy). Each function documents the estimator
used (for example sample variance with ``ddof=1``) and raises an explicit error
when a statistic is undefined for the data.
"""

from scientificlab.mathematics.statistics.descriptive import (
    Summary,
    correlation,
    covariance,
    describe,
    mean,
    median,
    quantile,
    standard_deviation,
    variance,
)
from scientificlab.mathematics.statistics.inference import (
    ConfidenceInterval,
    confidence_interval_mean,
    normal_cdf,
    normal_pdf,
    normal_quantile,
)
from scientificlab.mathematics.statistics.regression import (
    LinearFit,
    linear_regression,
)

__all__ = [
    "ConfidenceInterval",
    "LinearFit",
    "Summary",
    "confidence_interval_mean",
    "correlation",
    "covariance",
    "describe",
    "linear_regression",
    "mean",
    "median",
    "normal_cdf",
    "normal_pdf",
    "normal_quantile",
    "quantile",
    "standard_deviation",
    "variance",
]
