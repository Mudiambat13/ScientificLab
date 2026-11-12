import math

import numpy as np
import pytest
from scipy import stats

from scientificlab.mathematics import statistics as st
from scientificlab.utils.exceptions import InvalidInputError

CLASSIC = [2, 4, 4, 4, 5, 5, 7, 9]  # mean 5, population standard deviation 2
RNG = np.random.default_rng(12345)
SAMPLE = RNG.normal(10.0, 3.0, 50)
OTHER = 2.0 * SAMPLE + RNG.normal(0.0, 1.0, 50)


class TestDescriptive:
    def test_classic_dataset(self):
        assert st.mean(CLASSIC) == 5.0
        assert st.median(CLASSIC) == 4.5
        assert st.variance(CLASSIC, ddof=0) == 4.0
        assert st.standard_deviation(CLASSIC, ddof=0) == 2.0
        assert st.variance(CLASSIC) == pytest.approx(32 / 7)

    def test_matches_numpy_and_scipy(self):
        assert st.mean(SAMPLE) == pytest.approx(np.mean(SAMPLE))
        assert st.variance(SAMPLE) == pytest.approx(stats.tvar(SAMPLE))
        assert st.standard_deviation(SAMPLE) == pytest.approx(np.std(SAMPLE, ddof=1))
        assert st.median(SAMPLE) == pytest.approx(np.median(SAMPLE))

    def test_median_odd_and_even(self):
        assert st.median([3, 1, 2]) == 2.0
        assert st.median([1, 2, 3, 4]) == 2.5

    @pytest.mark.parametrize("q", [0.0, 0.1, 0.25, 0.5, 0.9, 1.0])
    def test_quantile_matches_numpy(self, q):
        assert st.quantile(SAMPLE, q) == pytest.approx(np.quantile(SAMPLE, q))

    def test_quantile_known_values(self):
        assert st.quantile([1, 2, 3, 4, 5], 0.5) == 3.0
        assert st.quantile([1, 2, 3, 4], 0.25) == 1.75  # type 7
        assert st.quantile([7], 0.3) == 7.0

    def test_describe(self):
        s = st.describe(CLASSIC)
        assert (s.count, s.mean, s.median, s.minimum, s.maximum) == (8, 5.0, 4.5, 2, 9)
        assert s.q1 == 4.0 and s.q3 == 5.5
        assert s.interquartile_range == 1.5
        assert s.standard_deviation == pytest.approx(math.sqrt(32 / 7))

    def test_constant_sample_has_zero_variance(self):
        assert st.variance([3.0, 3.0, 3.0]) == 0.0

    def test_translation_and_scaling_properties(self):
        shifted = [v + 1000.0 for v in CLASSIC]
        assert st.variance(shifted) == pytest.approx(st.variance(CLASSIC))
        scaled = [3.0 * v for v in CLASSIC]
        assert st.standard_deviation(scaled) == pytest.approx(
            3.0 * st.standard_deviation(CLASSIC)
        )

    def test_numerical_stability_with_large_offset(self):
        data = [1e9 + 4, 1e9 + 7, 1e9 + 13, 1e9 + 16]
        assert st.variance(data) == pytest.approx(30.0)

    def test_covariance_and_correlation(self):
        assert st.covariance(SAMPLE, OTHER) == pytest.approx(
            np.cov(SAMPLE, OTHER)[0, 1]
        )
        assert st.covariance(SAMPLE, OTHER, ddof=0) == pytest.approx(
            np.cov(SAMPLE, OTHER, ddof=0)[0, 1]
        )
        assert st.correlation(SAMPLE, OTHER) == pytest.approx(
            stats.pearsonr(SAMPLE, OTHER).statistic
        )

    def test_perfect_correlations(self):
        x = [1, 2, 3, 4]
        assert st.correlation(x, [2, 4, 6, 8]) == pytest.approx(1.0)
        assert st.correlation(x, [8, 6, 4, 2]) == pytest.approx(-1.0)

    @pytest.mark.parametrize(
        ("call", "message"),
        [
            (lambda: st.variance([5.0]), "ddof"),
            (lambda: st.variance([1.0, 2.0], ddof=-1), "non-negative"),
            (lambda: st.variance([1.0, 2.0], ddof=True), "non-negative"),
            (lambda: st.describe([1.0]), "at least 2"),
            (lambda: st.quantile([1, 2], 1.5), "between 0 and 1"),
            (lambda: st.quantile([1, 2], -0.1), "between 0 and 1"),
            (lambda: st.covariance([1, 2], [1, 2, 3]), "same length"),
            (lambda: st.correlation([1, 2, 3], [4, 4, 4]), "undefined"),
            (lambda: st.correlation([2, 2, 2], [1, 2, 3]), "undefined"),
            (lambda: st.correlation([1], [1]), "at least 2"),
            (lambda: st.mean([]), "empty"),
            (lambda: st.mean([1, float("nan")]), "finite"),
            (lambda: st.mean([[1, 2], [3, 4]]), "dimension"),
            (lambda: st.median(["a", "b"]), "numeric"),
        ],
    )
    def test_invalid_input(self, call, message):
        with pytest.raises(InvalidInputError, match=message):
            call()


class TestRegression:
    X = (1, 2, 3, 4, 5)
    Y = (2.1, 3.9, 6.2, 7.8, 10.1)

    def test_matches_scipy_linregress(self):
        fit = st.linear_regression(self.X, self.Y)
        ref = stats.linregress(self.X, self.Y)
        assert fit.slope == pytest.approx(ref.slope)
        assert fit.intercept == pytest.approx(ref.intercept)
        assert fit.r_squared == pytest.approx(ref.rvalue**2)
        assert fit.slope_stderr == pytest.approx(ref.stderr)
        assert fit.intercept_stderr == pytest.approx(ref.intercept_stderr)
        assert fit.n == 5

    def test_hand_computed_values(self):
        fit = st.linear_regression(self.X, self.Y)
        assert fit.slope == pytest.approx(1.99)
        assert fit.intercept == pytest.approx(0.05)

    def test_exact_line_is_recovered_with_zero_error(self):
        x = np.linspace(0, 10, 11)
        fit = st.linear_regression(x, 3.0 * x - 2.0)
        assert fit.slope == pytest.approx(3.0)
        assert fit.intercept == pytest.approx(-2.0)
        assert fit.r_squared == pytest.approx(1.0)
        assert fit.residual_std == pytest.approx(0.0, abs=1e-12)

    def test_noisy_large_sample_matches_scipy(self):
        ref = stats.linregress(SAMPLE, OTHER)
        fit = st.linear_regression(SAMPLE, OTHER)
        assert fit.slope == pytest.approx(ref.slope)
        assert fit.slope_stderr == pytest.approx(ref.stderr)
        assert fit.r_squared == pytest.approx(st.correlation(SAMPLE, OTHER) ** 2)

    def test_residual_std_definition(self):
        fit = st.linear_regression(self.X, self.Y)
        residuals = np.array(self.Y) - fit.predict(self.X)
        assert fit.residual_std == pytest.approx(math.sqrt(np.sum(residuals**2) / 3))

    def test_predict(self):
        fit = st.linear_regression(self.X, self.Y)
        assert fit.predict([0, 10]) == pytest.approx([0.05, 19.95])

    @pytest.mark.parametrize(
        ("x", "y", "message"),
        [
            ([1, 2], [1, 2], "at least 3"),
            ([1, 2, 3], [1, 2], "same length"),
            ([2, 2, 2], [1, 2, 3], "x is constant"),
            ([1, 2, 3], [5, 5, 5], "y is constant"),
            ([1, 2, float("inf")], [1, 2, 3], "finite"),
        ],
    )
    def test_invalid(self, x, y, message):
        with pytest.raises(InvalidInputError, match=message):
            st.linear_regression(x, y)


class TestInference:
    def test_confidence_interval_matches_scipy(self):
        ci = st.confidence_interval_mean(SAMPLE, 0.95)
        lo, hi = stats.t.interval(
            0.95, df=SAMPLE.size - 1, loc=np.mean(SAMPLE), scale=stats.sem(SAMPLE)
        )
        assert ci.lower == pytest.approx(lo) and ci.upper == pytest.approx(hi)
        assert ci.mean == pytest.approx(np.mean(SAMPLE))
        assert ci.margin == pytest.approx((hi - lo) / 2)
        assert ci.n == 50 and ci.level == 0.95

    def test_hand_value(self):
        # n=8, mean 5, s=2.13809; t(0.975, 7)=2.364624 -> margin 1.78749
        ci = st.confidence_interval_mean(CLASSIC)
        assert ci.margin == pytest.approx(1.787488, abs=1e-6)

    def test_higher_level_gives_wider_interval(self):
        narrow = st.confidence_interval_mean(SAMPLE, 0.80)
        wide = st.confidence_interval_mean(SAMPLE, 0.99)
        assert wide.margin > narrow.margin

    def test_coverage_of_the_true_mean(self):
        rng = np.random.default_rng(7)
        trials, covered = 2000, 0
        for _ in range(trials):
            ci = st.confidence_interval_mean(rng.normal(5.0, 2.0, 20), 0.95)
            covered += ci.lower <= 5.0 <= ci.upper
        assert covered / trials == pytest.approx(0.95, abs=0.02)

    @pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.5, True])
    def test_invalid_level(self, level):
        with pytest.raises(InvalidInputError, match="level"):
            st.confidence_interval_mean(CLASSIC, level)

    def test_too_few_values(self):
        with pytest.raises(InvalidInputError, match="at least 2"):
            st.confidence_interval_mean([1.0])

    def test_normal_distribution_reference_values(self):
        assert st.normal_pdf(0.0) == pytest.approx(1 / math.sqrt(2 * math.pi))
        assert st.normal_cdf(0.0) == 0.5
        assert st.normal_cdf(1.96) == pytest.approx(0.9750021048517795)
        assert st.normal_cdf(1.0) - st.normal_cdf(-1.0) == pytest.approx(0.6826894921)
        assert st.normal_quantile(0.975) == pytest.approx(1.959963984540054)
        assert st.normal_quantile(0.5, mean=3.0, std=2.0) == pytest.approx(3.0)

    def test_normal_matches_scipy_and_roundtrip(self):
        for x in (-3.0, -0.5, 0.0, 1.2, 4.0):
            assert st.normal_pdf(x, 1.0, 2.0) == pytest.approx(
                stats.norm.pdf(x, 1.0, 2.0)
            )
            assert st.normal_cdf(x, 1.0, 2.0) == pytest.approx(
                stats.norm.cdf(x, 1.0, 2.0)
            )
        for p in (0.001, 0.3, 0.5, 0.99):
            assert st.normal_cdf(
                st.normal_quantile(p, 5.0, 3.0), 5.0, 3.0
            ) == pytest.approx(p)

    def test_normal_cdf_tail_does_not_lose_precision(self):
        assert st.normal_cdf(-10.0) == pytest.approx(7.619853024160527e-24, rel=1e-12)

    def test_normal_invalid(self):
        with pytest.raises(InvalidInputError):
            st.normal_pdf(0.0, std=0.0)
        with pytest.raises(InvalidInputError):
            st.normal_cdf(float("nan"))
        with pytest.raises(InvalidInputError):
            st.normal_quantile(1.0)
        with pytest.raises(InvalidInputError):
            st.normal_quantile(0.5, std=-1.0)
