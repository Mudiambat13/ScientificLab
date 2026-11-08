import numpy as np
import pytest

from scientificlab.utils.exceptions import InvalidInputError, ScientificLabError
from scientificlab.utils.validation import (
    require_finite,
    require_non_negative,
    require_positive,
)


def test_accepts_int_float_and_numpy_scalars():
    assert require_finite(3, "x") == 3.0
    assert require_finite(np.float64(2.5), "x") == 2.5
    assert require_finite(np.int64(4), "x") == 4.0


@pytest.mark.parametrize("bad", [True, "1", None, 1j, float("nan"), float("inf")])
def test_rejects_non_real_or_non_finite(bad):
    with pytest.raises(InvalidInputError):
        require_finite(bad, "x")


def test_positive_and_non_negative_boundaries():
    assert require_non_negative(0, "x") == 0.0
    with pytest.raises(InvalidInputError):
        require_positive(0, "x")
    with pytest.raises(InvalidInputError):
        require_non_negative(-1e-300, "x")


def test_error_hierarchy():
    assert issubclass(InvalidInputError, ScientificLabError)
    assert issubclass(InvalidInputError, ValueError)


def test_error_message_names_the_argument():
    with pytest.raises(InvalidInputError, match="mass"):
        require_positive(-1, "mass")
