import pytest

from scientificlab.utils.exceptions import InvalidInputError
from scientificlab.utils.validation import require_n_given


def test_accepts_exactly_n_given():
    require_n_given({"a": 1.0, "b": None, "c": 2.0}, 2)
    require_n_given({"a": 0.0, "b": None}, 1)  # zero is a value, not "missing"


@pytest.mark.parametrize(
    "values", [{"a": None, "b": None}, {"a": 1, "b": 2, "c": 3}, {"a": 1}]
)
def test_rejects_wrong_count(values):
    with pytest.raises(InvalidInputError, match="exactly 2"):
        require_n_given(values, 2)


def test_message_lists_the_names():
    with pytest.raises(InvalidInputError, match=r"\(a, b, c\).*got 1: a"):
        require_n_given({"a": 1.0, "b": None, "c": None}, 2)
