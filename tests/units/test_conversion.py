import pytest

from scientificlab.units import (
    Q_,
    check_dimension,
    convert,
    convert_quantity,
    make_quantity,
    parse_unit,
)
from scientificlab.utils.exceptions import (
    IncompatibleUnitsError,
    InvalidInputError,
    PhysicalInconsistencyError,
    ScientificLabError,
    UnitError,
    UnknownUnitError,
)


@pytest.mark.parametrize(
    ("value", "src", "dst", "expected"),
    [
        (100, "km/h", "m/s", 100 / 3.6),
        (1, "atm", "Pa", 101325.0),
        (1, "bar", "Pa", 1e5),
        (1, "kWh", "J", 3.6e6),
        (1, "hour", "s", 3600.0),
        (1, "inch", "m", 0.0254),
        (1, "mile", "km", 1.609344),
        (1, "eV", "J", 1.602176634e-19),
        (0, "degC", "K", 273.15),
        (100, "degC", "K", 373.15),
        (212, "degF", "degC", 100.0),
        (300, "K", "degC", 26.85),
    ],
)
def test_reference_conversions(value, src, dst, expected):
    result = convert(value, src, dst)
    assert result.magnitude == pytest.approx(expected, rel=1e-12, abs=1e-9)


def test_convert_returns_quantity_that_prints_nicely():
    assert str(convert(100, "km/h", "m/s")) == "27.77777777777778 m/s"


def test_round_trip():
    back = convert(convert(37.5, "mph", "m/s").magnitude, "m/s", "mph")
    assert back.magnitude == pytest.approx(37.5, rel=1e-12)


def test_incompatible_dimensions():
    with pytest.raises(IncompatibleUnitsError, match="different dimensions"):
        convert(1, "m", "s")
    with pytest.raises(IncompatibleUnitsError):
        convert(1, "J", "W")


def test_unknown_and_malformed_units():
    with pytest.raises(UnknownUnitError, match="foo"):
        convert(1, "foo", "m")
    with pytest.raises(UnknownUnitError):
        parse_unit("m //")
    with pytest.raises(UnknownUnitError):
        parse_unit("")
    with pytest.raises(UnknownUnitError):
        parse_unit(3)  # type: ignore[arg-type]


def test_exception_hierarchy():
    assert issubclass(IncompatibleUnitsError, UnitError)
    assert issubclass(UnknownUnitError, UnitError)
    assert issubclass(UnitError, ScientificLabError)
    assert issubclass(UnitError, ValueError)


@pytest.mark.parametrize(
    ("value", "unit"), [(-0.1, "K"), (-300, "degC"), (-500, "degF")]
)
def test_below_absolute_zero_is_rejected(value, unit):
    with pytest.raises(PhysicalInconsistencyError, match="absolute zero"):
        make_quantity(value, unit)


def test_absolute_zero_itself_and_conversion_result_check():
    assert make_quantity(0, "K").magnitude == 0.0
    assert convert(-273.15, "degC", "K").magnitude == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(PhysicalInconsistencyError):
        convert(-10, "K", "degC")


def test_temperature_difference_may_be_negative():
    assert make_quantity(-5, "delta_degC").magnitude == -5.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "3", True, None])
def test_make_quantity_rejects_bad_values(bad):
    with pytest.raises(InvalidInputError):
        make_quantity(bad, "m")


def test_check_dimension():
    check_dimension(Q_(3, "km"), "m")
    with pytest.raises(IncompatibleUnitsError):
        check_dimension(Q_(3, "s"), "m")


def test_celsius_to_delta_is_refused():
    """An absolute degC reading is not a temperature difference."""
    with pytest.raises(IncompatibleUnitsError):
        convert_quantity(Q_(20, "degC"), "delta_degC")
    assert convert_quantity(Q_(20, "K"), "delta_degC").magnitude == pytest.approx(20)
