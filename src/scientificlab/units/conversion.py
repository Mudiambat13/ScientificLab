"""Unit conversion with explicit dimensional checks.

Conversions never guess: an unknown unit raises :class:`UnknownUnitError` and a
conversion between different physical dimensions raises
:class:`IncompatibleUnitsError`. Offset temperatures (``degC``, ``degF``) are
converted correctly (``0 degC -> 273.15 K``); a temperature *below absolute
zero* is rejected as :class:`PhysicalInconsistencyError`.
"""

import math
from typing import Any

import pint

from scientificlab.units.registry import Q_, ureg
from scientificlab.utils.exceptions import (
    IncompatibleUnitsError,
    InvalidInputError,
    PhysicalInconsistencyError,
    UnitError,
    UnknownUnitError,
)


def parse_unit(unit: str) -> Any:
    """Parse a unit string into a Pint unit.

    Raises:
        UnknownUnitError: If the string is not a valid unit expression.
    """
    if not isinstance(unit, str) or not unit.strip():
        raise UnknownUnitError("A unit must be a non-empty string.")
    try:
        return ureg.Unit(unit)
    except pint.errors.UndefinedUnitError as exc:
        raise UnknownUnitError(f"Unknown unit: {unit!r}.") from exc
    except Exception as exc:  # malformed expression such as "m //"
        raise UnknownUnitError(f"Cannot parse unit {unit!r}: {exc}") from exc


def make_quantity(value: float, unit: str) -> Any:
    """Build a Pint quantity ``value * unit`` after validating both.

    Raises:
        InvalidInputError: If ``value`` is not a finite real number.
        UnknownUnitError: If ``unit`` is unknown.
        PhysicalInconsistencyError: If it is an absolute temperature below 0 K.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvalidInputError(f"value must be a real number, got {value!r}.")
    if not math.isfinite(value):
        raise InvalidInputError(f"value must be finite, got {value!r}.")
    quantity = Q_(float(value), parse_unit(unit))
    _check_absolute_zero(quantity)
    return quantity


def _check_absolute_zero(quantity: Any) -> None:
    """Reject absolute temperatures below 0 K (Pint does not check this)."""
    is_temperature = quantity.dimensionality == ureg.kelvin.dimensionality
    if (
        is_temperature
        and not _is_delta(quantity)
        and float(quantity.to("kelvin").magnitude) < 0.0
    ):
        raise PhysicalInconsistencyError(
            f"{quantity} is below absolute zero (0 K); no such temperature exists."
        )


def _is_delta(quantity: Any) -> bool:
    return any(name.startswith("delta_") for name in quantity.units._units)


def convert(value: float, from_unit: str, to_unit: str) -> Any:
    """Convert ``value`` from ``from_unit`` to ``to_unit``.

    Returns:
        A Pint quantity, e.g. ``convert(100, "km/h", "m/s")`` prints as
        ``27.77777777777778 m/s``. Use ``.magnitude`` for the bare float.
        Offset-scale conversions go through floating point, so compare with a
        tolerance (``32 degF -> degC`` gives ~5.7e-14, not exactly 0).

    Raises:
        UnknownUnitError: If a unit is unknown.
        IncompatibleUnitsError: If the dimensions differ (e.g. m -> s).
        PhysicalInconsistencyError: For temperatures below absolute zero.
    """
    quantity = make_quantity(value, from_unit)
    return convert_quantity(quantity, to_unit)


def convert_quantity(quantity: Any, to_unit: str) -> Any:
    """Convert an existing Pint quantity to ``to_unit`` (checked)."""
    target = parse_unit(to_unit)
    try:
        result = quantity.to(target)
    except pint.errors.DimensionalityError as exc:
        raise IncompatibleUnitsError(
            f"Cannot convert {quantity.units} to {target}: different dimensions."
        ) from exc
    except pint.errors.PintError as exc:
        raise UnitError(f"Unit conversion failed: {exc}") from exc
    _check_absolute_zero(result)
    return result


def check_dimension(quantity: Any, expected_unit: str) -> None:
    """Check that ``quantity`` has the dimension of ``expected_unit``.

    Raises:
        IncompatibleUnitsError: If the dimensions differ.
    """
    convert_quantity(quantity, expected_unit)
