"""Heat and temperature (SI units: J, kg, K, J/(kg*K), J/kg).

Conventions: heat is positive when received by the body. A temperature
*change* has the same value in kelvin and degrees Celsius.
"""

from scientificlab.constants import ZERO_CELSIUS
from scientificlab.utils.exceptions import InvalidInputError, PhysicalInconsistencyError
from scientificlab.utils.validation import require_finite, require_positive

_UNITS = ("K", "C", "F")
_FAHRENHEIT_OFFSET = 32.0
_FAHRENHEIT_RATIO = 9.0 / 5.0


def heat_energy(mass: float, specific_heat: float, temperature_change: float) -> float:
    """Sensible heat exchanged: ``Q = m * c * delta_T`` (J).

    Assumes a constant specific heat over the temperature range and no phase
    change (use :func:`latent_heat` for that).

    Args:
        mass: Mass in kg (> 0).
        specific_heat: Specific heat capacity in J/(kg*K) (> 0); e.g. liquid
            water is about 4186.
        temperature_change: ``T_final - T_initial`` in K or degrees C; negative
            when the body cools (heat released, ``Q < 0``).
    """
    return (
        require_positive(mass, "mass")
        * require_positive(specific_heat, "specific_heat")
        * require_finite(temperature_change, "temperature_change")
    )


def latent_heat(mass: float, specific_latent_heat: float) -> float:
    """Heat of a phase change at constant temperature: ``Q = m * L`` (J).

    Args:
        mass: Mass that changes phase, in kg (> 0).
        specific_latent_heat: Specific latent heat ``L`` in J/kg. Pass a positive
            value for melting/vaporisation (heat absorbed) and a negative
            value for freezing/condensation (heat released).
    """
    return require_positive(mass, "mass") * require_finite(
        specific_latent_heat, "specific_latent_heat"
    )


def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """Convert an absolute temperature between kelvin, Celsius and Fahrenheit.

    Args:
        value: Temperature in ``from_unit``.
        from_unit: ``"K"``, ``"C"`` or ``"F"``.
        to_unit: ``"K"``, ``"C"`` or ``"F"``.

    Raises:
        InvalidInputError: On an unknown unit or non-finite value.
        PhysicalInconsistencyError: If the temperature is below absolute zero
            (0 K = -273.15 C = -459.67 F).
    """
    if from_unit not in _UNITS or to_unit not in _UNITS:
        raise InvalidInputError(
            f"Units must be one of {_UNITS}, got {from_unit!r}, {to_unit!r}."
        )
    number = require_finite(value, "value")
    kelvin = _to_kelvin(number, from_unit)
    if kelvin < 0.0:
        raise PhysicalInconsistencyError(
            f"{number} {from_unit} is below absolute zero."
        )
    return _from_kelvin(kelvin, to_unit)


def _to_kelvin(value: float, unit: str) -> float:
    if unit == "K":
        return value
    if unit == "C":
        return value + ZERO_CELSIUS
    return (value - _FAHRENHEIT_OFFSET) / _FAHRENHEIT_RATIO + ZERO_CELSIUS


def _from_kelvin(kelvin: float, unit: str) -> float:
    if unit == "K":
        return kelvin
    celsius = kelvin - ZERO_CELSIUS
    if unit == "C":
        return celsius
    return celsius * _FAHRENHEIT_RATIO + _FAHRENHEIT_OFFSET
