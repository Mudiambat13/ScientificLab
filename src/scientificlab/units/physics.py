"""Unit-aware wrappers around the float-based physics functions.

The core physics functions take plain floats in SI units. :func:`with_units`
turns such a function into one that accepts Pint quantities (any compatible
unit), checks dimensions, converts to SI, calls the float function and returns
a quantity with the proper SI unit. Plain numbers are still accepted and are
interpreted as SI, exactly as in the core functions.

A temperature *difference* parameter is declared with ``"delta_degC"``, so an
absolute ``degC`` reading is rejected there instead of being silently
mistaken for a difference.
"""

import functools
import inspect
from collections.abc import Callable
from typing import Any

from scientificlab.physics.mechanics import energy, kinematics
from scientificlab.physics.thermodynamics import heat
from scientificlab.units.conversion import convert_quantity, parse_unit
from scientificlab.units.registry import Q_
from scientificlab.utils.exceptions import IncompatibleUnitsError, InvalidInputError


def _si_value(name: str, value: Any, si_unit: str) -> Any:
    if isinstance(value, bool):
        raise InvalidInputError(f"{name} must be a number or a quantity, got bool.")
    if isinstance(value, int | float):
        return value  # plain number: interpreted as SI
    if hasattr(value, "to") and hasattr(value, "magnitude"):
        try:
            return float(convert_quantity(value, si_unit).magnitude)
        except IncompatibleUnitsError as exc:
            raise IncompatibleUnitsError(
                f"{name}: expected a quantity in {si_unit}-compatible units, "
                f"got {value.units}."
            ) from exc
    raise InvalidInputError(f"{name} must be a number or a quantity, got {value!r}.")


def with_units(
    inputs: dict[str, str], output: str
) -> Callable[[Callable[..., float]], Callable[..., Any]]:
    """Decorate a float/SI function so it accepts and returns Pint quantities.

    Args:
        inputs: Maps parameter names to their SI unit string. Parameters not
            listed (e.g. dimensionless coefficients) are passed through.
        output: SI unit of the returned value.

    Raises:
        UnknownUnitError: At decoration time, if a unit string is invalid.
    """
    for unit in (*inputs.values(), output):
        parse_unit(unit)

    def decorator(func: Callable[..., float]) -> Callable[..., Any]:
        signature = inspect.signature(func)
        for name in inputs:
            if name not in signature.parameters:
                raise ValueError(f"{func.__name__} has no parameter {name!r}")

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = signature.bind(*args, **kwargs)
            for name, value in list(bound.arguments.items()):
                if name in inputs:
                    bound.arguments[name] = _si_value(name, value, inputs[name])
            return Q_(func(*bound.args, **bound.kwargs), output)

        wrapper.__doc__ = (
            f"{func.__doc__ or ''}\n\n    Unit-aware version: parameters accept Pint "
            f"quantities; returns a quantity in {output}."
        )
        return wrapper

    return decorator


kinetic_energy = with_units({"mass": "kg", "velocity": "m/s"}, "J")(
    energy.kinetic_energy
)
potential_energy = with_units({"mass": "kg", "height": "m", "gravity": "m/s**2"}, "J")(
    energy.potential_energy
)
average_speed = with_units({"distance": "m", "time": "s"}, "m/s")(
    kinematics.average_speed
)
distance_traveled = with_units({"speed": "m/s", "time": "s"}, "m")(
    kinematics.distance_traveled
)
free_fall_time = with_units({"height": "m", "gravity": "m/s**2"}, "s")(
    kinematics.free_fall_time
)
heat_energy = with_units(
    {
        "mass": "kg",
        "specific_heat": "J/(kg*K)",
        "temperature_change": "delta_degC",
    },
    "J",
)(heat.heat_energy)

__all__ = [
    "average_speed",
    "distance_traveled",
    "free_fall_time",
    "heat_energy",
    "kinetic_energy",
    "potential_energy",
    "with_units",
]
