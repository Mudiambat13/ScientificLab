"""Ideal gases and thermodynamic work (SI units: Pa, m**3, mol, K, J).

Model: ideal gas (no intermolecular forces, point molecules), quasi-static
(reversible) processes. Real gases deviate at high pressure or near
condensation. Temperatures are absolute (kelvin).

Sign convention for work: ``W > 0`` is work done **by** the gas on its
surroundings (expansion); ``W < 0`` is work done on the gas (compression).
"""

import math
from dataclasses import dataclass

from scientificlab.constants import GAS_CONSTANT
from scientificlab.utils.exceptions import InvalidInputError
from scientificlab.utils.validation import (
    require_finite,
    require_n_given,
    require_positive,
)


@dataclass(frozen=True)
class IdealGasState:
    """State of an ideal gas satisfying ``P * V = n * R * T``.

    Attributes:
        pressure: Absolute pressure in Pa.
        volume: Volume in m**3.
        amount: Amount of substance in mol.
        temperature: Absolute temperature in K.
    """

    pressure: float
    volume: float
    amount: float
    temperature: float


def ideal_gas(
    pressure: float | None = None,
    volume: float | None = None,
    amount: float | None = None,
    temperature: float | None = None,
) -> IdealGasState:
    """Solve ``P * V = n * R * T`` for the one missing quantity.

    Exactly three of the four arguments must be given. All must be strictly
    positive (absolute pressure and temperature, real gas amount).

    Args:
        pressure: Absolute pressure in Pa (not gauge pressure).
        volume: Volume in m**3.
        amount: Amount of substance in mol.
        temperature: Absolute temperature in K (not degrees Celsius).

    Raises:
        InvalidInputError: If not exactly three values are given or a value is
            not a strictly positive finite number.
    """
    require_n_given(
        {
            "pressure": pressure,
            "volume": volume,
            "amount": amount,
            "temperature": temperature,
        },
        3,
    )
    p = None if pressure is None else require_positive(pressure, "pressure")
    v = None if volume is None else require_positive(volume, "volume")
    n = None if amount is None else require_positive(amount, "amount")
    t = None if temperature is None else require_positive(temperature, "temperature")
    if p is None:
        assert n is not None and t is not None and v is not None
        p = n * GAS_CONSTANT * t / v
    elif v is None:
        assert n is not None and t is not None
        v = n * GAS_CONSTANT * t / p
    elif n is None:
        assert t is not None
        n = p * v / (GAS_CONSTANT * t)
    else:
        t = p * v / (n * GAS_CONSTANT)
    assert t is not None and n is not None
    return IdealGasState(pressure=p, volume=v, amount=n, temperature=t)


def isobaric_work(pressure: float, volume_change: float) -> float:
    """Work at constant pressure: ``W = P * delta_V`` (J).

    Args:
        pressure: Constant absolute pressure in Pa (> 0).
        volume_change: ``V_final - V_initial`` in m**3 (negative: compression).
    """
    return require_positive(pressure, "pressure") * require_finite(
        volume_change, "volume_change"
    )


def isothermal_work(
    amount: float,
    temperature: float,
    initial_volume: float,
    final_volume: float,
) -> float:
    """Work of an ideal gas at constant temperature: ``W = n*R*T*ln(V_f/V_i)`` (J)."""
    return (
        require_positive(amount, "amount")
        * GAS_CONSTANT
        * require_positive(temperature, "temperature")
        * math.log(
            require_positive(final_volume, "final_volume")
            / require_positive(initial_volume, "initial_volume")
        )
    )


def _check_gamma(gamma: float) -> float:
    g = require_finite(gamma, "gamma")
    if g <= 1.0:
        raise InvalidInputError(
            f"gamma (heat capacity ratio) must be > 1, got {gamma!r}."
        )
    return g


def adiabatic_pressure(
    initial_pressure: float,
    initial_volume: float,
    final_volume: float,
    gamma: float,
) -> float:
    """Final pressure of a reversible adiabatic process: ``P V**gamma = const`` (Pa).

    Args:
        initial_pressure: Absolute pressure in Pa (> 0).
        initial_volume: Initial volume in m**3 (> 0).
        final_volume: Final volume in m**3 (> 0).
        gamma: Heat capacity ratio ``C_p/C_v`` (> 1), e.g. 5/3 for a
            monatomic gas and about 7/5 for a diatomic gas near room temperature.
    """
    g = _check_gamma(gamma)
    ratio = require_positive(initial_volume, "initial_volume") / require_positive(
        final_volume, "final_volume"
    )
    return require_positive(initial_pressure, "initial_pressure") * float(ratio**g)


def adiabatic_work(
    initial_pressure: float,
    initial_volume: float,
    final_volume: float,
    gamma: float,
) -> float:
    """Work of a reversible adiabatic process (J).

    ``W = (P_i V_i - P_f V_f) / (gamma - 1)``. Equals the loss of internal
    energy (``Q = 0``). See :func:`adiabatic_pressure` for the arguments.
    """
    g = _check_gamma(gamma)
    p_i = require_positive(initial_pressure, "initial_pressure")
    v_i = require_positive(initial_volume, "initial_volume")
    v_f = require_positive(final_volume, "final_volume")
    p_f = adiabatic_pressure(p_i, v_i, v_f, g)
    return (p_i * v_i - p_f * v_f) / (g - 1.0)
