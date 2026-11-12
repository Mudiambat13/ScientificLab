"""Physical constants (SI units) with source and uncertainty.

Two ways to use them:

* Plain floats for formulas: ``SPEED_OF_LIGHT``, ``GAS_CONSTANT``, ...
* Structured records for traceability: ``get_constant("speed of light")``
  returns a :class:`Constant` with value, unit, uncertainty, source and CODATA
  edition, and ``.quantity`` gives a Pint quantity.

Exact constants come from the 2019 SI redefinition or CGPM definitions;
measured ones from CODATA 2022. See :mod:`scientificlab.constants.codata`.
"""

from scientificlab.constants.codata import (
    CODATA_EDITION,
    CONSTANTS,
    Constant,
    get_constant,
    list_constants,
)

STANDARD_GRAVITY: float = CONSTANTS["standard_gravity"].value
"""Standard acceleration of gravity ``g_n`` in m/s**2 (exact, CGPM 1901)."""

SPEED_OF_LIGHT: float = CONSTANTS["speed_of_light"].value
"""Speed of light in vacuum ``c`` in m/s (exact, SI 2019)."""

GAS_CONSTANT: float = CONSTANTS["gas_constant"].value
"""Molar gas constant ``R = N_A * k_B`` in J/(mol*K) (exact, SI 2019)."""

VACUUM_PERMITTIVITY: float = CONSTANTS["vacuum_permittivity"].value
"""Vacuum electric permittivity ``epsilon_0`` in F/m (CODATA 2022, u = 1.4e-21)."""

STANDARD_ATMOSPHERE: float = CONSTANTS["standard_atmosphere"].value
"""Standard atmospheric pressure in Pa (exact, CGPM 1954)."""

ZERO_CELSIUS: float = 273.15
"""Kelvin value of 0 degrees Celsius (exact by definition of the Celsius scale)."""

__all__ = [
    "CODATA_EDITION",
    "CONSTANTS",
    "GAS_CONSTANT",
    "SPEED_OF_LIGHT",
    "STANDARD_ATMOSPHERE",
    "STANDARD_GRAVITY",
    "VACUUM_PERMITTIVITY",
    "ZERO_CELSIUS",
    "Constant",
    "get_constant",
    "list_constants",
]
