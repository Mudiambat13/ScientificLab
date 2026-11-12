"""Registry of physical constants with value, unit, uncertainty and source.

Every constant records its provenance, so results can be traced:

* **Exact** constants (uncertainty 0) are fixed by definition: the 2019 SI
  redefinition (``c``, ``h``, ``k_B``, ``e``, ``N_A`` and quantities derived
  from them such as ``R``, ``F``, ``hbar``, ``sigma``), the 1901 CGPM
  (``g_n``) and the 1954 CGPM (standard atmosphere).
* **Measured** constants come from the CODATA 2022 adjustment published by NIST
  (Mohr, Newell, Taylor, Tiesinga, 2024) and carry their standard uncertainty.

Values are cross-checked against SciPy's independent CODATA table in the test
suite. Units are Pint-compatible strings.
"""

from dataclasses import dataclass

from scientificlab.utils.exceptions import InvalidInputError

CODATA_EDITION: str = "CODATA 2022"
"""Edition of the CODATA adjustment used for the measured constants."""

_SI_2019 = "SI 2019 (exact by definition)"
_CODATA_2022 = "CODATA 2022 (NIST)"


@dataclass(frozen=True)
class Constant:
    """A physical constant.

    Attributes:
        key: Python-friendly identifier, e.g. ``"speed_of_light"``.
        name: Official name, e.g. ``"speed of light in vacuum"``.
        symbol: Conventional symbol, e.g. ``"c"``.
        value: Numerical value in ``unit``.
        unit: Unit as a Pint-compatible string (``"dimensionless"`` if none).
        uncertainty: Absolute standard uncertainty in ``unit``; 0 if exact.
        source: Origin of the value.
        edition: Edition or definition the value belongs to.
    """

    key: str
    name: str
    symbol: str
    value: float
    unit: str
    uncertainty: float
    source: str
    edition: str

    @property
    def exact(self) -> bool:
        """``True`` if the value is exact by definition (no uncertainty)."""
        return self.uncertainty == 0.0

    @property
    def relative_uncertainty(self) -> float:
        """Standard uncertainty divided by ``abs(value)`` (0 if exact)."""
        return self.uncertainty / abs(self.value)

    @property
    def quantity(self):  # type: ignore[no-untyped-def]
        """The constant as a Pint quantity (value with its unit)."""
        from scientificlab.units import Q_

        return Q_(self.value, self.unit)

    def __str__(self) -> str:
        """Human-readable one-line description with uncertainty and source."""
        spread = "exact" if self.exact else f"+/- {self.uncertainty:.2g}"
        return (
            f"{self.name} ({self.symbol}) = {self.value!r} {self.unit} "
            f"[{spread}; {self.edition}]"
        )


def _exact(
    key: str, name: str, symbol: str, value: float, unit: str, source: str = _SI_2019
) -> Constant:
    return Constant(key, name, symbol, value, unit, 0.0, source, source)


def _measured(
    key: str, name: str, symbol: str, value: float, unit: str, uncertainty: float
) -> Constant:
    return Constant(
        key, name, symbol, value, unit, uncertainty, _CODATA_2022, CODATA_EDITION
    )


_ALL: tuple[Constant, ...] = (
    _exact("speed_of_light", "speed of light in vacuum", "c", 299_792_458.0, "m / s"),
    _measured(
        "gravitational_constant",
        "Newtonian constant of gravitation",
        "G",
        6.6743e-11,
        "m ** 3 / (kg * s ** 2)",
        1.5e-15,
    ),
    _exact("planck_constant", "Planck constant", "h", 6.62607015e-34, "J * s"),
    _exact(
        "reduced_planck_constant",
        "reduced Planck constant",
        "hbar",
        1.0545718176461565e-34,
        "J * s",
    ),
    _exact("boltzmann_constant", "Boltzmann constant", "k_B", 1.380649e-23, "J / K"),
    _exact("elementary_charge", "elementary charge", "e", 1.602176634e-19, "C"),
    _measured("electron_mass", "electron mass", "m_e", 9.1093837139e-31, "kg", 2.8e-40),
    _measured("proton_mass", "proton mass", "m_p", 1.67262192595e-27, "kg", 5.2e-37),
    _measured("neutron_mass", "neutron mass", "m_n", 1.67492750056e-27, "kg", 8.5e-37),
    _exact("avogadro_constant", "Avogadro constant", "N_A", 6.02214076e23, "1 / mol"),
    _exact(
        "gas_constant",
        "molar gas constant",
        "R",
        8.31446261815324,
        "J / (mol * K)",
    ),
    _exact(
        "standard_gravity",
        "standard acceleration of gravity",
        "g_n",
        9.80665,
        "m / s ** 2",
        source="3rd CGPM, 1901 (exact by definition)",
    ),
    _measured(
        "vacuum_permittivity",
        "vacuum electric permittivity",
        "epsilon_0",
        8.8541878188e-12,
        "F / m",
        1.4e-21,
    ),
    _measured(
        "vacuum_permeability",
        "vacuum magnetic permeability",
        "mu_0",
        1.25663706127e-06,
        "N / A ** 2",
        2.0e-16,
    ),
    _exact(
        "stefan_boltzmann_constant",
        "Stefan-Boltzmann constant",
        "sigma",
        5.6703744191844314e-08,
        "W / (m ** 2 * K ** 4)",
    ),
    _measured(
        "fine_structure_constant",
        "fine-structure constant",
        "alpha",
        0.0072973525643,
        "dimensionless",
        1.1e-12,
    ),
    _exact("faraday_constant", "Faraday constant", "F", 96485.33212331001, "C / mol"),
    _exact(
        "standard_atmosphere",
        "standard atmosphere",
        "atm",
        101_325.0,
        "Pa",
        source="10th CGPM, 1954 (exact by definition)",
    ),
)

CONSTANTS: dict[str, Constant] = {constant.key: constant for constant in _ALL}
"""All constants, indexed by their ``key``."""


def get_constant(identifier: str) -> Constant:
    """Look a constant up by key, official name or symbol (case-insensitive).

    Args:
        identifier: e.g. ``"speed_of_light"`` (``"speed of light"`` also works),
            ``"speed of light in vacuum"`` or ``"c"``. Symbols match exactly
            first, then case-insensitively **only if unambiguous**.

    Raises:
        InvalidInputError: If nothing, or more than one constant, matches.
    """
    if not isinstance(identifier, str) or not identifier.strip():
        raise InvalidInputError("The constant identifier must be a non-empty string.")
    wanted = identifier.strip().lower()
    spaced_key = wanted.replace(" ", "_").replace("-", "_")
    by_key = [
        c for c in _ALL if c.key in (wanted, spaced_key) or c.name.lower() == wanted
    ]
    if by_key:
        return by_key[0]
    by_symbol = [c for c in _ALL if c.symbol == identifier.strip()]
    if not by_symbol:
        by_symbol = [c for c in _ALL if c.symbol.lower() == wanted]
    if len(by_symbol) == 1:
        return by_symbol[0]
    if len(by_symbol) > 1:
        options = ", ".join(c.key for c in by_symbol)
        raise InvalidInputError(f"{identifier!r} is ambiguous; use one of: {options}.")
    raise InvalidInputError(
        f"Unknown constant {identifier!r}. Available: {', '.join(sorted(CONSTANTS))}."
    )


def list_constants() -> list[Constant]:
    """All registered constants, sorted by key."""
    return sorted(_ALL, key=lambda constant: constant.key)
