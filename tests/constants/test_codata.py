import dataclasses

import pytest
from scipy import constants as sc

from scientificlab.constants import (
    CONSTANTS,
    GAS_CONSTANT,
    SPEED_OF_LIGHT,
    STANDARD_GRAVITY,
    ZERO_CELSIUS,
    get_constant,
    list_constants,
)
from scientificlab.utils.exceptions import InvalidInputError

SCIPY_HAS_CODATA_2022 = abs(sc.epsilon_0 - 8.8541878188e-12) < 1e-21
needs_codata_2022 = pytest.mark.skipif(
    not SCIPY_HAS_CODATA_2022,
    reason="this SciPy ships CODATA 2018 (SciPy >= 1.15 has CODATA 2022)",
)


# (key, scipy.constants name): independent CODATA 2022 table.
SCIPY_REFERENCE = [
    ("speed_of_light", "speed of light in vacuum"),
    ("gravitational_constant", "Newtonian constant of gravitation"),
    ("planck_constant", "Planck constant"),
    ("reduced_planck_constant", "reduced Planck constant"),
    ("boltzmann_constant", "Boltzmann constant"),
    ("elementary_charge", "elementary charge"),
    ("electron_mass", "electron mass"),
    ("proton_mass", "proton mass"),
    ("neutron_mass", "neutron mass"),
    ("avogadro_constant", "Avogadro constant"),
    ("gas_constant", "molar gas constant"),
    ("vacuum_permittivity", "vacuum electric permittivity"),
    ("vacuum_permeability", "vacuum mag. permeability"),
    ("stefan_boltzmann_constant", "Stefan-Boltzmann constant"),
    ("fine_structure_constant", "fine-structure constant"),
    ("faraday_constant", "Faraday constant"),
    ("standard_gravity", "standard acceleration of gravity"),
    ("standard_atmosphere", "standard atmosphere"),
]


def test_every_constant_is_checked_against_scipy():
    assert {k for k, _ in SCIPY_REFERENCE} == set(CONSTANTS)


@needs_codata_2022
@pytest.mark.parametrize(("key", "scipy_name"), SCIPY_REFERENCE)
def test_value_matches_scipy_codata(key, scipy_name):
    reference, _unit, uncertainty = sc.physical_constants[scipy_name]
    constant = CONSTANTS[key]
    assert constant.value == pytest.approx(reference, rel=1e-12)
    assert constant.uncertainty == pytest.approx(uncertainty, rel=0.1, abs=0.0)


@pytest.mark.parametrize(
    "key", ["speed_of_light", "planck_constant", "boltzmann_constant"]
)
def test_si_defining_constants_are_exact(key):
    assert CONSTANTS[key].exact
    assert CONSTANTS[key].relative_uncertainty == 0.0


def test_measured_constant_is_not_exact():
    g = CONSTANTS["gravitational_constant"]
    assert not g.exact
    assert g.relative_uncertainty == pytest.approx(2.2e-5, rel=0.05)


def test_internal_consistency_of_derived_constants():
    c = CONSTANTS
    assert c["gas_constant"].value == pytest.approx(
        c["avogadro_constant"].value * c["boltzmann_constant"].value, rel=1e-12
    )
    assert c["faraday_constant"].value == pytest.approx(
        c["avogadro_constant"].value * c["elementary_charge"].value, rel=1e-12
    )
    # c**2 * mu_0 * epsilon_0 = 1 within the (relative 1.6e-10) uncertainty.
    product = (
        c["speed_of_light"].value ** 2
        * c["vacuum_permeability"].value
        * c["vacuum_permittivity"].value
    )
    assert product == pytest.approx(1.0, rel=1e-9)


def test_float_shortcuts_match_registry():
    assert SPEED_OF_LIGHT == 299_792_458.0
    assert STANDARD_GRAVITY == 9.80665
    assert CONSTANTS["gas_constant"].value == GAS_CONSTANT
    assert ZERO_CELSIUS == 273.15


@pytest.mark.parametrize(
    "identifier",
    ["speed_of_light", "speed of light", "SPEED OF LIGHT IN VACUUM", "c", " c "],
)
def test_get_constant_aliases(identifier):
    assert get_constant(identifier).key == "speed_of_light"


def test_symbols_are_case_sensitive_first():
    assert get_constant("G").key == "gravitational_constant"
    assert get_constant("R").key == "gas_constant"


def test_unknown_and_invalid_identifier():
    with pytest.raises(InvalidInputError, match="Unknown constant"):
        get_constant("not_a_constant")
    with pytest.raises(InvalidInputError):
        get_constant("")
    with pytest.raises(InvalidInputError):
        get_constant(5)  # type: ignore[arg-type]


def test_quantity_has_units():
    q = get_constant("speed_of_light").quantity
    assert q.magnitude == 299_792_458.0
    assert str(q.units) == "m/s"
    assert get_constant("alpha").quantity.dimensionless


def test_all_units_parse_in_pint():
    from scientificlab.units import parse_unit

    for constant in list_constants():
        parse_unit(constant.unit)


def test_str_and_list():
    assert "exact" in str(get_constant("c"))
    assert "+/-" in str(get_constant("G"))
    keys = [c.key for c in list_constants()]
    assert keys == sorted(keys)
    assert len(keys) == len(CONSTANTS)


def test_constant_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        CONSTANTS["speed_of_light"].value = 3.0  # type: ignore[misc]


def test_ambiguous_symbol_is_reported(monkeypatch):
    from scientificlab.constants import codata

    twin = dataclasses.replace(CONSTANTS["speed_of_light"], key="other_c")
    monkeypatch.setattr(codata, "_ALL", (*codata._ALL, twin))
    with pytest.raises(InvalidInputError, match="ambiguous"):
        codata.get_constant("c")
