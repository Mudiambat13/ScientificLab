"""Cross-check the constants against SciPy's independent CODATA table."""

import pytest
import scipy.constants as sc

from scientificlab import constants

SCIPY_HAS_CODATA_2022 = abs(sc.epsilon_0 - 8.8541878188e-12) < 1e-21
needs_codata_2022 = pytest.mark.skipif(
    not SCIPY_HAS_CODATA_2022,
    reason="this SciPy ships CODATA 2018 (SciPy >= 1.15 has CODATA 2022)",
)


@needs_codata_2022
def test_exact_constants_match_codata():
    assert sc.c == constants.SPEED_OF_LIGHT
    assert sc.g == constants.STANDARD_GRAVITY
    assert sc.atm == constants.STANDARD_ATMOSPHERE
    assert constants.GAS_CONSTANT == sc.R
    assert sc.zero_Celsius == constants.ZERO_CELSIUS


def test_gas_constant_is_avogadro_times_boltzmann():
    assert sc.N_A * sc.k == constants.GAS_CONSTANT


@needs_codata_2022
def test_vacuum_permittivity_matches_codata_value():
    assert sc.epsilon_0 == constants.VACUUM_PERMITTIVITY


def test_vacuum_permittivity_relation_to_light_speed_and_permeability():
    # eps_0 = 1 / (mu_0 * c**2), consistent to the experimental precision (~1e-10).
    derived = 1.0 / (sc.mu_0 * constants.SPEED_OF_LIGHT**2)
    assert abs(derived - constants.VACUUM_PERMITTIVITY) / derived < 1e-9
