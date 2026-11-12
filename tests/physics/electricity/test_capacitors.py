import pytest

from scientificlab.physics.electricity import (
    capacitance_from_charge,
    capacitor_charge,
    capacitor_energy,
    charge_from_current,
    parallel_plate_capacitance,
)
from scientificlab.utils.exceptions import (
    InvalidInputError,
    PhysicalInconsistencyError,
)


def test_charge_from_current():
    assert charge_from_current(2.0, 3.0) == pytest.approx(6.0)


def test_capacitor_reference_values():
    # 100 microfarad at 12 V.
    assert capacitor_charge(100e-6, 12) == pytest.approx(1.2e-3)
    assert capacitor_energy(100e-6, 12) == pytest.approx(7.2e-3)


def test_capacitance_round_trip():
    q = capacitor_charge(4.7e-6, 9.0)
    assert capacitance_from_charge(q, 9.0) == pytest.approx(4.7e-6)


def test_energy_formulas_agree():
    c, v = 2.2e-6, 5.0
    q = capacitor_charge(c, v)
    assert capacitor_energy(c, v) == pytest.approx(q * v / 2)
    assert capacitor_energy(c, v) == pytest.approx(q * q / (2 * c))


def test_reversed_polarity_stores_same_energy():
    assert capacitor_energy(1e-6, -10) == capacitor_energy(1e-6, 10)
    assert capacitor_charge(1e-6, -10) == pytest.approx(-1e-5)


def test_parallel_plate_reference_value():
    # 1 m**2 plates 1 mm apart in vacuum: C = eps_0 / 1e-3 = 8.854 nF.
    assert parallel_plate_capacitance(1.0, 1e-3) == pytest.approx(8.8541878188e-9)


def test_dielectric_multiplies_capacitance():
    vacuum = parallel_plate_capacitance(0.01, 1e-3)
    assert parallel_plate_capacitance(
        0.01, 1e-3, relative_permittivity=80
    ) == pytest.approx(80 * vacuum)


def test_capacitance_scales_with_geometry():
    base = parallel_plate_capacitance(1e-2, 1e-3)
    assert parallel_plate_capacitance(2e-2, 1e-3) == pytest.approx(2 * base)
    assert parallel_plate_capacitance(1e-2, 2e-3) == pytest.approx(base / 2)


def test_zero_voltage_leaves_capacitance_undetermined():
    with pytest.raises(InvalidInputError):
        capacitance_from_charge(1e-3, 0.0)


@pytest.mark.parametrize(("q", "v"), [(1e-3, -5.0), (-1e-3, 5.0), (0.0, 5.0)])
def test_non_positive_capacitance_is_inconsistent(q, v):
    with pytest.raises(PhysicalInconsistencyError):
        capacitance_from_charge(q, v)


@pytest.mark.parametrize(
    "call",
    [
        lambda: capacitor_charge(0, 5),
        lambda: capacitor_charge(-1e-6, 5),
        lambda: capacitor_energy(1e-6, float("nan")),
        lambda: parallel_plate_capacitance(0, 1e-3),
        lambda: parallel_plate_capacitance(1, 0),
        lambda: parallel_plate_capacitance(1, 1e-3, relative_permittivity=0),
        lambda: charge_from_current(1, -1),
    ],
)
def test_invalid_inputs(call):
    with pytest.raises(InvalidInputError):
        call()
