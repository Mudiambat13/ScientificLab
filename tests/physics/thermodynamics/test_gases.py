import math

import pytest

from scientificlab.physics.thermodynamics import (
    adiabatic_pressure,
    adiabatic_work,
    ideal_gas,
    isobaric_work,
    isothermal_work,
)
from scientificlab.utils.exceptions import InvalidInputError

GAS_CONSTANT = 8.31446261815324


class TestIdealGas:
    def test_molar_volume_at_stp(self):
        # 1 mol at 273.15 K and 101325 Pa occupies 22.414 L.
        state = ideal_gas(pressure=101_325, amount=1, temperature=273.15)
        assert state.volume == pytest.approx(0.022413969545, rel=1e-9)

    def test_each_quantity_can_be_solved(self):
        reference = ideal_gas(pressure=2e5, volume=0.01, temperature=300.0)
        assert reference.amount == pytest.approx(2e5 * 0.01 / (GAS_CONSTANT * 300.0))
        p = ideal_gas(volume=0.01, amount=reference.amount, temperature=300.0)
        t = ideal_gas(pressure=2e5, volume=0.01, amount=reference.amount)
        v = ideal_gas(pressure=2e5, amount=reference.amount, temperature=300.0)
        assert p.pressure == pytest.approx(2e5)
        assert t.temperature == pytest.approx(300.0)
        assert v.volume == pytest.approx(0.01)

    def test_state_satisfies_the_law(self):
        state = ideal_gas(pressure=1.5e5, volume=0.02, amount=0.8)
        assert state.pressure * state.volume == pytest.approx(
            state.amount * GAS_CONSTANT * state.temperature
        )

    def test_boyle_and_charles_laws(self):
        base = ideal_gas(pressure=1e5, amount=1.0, temperature=300.0)
        compressed = ideal_gas(pressure=2e5, amount=1.0, temperature=300.0)
        assert compressed.volume == pytest.approx(base.volume / 2)
        heated = ideal_gas(pressure=1e5, amount=1.0, temperature=600.0)
        assert heated.volume == pytest.approx(2 * base.volume)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"pressure": 1e5, "volume": 0.01},
            {"pressure": 1e5},
            {},
            {"pressure": 1e5, "volume": 0.01, "amount": 1.0, "temperature": 300.0},
        ],
    )
    def test_exactly_three_quantities_required(self, kwargs):
        with pytest.raises(InvalidInputError, match="exactly 3"):
            ideal_gas(**kwargs)

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("pressure", 0),
            ("pressure", -1e5),
            ("volume", 0),
            ("amount", -1),
            ("temperature", 0),
            ("temperature", -10),
            ("temperature", float("nan")),
        ],
    )
    def test_non_positive_values_are_rejected(self, name, value):
        base = {"pressure": 1e5, "volume": 0.01, "amount": 1.0, "temperature": 300.0}
        base.pop("amount" if name != "amount" else "pressure")
        base[name] = value
        with pytest.raises(InvalidInputError):
            ideal_gas(**base)


class TestWork:
    def test_isobaric(self):
        assert isobaric_work(101_325, 0.01) == pytest.approx(1013.25)
        assert isobaric_work(101_325, -0.01) == pytest.approx(-1013.25)
        assert isobaric_work(101_325, 0.0) == 0.0

    def test_isothermal_reference_value(self):
        # Doubling the volume of 1 mol at 300 K: R*T*ln(2) = 1728.94 J.
        assert isothermal_work(1.0, 300.0, 1.0, 2.0) == pytest.approx(
            1728.9438964931937
        )

    def test_isothermal_compression_is_negative_and_symmetric(self):
        assert isothermal_work(1.0, 300.0, 2.0, 1.0) == pytest.approx(
            -1728.9438964931937
        )
        assert isothermal_work(1.0, 300.0, 3.0, 3.0) == 0.0

    def test_isothermal_work_equals_integral_of_pressure(self):
        # Trapezoid integration of P(V) = nRT/V over [1, 2] m**3 (independent check).
        n, t, steps = 1.0, 300.0, 20_000
        volumes = [1.0 + k / steps for k in range(steps + 1)]
        pressures = [n * GAS_CONSTANT * t / v for v in volumes]
        integral = sum(
            0.5 * (pressures[k] + pressures[k + 1]) * (volumes[k + 1] - volumes[k])
            for k in range(steps)
        )
        assert isothermal_work(n, t, 1.0, 2.0) == pytest.approx(integral, rel=1e-8)

    def test_adiabatic_pressure_reference_value(self):
        assert adiabatic_pressure(1e5, 1e-3, 2e-3, 5 / 3) == pytest.approx(
            31_498.02624737183
        )

    def test_adiabatic_drop_is_steeper_than_isothermal(self):
        adiabatic = adiabatic_pressure(1e5, 1e-3, 2e-3, 1.4)
        isothermal = 1e5 * 1e-3 / 2e-3
        assert adiabatic < isothermal

    def test_adiabatic_work_obeys_the_first_law(self):
        # Monatomic gas: delta_U = (3/2) * (P_f V_f - P_i V_i) and W = -delta_U (Q = 0).
        p_i, v_i, v_f = 1e5, 1e-3, 2e-3
        p_f = adiabatic_pressure(p_i, v_i, v_f, 5 / 3)
        delta_u = 1.5 * (p_f * v_f - p_i * v_i)
        assert adiabatic_work(p_i, v_i, v_f, 5 / 3) == pytest.approx(-delta_u)
        assert adiabatic_work(p_i, v_i, v_f, 5 / 3) > 0  # expansion: gas does work

    def test_adiabatic_expansion_cools_the_gas(self):
        p_i, v_i, v_f = 1e5, 1e-3, 2e-3
        p_f = adiabatic_pressure(p_i, v_i, v_f, 1.4)
        assert p_f * v_f < p_i * v_i  # PV proportional to T

    def test_adiabatic_compression_is_negative_work(self):
        assert adiabatic_work(1e5, 2e-3, 1e-3, 1.4) < 0

    def test_adiabatic_without_volume_change(self):
        assert adiabatic_work(1e5, 1e-3, 1e-3, 1.4) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("gamma", [1.0, 0.9, -1.4, float("nan"), float("inf")])
    def test_invalid_gamma(self, gamma):
        with pytest.raises(InvalidInputError):
            adiabatic_work(1e5, 1e-3, 2e-3, gamma)
        with pytest.raises(InvalidInputError):
            adiabatic_pressure(1e5, 1e-3, 2e-3, gamma)

    @pytest.mark.parametrize(
        "call",
        [
            lambda: isobaric_work(0, 0.01),
            lambda: isobaric_work(1e5, float("nan")),
            lambda: isothermal_work(0, 300, 1, 2),
            lambda: isothermal_work(1, -300, 1, 2),
            lambda: isothermal_work(1, 300, 0, 2),
            lambda: isothermal_work(1, 300, 1, -2),
            lambda: adiabatic_work(-1e5, 1e-3, 2e-3, 1.4),
            lambda: adiabatic_pressure(1e5, 0, 2e-3, 1.4),
        ],
    )
    def test_invalid_inputs(self, call):
        with pytest.raises(InvalidInputError):
            call()

    def test_isothermal_log_is_natural(self):
        assert isothermal_work(1.0, 1.0, 1.0, math.e) == pytest.approx(GAS_CONSTANT)
