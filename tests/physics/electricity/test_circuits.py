import pytest

from scientificlab.physics.electricity import (
    electric_energy,
    electric_power,
    ohms_law,
    parallel_resistance,
    series_resistance,
)
from scientificlab.utils.exceptions import (
    InvalidInputError,
    PhysicalInconsistencyError,
)


class TestOhmsLaw:
    def test_reference_example(self):
        result = ohms_law(voltage=12, resistance=4)
        assert result.current == pytest.approx(3.0)
        assert result.power == pytest.approx(36.0)

    def test_each_quantity_can_be_the_unknown(self):
        assert ohms_law(current=2, resistance=5).voltage == pytest.approx(10.0)
        assert ohms_law(voltage=12, current=3).resistance == pytest.approx(4.0)

    def test_reversed_polarity_keeps_power_positive(self):
        result = ohms_law(voltage=-12, resistance=4)
        assert result.current == pytest.approx(-3.0)
        assert result.power == pytest.approx(36.0)

    def test_negative_voltage_and_current_give_positive_resistance(self):
        assert ohms_law(voltage=-12, current=-3).resistance == pytest.approx(4.0)

    def test_zero_voltage_zero_current_with_known_resistance(self):
        result = ohms_law(voltage=0, resistance=10)
        assert result.current == 0.0
        assert result.power == 0.0

    @pytest.mark.parametrize(
        ("voltage", "current"), [(12, -3), (-12, 3), (12, 0), (0, 3)]
    )
    def test_negative_or_infinite_or_zero_resistance_is_inconsistent(
        self, voltage, current
    ):
        with pytest.raises(PhysicalInconsistencyError):
            ohms_law(voltage=voltage, current=current)

    def test_undetermined_resistance(self):
        with pytest.raises(InvalidInputError, match="undetermined"):
            ohms_law(voltage=0, current=0)

    @pytest.mark.parametrize("resistance", [0, -4, float("nan"), float("inf"), "4"])
    def test_invalid_resistance(self, resistance):
        with pytest.raises(InvalidInputError):
            ohms_law(voltage=12, resistance=resistance)

    @pytest.mark.parametrize(
        "kwargs",
        [{}, {"voltage": 12}, {"voltage": 12, "current": 3, "resistance": 4}],
    )
    def test_exactly_two_quantities_required(self, kwargs):
        with pytest.raises(InvalidInputError, match="exactly 2"):
            ohms_law(**kwargs)


class TestPower:
    def test_three_formulas_agree(self):
        assert electric_power(voltage=12, current=3) == pytest.approx(36.0)
        assert electric_power(current=3, resistance=4) == pytest.approx(36.0)
        assert electric_power(voltage=12, resistance=4) == pytest.approx(36.0)

    def test_source_delivering_power_is_negative_with_vi(self):
        assert electric_power(voltage=12, current=-3) == pytest.approx(-36.0)

    def test_resistive_formulas_never_negative(self):
        assert electric_power(voltage=-5, resistance=2) > 0
        assert electric_power(current=-5, resistance=2) > 0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"voltage": 12},
            {"voltage": 12, "current": 3, "resistance": 4},
            {"voltage": 12, "resistance": 0},
            {"current": 1, "resistance": -1},
            {"voltage": float("nan"), "current": 1},
        ],
    )
    def test_invalid_arguments(self, kwargs):
        with pytest.raises(InvalidInputError):
            electric_power(**kwargs)


class TestEnergy:
    def test_hundred_watts_for_an_hour(self):
        assert electric_energy(100, 3600) == pytest.approx(360_000.0)  # 0.1 kWh

    @pytest.mark.parametrize("time", [-1, float("nan")])
    def test_invalid_time(self, time):
        with pytest.raises(InvalidInputError):
            electric_energy(100, time)


class TestResistorNetworks:
    def test_series(self):
        assert series_resistance([10, 20, 30]) == pytest.approx(60.0)
        assert series_resistance([5]) == 5.0
        assert series_resistance([5, 0]) == 5.0  # ideal wire

    def test_parallel_reference_values(self):
        assert parallel_resistance([10, 10]) == pytest.approx(5.0)
        assert parallel_resistance([6, 3]) == pytest.approx(2.0)
        assert parallel_resistance([100]) == pytest.approx(100.0)

    def test_n_equal_resistors_in_parallel(self):
        assert parallel_resistance([12.0] * 4) == pytest.approx(3.0)

    def test_parallel_is_below_the_smallest_branch(self):
        values = [3.0, 7.0, 11.0]
        assert parallel_resistance(values) < min(values)
        assert series_resistance(values) > max(values)

    def test_short_circuit_in_parallel_gives_zero(self):
        assert parallel_resistance([10, 0, 5]) == 0.0

    def test_voltage_divider_consistency(self):
        # Series current times each resistance sums to the source voltage.
        resistances = [100.0, 220.0, 330.0]
        current = ohms_law(
            voltage=13.0, resistance=series_resistance(resistances)
        ).current
        drops = [current * r for r in resistances]
        assert sum(drops) == pytest.approx(13.0)

    @pytest.mark.parametrize("bad", [[], "123", [1, -2], [1, float("nan")], 5, None])
    def test_invalid_input(self, bad):
        with pytest.raises(InvalidInputError):
            series_resistance(bad)
        with pytest.raises(InvalidInputError):
            parallel_resistance(bad)
