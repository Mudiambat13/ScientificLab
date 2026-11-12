import pytest

from scientificlab.physics.thermodynamics import (
    carnot_efficiency,
    efficiency_from_heats,
    thermal_efficiency,
)
from scientificlab.utils.exceptions import (
    InvalidInputError,
    PhysicalInconsistencyError,
)


class TestCarnot:
    def test_reference_value(self):
        assert carnot_efficiency(500, 300) == pytest.approx(0.4)

    def test_grows_with_the_temperature_gap(self):
        assert carnot_efficiency(600, 300) > carnot_efficiency(500, 300)
        assert 0.0 < carnot_efficiency(1000, 1) < 1.0

    @pytest.mark.parametrize(("hot", "cold"), [(300, 300), (300, 500)])
    def test_hot_must_exceed_cold(self, hot, cold):
        with pytest.raises(PhysicalInconsistencyError):
            carnot_efficiency(hot, cold)

    @pytest.mark.parametrize(
        ("hot", "cold"), [(500, 0), (0, 300), (-500, 300), (500, -1)]
    )
    def test_temperatures_must_be_positive_kelvin(self, hot, cold):
        with pytest.raises(InvalidInputError):
            carnot_efficiency(hot, cold)


class TestThermalEfficiency:
    def test_reference_value(self):
        assert thermal_efficiency(400, 1000) == pytest.approx(0.4)

    def test_no_work_means_zero_efficiency(self):
        assert thermal_efficiency(0, 1000) == 0.0

    def test_all_heat_converted_is_the_energy_limit(self):
        assert thermal_efficiency(1000, 1000) == 1.0

    def test_energy_conservation_violation_is_reported(self):
        with pytest.raises(PhysicalInconsistencyError, match="energy"):
            thermal_efficiency(1200, 1000)

    def test_engine_at_the_carnot_limit_is_accepted(self):
        assert thermal_efficiency(400, 1000, 500, 300) == pytest.approx(0.4)

    def test_engine_below_the_carnot_limit_is_accepted(self):
        assert thermal_efficiency(300, 1000, 500, 300) == pytest.approx(0.3)

    def test_second_law_violation_is_reported(self):
        with pytest.raises(PhysicalInconsistencyError, match="Carnot"):
            thermal_efficiency(500, 1000, 500, 300)

    def test_both_temperatures_or_none(self):
        with pytest.raises(InvalidInputError):
            thermal_efficiency(300, 1000, hot_temperature=500)
        with pytest.raises(InvalidInputError):
            thermal_efficiency(300, 1000, cold_temperature=300)

    @pytest.mark.parametrize(
        ("work", "heat"), [(-1, 1000), (100, 0), (100, -5), (float("nan"), 1000)]
    )
    def test_invalid_inputs(self, work, heat):
        with pytest.raises(InvalidInputError):
            thermal_efficiency(work, heat)


class TestEfficiencyFromHeats:
    def test_reference_value(self):
        assert efficiency_from_heats(1000, 600) == pytest.approx(0.4)

    def test_consistent_with_work_balance(self):
        heat_in, heat_out = 1000.0, 650.0
        work = heat_in - heat_out
        assert efficiency_from_heats(heat_in, heat_out) == pytest.approx(
            thermal_efficiency(work, heat_in)
        )

    def test_rejected_heat_cannot_exceed_absorbed_heat(self):
        with pytest.raises(PhysicalInconsistencyError):
            efficiency_from_heats(1000, 1200)

    def test_invalid_inputs(self):
        with pytest.raises(InvalidInputError):
            efficiency_from_heats(0, 0)
        with pytest.raises(InvalidInputError):
            efficiency_from_heats(1000, -1)
