import math

import pytest

from scientificlab.physics.optics import (
    critical_angle,
    reflection_angle,
    snell_refraction,
)
from scientificlab.utils.exceptions import (
    InvalidInputError,
    PhysicalInconsistencyError,
)

GLASS_CRITICAL = 41.810314895778596  # asin(1/1.5) in degrees


class TestSnell:
    def test_air_to_glass_reference_value(self):
        result = snell_refraction(1.0, 1.5, 30.0)
        assert result.refraction_angle == pytest.approx(19.47122063449069)
        assert not result.total_internal_reflection
        assert result.critical_angle is None

    def test_glass_to_air_bends_away_from_the_normal(self):
        result = snell_refraction(1.5, 1.0, 30.0)
        assert result.refraction_angle == pytest.approx(48.590377890729144)
        assert result.refraction_angle > 30.0
        assert result.critical_angle == pytest.approx(GLASS_CRITICAL)

    def test_normal_incidence_is_not_deviated(self):
        assert snell_refraction(1.0, 1.5, 0.0).refraction_angle == 0.0

    def test_equal_indices_do_not_deviate(self):
        assert snell_refraction(1.33, 1.33, 37.0).refraction_angle == pytest.approx(
            37.0
        )

    def test_total_internal_reflection(self):
        result = snell_refraction(1.5, 1.0, 45.0)
        assert result.total_internal_reflection
        assert result.refraction_angle is None
        assert result.reflection_angle == 45.0

    def test_exactly_at_the_critical_angle_grazes(self):
        result = snell_refraction(1.5, 1.0, critical_angle(1.5, 1.0))
        assert not result.total_internal_reflection
        assert result.refraction_angle == pytest.approx(90.0)

    def test_just_beyond_the_critical_angle_reflects_totally(self):
        assert snell_refraction(
            1.5, 1.0, GLASS_CRITICAL + 0.01
        ).total_internal_reflection

    def test_grazing_incidence_into_denser_medium_reaches_the_critical_angle(self):
        result = snell_refraction(1.0, 1.5, 90.0)
        assert result.refraction_angle == pytest.approx(GLASS_CRITICAL)

    def test_reversibility_of_light_paths(self):
        forward = snell_refraction(1.0, 1.5, 30.0)
        back = snell_refraction(1.5, 1.0, forward.refraction_angle)
        assert back.refraction_angle == pytest.approx(30.0)

    def test_snell_invariant_holds(self):
        result = snell_refraction(1.0, 1.33, 52.0)
        lhs = 1.0 * math.sin(math.radians(52.0))
        rhs = 1.33 * math.sin(math.radians(result.refraction_angle))
        assert lhs == pytest.approx(rhs)

    @pytest.mark.parametrize("angle", [-0.1, 90.5, 180, float("nan"), float("inf")])
    def test_invalid_angle(self, angle):
        with pytest.raises(InvalidInputError):
            snell_refraction(1.0, 1.5, angle)

    @pytest.mark.parametrize(("n1", "n2"), [(0, 1.5), (1.0, -1.5), (float("nan"), 1.0)])
    def test_invalid_indices(self, n1, n2):
        with pytest.raises(InvalidInputError):
            snell_refraction(n1, n2, 30.0)


class TestCriticalAngle:
    def test_glass_air(self):
        assert critical_angle(1.5, 1.0) == pytest.approx(GLASS_CRITICAL)

    def test_diamond_has_a_smaller_critical_angle_than_glass(self):
        assert critical_angle(2.42, 1.0) < critical_angle(1.5, 1.0)

    @pytest.mark.parametrize(("n1", "n2"), [(1.0, 1.5), (1.5, 1.5)])
    def test_no_critical_angle_into_a_denser_or_equal_medium(self, n1, n2):
        with pytest.raises(PhysicalInconsistencyError):
            critical_angle(n1, n2)


def test_reflection_angle_equals_incidence_angle():
    assert reflection_angle(30.0) == 30.0
    with pytest.raises(InvalidInputError):
        reflection_angle(120.0)
