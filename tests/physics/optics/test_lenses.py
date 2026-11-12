import math

import pytest

from scientificlab.physics.optics import (
    NEAR_POINT,
    lens_power,
    lensmaker_focal_length,
    linear_magnification,
    magnifier_angular_magnification,
    telescope_magnification,
    thin_lens_equation,
)
from scientificlab.utils.exceptions import (
    InvalidInputError,
    PhysicalInconsistencyError,
)


class TestThinLens:
    def test_real_inverted_image(self):
        result = thin_lens_equation(focal_length=0.1, object_distance=0.3)
        assert result.image_distance == pytest.approx(0.15)
        assert result.magnification == pytest.approx(-0.5)
        assert result.real_image
        assert not result.upright

    def test_object_at_twice_the_focal_length_gives_a_same_size_image(self):
        result = thin_lens_equation(focal_length=0.1, object_distance=0.2)
        assert result.image_distance == pytest.approx(0.2)
        assert result.magnification == pytest.approx(-1.0)

    def test_magnifying_glass_gives_a_virtual_upright_enlarged_image(self):
        result = thin_lens_equation(focal_length=0.1, object_distance=0.05)
        assert result.image_distance == pytest.approx(-0.1)
        assert result.magnification == pytest.approx(2.0)
        assert not result.real_image
        assert result.upright

    def test_diverging_lens_always_gives_a_virtual_reduced_image(self):
        result = thin_lens_equation(focal_length=-0.1, object_distance=0.3)
        assert result.image_distance == pytest.approx(-0.075)
        assert result.magnification == pytest.approx(0.25)
        assert not result.real_image
        assert result.upright

    def test_solving_for_focal_length_and_object_distance(self):
        assert thin_lens_equation(
            object_distance=0.3, image_distance=0.15
        ).focal_length == pytest.approx(0.1)
        assert thin_lens_equation(
            focal_length=0.1, image_distance=0.15
        ).object_distance == pytest.approx(0.3)

    def test_the_lens_equation_holds_for_the_result(self):
        r = thin_lens_equation(focal_length=0.37, object_distance=0.91)
        assert 1 / r.object_distance + 1 / r.image_distance == pytest.approx(
            1 / r.focal_length
        )

    def test_object_at_the_focal_point_has_its_image_at_infinity(self):
        with pytest.raises(PhysicalInconsistencyError, match="infinity"):
            thin_lens_equation(focal_length=0.1, object_distance=0.1)

    def test_image_at_the_focal_point_needs_an_object_at_infinity(self):
        with pytest.raises(PhysicalInconsistencyError):
            thin_lens_equation(focal_length=0.1, image_distance=0.1)

    def test_opposite_object_and_image_distances_mean_infinite_focal_length(self):
        with pytest.raises(PhysicalInconsistencyError):
            thin_lens_equation(object_distance=0.3, image_distance=-0.3)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"focal_length": 0.1},
            {"focal_length": 0.1, "object_distance": 0.3, "image_distance": 0.15},
            {"focal_length": 0, "object_distance": 0.3},
            {"focal_length": 0.1, "object_distance": 0},
            {"focal_length": 0.1, "object_distance": float("nan")},
        ],
    )
    def test_invalid_arguments(self, kwargs):
        with pytest.raises(InvalidInputError):
            thin_lens_equation(**kwargs)


def test_linear_magnification():
    assert linear_magnification(0.15, 0.3) == pytest.approx(-0.5)
    assert linear_magnification(-0.1, 0.05) == pytest.approx(2.0)
    with pytest.raises(InvalidInputError):
        linear_magnification(0.0, 0.3)


def test_lens_power_in_diopters():
    assert lens_power(0.5) == pytest.approx(2.0)
    assert lens_power(-0.25) == pytest.approx(-4.0)
    with pytest.raises(InvalidInputError):
        lens_power(0.0)


class TestLensmaker:
    def test_biconvex_lens(self):
        assert lensmaker_focal_length(1.5, 0.2, -0.2) == pytest.approx(0.2)

    def test_plano_convex_lens(self):
        assert lensmaker_focal_length(1.5, 0.1, math.inf) == pytest.approx(0.2)
        assert lensmaker_focal_length(1.5, math.inf, -0.1) == pytest.approx(0.2)

    def test_biconcave_lens_is_diverging(self):
        assert lensmaker_focal_length(1.5, -0.2, 0.2) == pytest.approx(-0.2)

    def test_higher_index_means_a_stronger_lens(self):
        assert lensmaker_focal_length(1.7, 0.2, -0.2) < lensmaker_focal_length(
            1.5, 0.2, -0.2
        )

    def test_lens_in_water_is_weaker_than_in_air(self):
        in_air = lensmaker_focal_length(1.5, 0.2, -0.2)
        in_water = lensmaker_focal_length(1.5, 0.2, -0.2, medium_index=1.33)
        assert in_water > in_air
        assert in_water == pytest.approx(1 / ((1.5 / 1.33 - 1) * 10))

    @pytest.mark.parametrize(
        "args",
        [(1.5, math.inf, math.inf), (1.5, 0.2, 0.2), (1.0, 0.2, -0.2)],
    )
    def test_lens_without_optical_power(self, args):
        with pytest.raises(PhysicalInconsistencyError):
            lensmaker_focal_length(*args)

    @pytest.mark.parametrize(
        "args",
        [(1.5, 0, -0.2), (1.5, 0.2, float("nan")), (0, 0.2, -0.2), (1.5, True, 0.2)],
    )
    def test_invalid_arguments(self, args):
        with pytest.raises(InvalidInputError):
            lensmaker_focal_length(*args)


class TestInstruments:
    def test_magnifier(self):
        assert magnifier_angular_magnification(0.05) == pytest.approx(6.0)
        assert magnifier_angular_magnification(0.05, relaxed_eye=True) == pytest.approx(
            5.0
        )
        assert NEAR_POINT == 0.25

    def test_magnifier_with_another_near_point(self):
        assert magnifier_angular_magnification(0.1, near_point=0.2) == pytest.approx(
            3.0
        )

    def test_telescope(self):
        assert telescope_magnification(1.0, 0.02) == pytest.approx(50.0)

    @pytest.mark.parametrize(
        "call",
        [
            lambda: magnifier_angular_magnification(0),
            lambda: magnifier_angular_magnification(-0.05),
            lambda: magnifier_angular_magnification(0.05, near_point=0),
            lambda: telescope_magnification(1.0, 0),
            lambda: telescope_magnification(-1.0, 0.02),
        ],
    )
    def test_invalid_inputs(self, call):
        with pytest.raises(InvalidInputError):
            call()
