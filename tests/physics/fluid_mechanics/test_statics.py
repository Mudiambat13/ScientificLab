import pytest

from scientificlab.physics.fluid_mechanics import (
    buoyant_force,
    density,
    floats,
    hydraulic_force,
    hydrostatic_pressure,
    pressure,
    submerged_fraction,
)
from scientificlab.physics.mechanics import weight
from scientificlab.utils.exceptions import (
    InvalidInputError,
    PhysicalInconsistencyError,
)

WATER = 1000.0
ATMOSPHERE = 101_325.0


def test_density_and_pressure():
    assert density(1000, 1.0) == pytest.approx(1000.0)
    assert pressure(100, 0.5) == pytest.approx(200.0)
    assert pressure(0, 0.5) == 0.0


class TestHydrostatic:
    def test_ten_metres_of_water(self):
        assert hydrostatic_pressure(WATER, 10.0) == pytest.approx(98_066.5)

    def test_absolute_pressure_adds_the_atmosphere(self):
        assert hydrostatic_pressure(
            WATER, 10.0, surface_pressure=ATMOSPHERE
        ) == pytest.approx(199_391.5)

    def test_ten_metres_of_water_is_about_one_atmosphere(self):
        assert hydrostatic_pressure(WATER, 10.0) / ATMOSPHERE == pytest.approx(
            0.968, abs=0.001
        )

    def test_surface_has_only_the_surface_pressure(self):
        assert (
            hydrostatic_pressure(WATER, 0.0, surface_pressure=ATMOSPHERE) == ATMOSPHERE
        )

    def test_pressure_is_linear_in_depth(self):
        assert hydrostatic_pressure(WATER, 20.0) == pytest.approx(
            2 * hydrostatic_pressure(WATER, 10.0)
        )

    def test_other_planet(self):
        assert hydrostatic_pressure(WATER, 10.0, gravity=1.62) == pytest.approx(
            16_200.0
        )

    def test_force_on_a_dam_wall_element_uses_pressure_times_area(self):
        assert pressure(hydrostatic_pressure(WATER, 5.0) * 2.0, 2.0) == pytest.approx(
            hydrostatic_pressure(WATER, 5.0)
        )


class TestPascal:
    def test_hydraulic_press(self):
        assert hydraulic_force(100, 0.01, 0.5) == pytest.approx(5000.0)

    def test_pressure_is_preserved(self):
        f_in, a_in, a_out = 80.0, 0.004, 0.3
        assert pressure(f_in, a_in) == pytest.approx(
            pressure(hydraulic_force(f_in, a_in, a_out), a_out)
        )

    def test_equal_pistons_transmit_the_same_force(self):
        assert hydraulic_force(75.0, 0.02, 0.02) == pytest.approx(75.0)


class TestArchimedes:
    def test_buoyant_force_reference_value(self):
        assert buoyant_force(WATER, 0.002) == pytest.approx(19.6133)

    def test_no_displacement_no_buoyancy(self):
        assert buoyant_force(WATER, 0.0) == 0.0

    def test_floating_body_displaces_its_own_weight(self):
        # Ice (917 kg/m3) block of 0.5 m3 in water: buoyancy balances weight.
        volume, ice = 0.5, 917.0
        immersed = volume * submerged_fraction(ice, WATER)
        assert buoyant_force(WATER, immersed) == pytest.approx(weight(ice * volume))

    def test_submerged_fraction_of_ice(self):
        assert submerged_fraction(917.0, WATER) == pytest.approx(0.917)

    def test_floats(self):
        assert floats(917.0, WATER)
        assert not floats(7870.0, WATER)
        assert not floats(WATER, WATER)

    @pytest.mark.parametrize("object_density", [7870.0, 1000.0])
    def test_non_floating_body_has_no_submerged_fraction(self, object_density):
        with pytest.raises(PhysicalInconsistencyError):
            submerged_fraction(object_density, WATER)


@pytest.mark.parametrize(
    "call",
    [
        lambda: density(0, 1),
        lambda: density(1, -1),
        lambda: pressure(-1, 1),
        lambda: pressure(1, 0),
        lambda: hydrostatic_pressure(0, 1),
        lambda: hydrostatic_pressure(WATER, -1),
        lambda: hydrostatic_pressure(WATER, 1, gravity=0),
        lambda: hydrostatic_pressure(WATER, 1, surface_pressure=-1),
        lambda: hydraulic_force(-1, 1, 1),
        lambda: hydraulic_force(1, 0, 1),
        lambda: buoyant_force(WATER, -0.1),
        lambda: buoyant_force(float("nan"), 0.1),
        lambda: floats(0, WATER),
    ],
)
def test_invalid_inputs(call):
    with pytest.raises(InvalidInputError):
        call()
