import pytest

from scientificlab.physics.fluid_mechanics import (
    bernoulli_pressure,
    continuity_velocity,
    hydrostatic_pressure,
    mass_flow_rate,
    volumetric_flow_rate,
)
from scientificlab.physics.mechanics import free_fall_impact_speed
from scientificlab.utils.exceptions import (
    InvalidInputError,
    PhysicalInconsistencyError,
)

WATER = 1000.0
ATMOSPHERE = 101_325.0


def test_flow_rates():
    assert volumetric_flow_rate(0.1, 2.0) == pytest.approx(0.2)
    assert mass_flow_rate(WATER, 0.1, 2.0) == pytest.approx(200.0)
    assert volumetric_flow_rate(0.1, 0.0) == 0.0


def test_continuity_narrowing_speeds_the_flow_up():
    assert continuity_velocity(0.1, 2.0, 0.05) == pytest.approx(4.0)
    assert continuity_velocity(0.1, 2.0, 0.2) == pytest.approx(1.0)


def test_continuity_conserves_the_flow_rate():
    v2 = continuity_velocity(0.08, 1.7, 0.013)
    assert volumetric_flow_rate(0.013, v2) == pytest.approx(
        volumetric_flow_rate(0.08, 1.7)
    )


class TestBernoulli:
    def test_horizontal_pipe_reference_value(self):
        p2 = bernoulli_pressure(WATER, 200_000, 2.0, 0.0, 4.0, 0.0)
        assert p2 == pytest.approx(194_000.0)

    def test_pressure_drops_where_the_flow_speeds_up(self):
        assert bernoulli_pressure(WATER, 150_000, 1.0, 0.0, 5.0, 0.0) < 150_000

    def test_venturi_with_continuity(self):
        v2 = continuity_velocity(0.1, 2.0, 0.05)
        assert bernoulli_pressure(WATER, 200_000, 2.0, 0.0, v2, 0.0) == pytest.approx(
            194_000.0
        )

    def test_fluid_at_rest_reduces_to_hydrostatic_pressure(self):
        # Point 1 is 3 m above point 2 in still water, open to the atmosphere.
        p2 = bernoulli_pressure(WATER, ATMOSPHERE, 0.0, 3.0, 0.0, 0.0)
        assert p2 == pytest.approx(
            hydrostatic_pressure(WATER, 3.0, surface_pressure=ATMOSPHERE)
        )

    def test_torricelli_efflux_matches_free_fall_speed(self):
        # Open tank, hole 2.5 m below the surface: both ends at atmospheric
        # pressure, so v = sqrt(2 g h), the speed of a body falling h.
        height = 2.5
        speed = free_fall_impact_speed(height)
        p2 = bernoulli_pressure(WATER, ATMOSPHERE, 0.0, height, speed, 0.0)
        assert p2 == pytest.approx(ATMOSPHERE)

    def test_climbing_costs_pressure(self):
        low = bernoulli_pressure(WATER, 300_000, 2.0, 0.0, 2.0, 10.0)
        assert low == pytest.approx(300_000 - WATER * 9.80665 * 10.0)

    def test_the_impossible_flow_that_would_need_negative_pressure(self):
        with pytest.raises(PhysicalInconsistencyError, match="cavitate"):
            bernoulli_pressure(WATER, 1_000, 0.0, 0.0, 20.0, 0.0)

    def test_total_head_is_conserved(self):
        rho, g = WATER, 9.80665
        args = (rho, 250_000.0, 1.5, 4.0, 3.2, 1.0)
        p2 = bernoulli_pressure(*args)
        head_1 = 250_000.0 + 0.5 * rho * 1.5**2 + rho * g * 4.0
        head_2 = p2 + 0.5 * rho * 3.2**2 + rho * g * 1.0
        assert head_1 == pytest.approx(head_2)

    @pytest.mark.parametrize(
        "args",
        [
            (0, 2e5, 2, 0, 4, 0),
            (WATER, -1, 2, 0, 4, 0),
            (WATER, 2e5, -2, 0, 4, 0),
            (WATER, 2e5, 2, 0, -4, 0),
            (WATER, 2e5, 2, float("nan"), 4, 0),
            (WATER, 2e5, 2, 0, 4, float("inf")),
        ],
    )
    def test_invalid_inputs(self, args):
        with pytest.raises(InvalidInputError):
            bernoulli_pressure(*args)


@pytest.mark.parametrize(
    "call",
    [
        lambda: volumetric_flow_rate(0, 1),
        lambda: volumetric_flow_rate(1, -1),
        lambda: mass_flow_rate(0, 1, 1),
        lambda: continuity_velocity(0.1, 2, 0),
        lambda: continuity_velocity(0.1, -2, 0.1),
    ],
)
def test_invalid_flow_inputs(call):
    with pytest.raises(InvalidInputError):
        call()
