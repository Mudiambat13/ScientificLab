import math

import numpy as np
import pytest

from scientificlab.mathematics.differential_equations import (
    solve_ode_numeric,
    solve_second_order_numeric,
)
from scientificlab.utils.exceptions import ConvergenceError, InvalidInputError


def test_exponential_growth_reaches_e():
    result = solve_ode_numeric(lambda t, y: y, (0.0, 1.0), [1.0])
    assert result.y[0, -1] == pytest.approx(math.e, rel=1e-7)
    assert result.t[0] == 0.0
    assert result.t[-1] == pytest.approx(1.0)
    assert result.function_evaluations > 0


def test_t_eval_controls_output_times():
    times = np.linspace(0.0, 2.0, 11)
    result = solve_ode_numeric(lambda t, y: -y, (0.0, 2.0), [1.0], t_eval=times)
    assert result.t == pytest.approx(times)
    assert result.y.shape == (1, 11)
    assert result.y[0] == pytest.approx(np.exp(-times), rel=1e-6)


def test_logistic_equation_against_closed_form():
    r, capacity, y0 = 0.8, 10.0, 1.0
    times = np.linspace(0.0, 10.0, 21)
    result = solve_ode_numeric(
        lambda t, y: r * y * (1 - y / capacity), (0.0, 10.0), [y0], t_eval=times
    )
    exact = capacity / (1 + (capacity / y0 - 1) * np.exp(-r * times))
    assert result.y[0] == pytest.approx(exact, rel=1e-6)


def test_system_of_equations_rotation_conserves_radius():
    result = solve_ode_numeric(lambda t, y: [-y[1], y[0]], (0.0, 20.0), [1.0, 0.0])
    radius = np.hypot(result.y[0], result.y[1])
    assert radius == pytest.approx(np.ones_like(radius), abs=1e-6)


def test_second_order_oscillator_matches_sine():
    times = np.linspace(0.0, 2 * math.pi, 50)
    result = solve_second_order_numeric(
        lambda t, y, v: -y, (0.0, 2 * math.pi), 0.0, 1.0, t_eval=times
    )
    assert result.y[0] == pytest.approx(np.sin(times), abs=1e-6)
    assert result.y[1] == pytest.approx(np.cos(times), abs=1e-6)
    energy = result.y[0] ** 2 + result.y[1] ** 2
    assert energy == pytest.approx(np.ones_like(energy), abs=1e-6)


def test_damped_oscillator_decay_envelope():
    damping, omega0 = 0.2, 3.0
    result = solve_second_order_numeric(
        lambda t, y, v: -2 * damping * omega0 * v - omega0**2 * y,
        (0.0, 5.0),
        1.0,
        0.0,
    )
    assert abs(result.y[0, -1]) < math.exp(-damping * omega0 * 5.0) * 1.1


def test_stiff_problem_with_implicit_method():
    result = solve_ode_numeric(
        lambda t, y: -50.0 * y, (0.0, 1.0), [1.0], method="Radau"
    )
    assert abs(result.y[0, -1] - math.exp(-50.0)) < 1e-9
    assert result.method == "Radau"


def test_backward_integration():
    result = solve_ode_numeric(lambda t, y: y, (1.0, 0.0), [math.e])
    assert result.y[0, -1] == pytest.approx(1.0, rel=1e-6)


def test_finite_time_blow_up_is_reported():
    # y' = y**2, y(0) = 1 has the exact solution 1/(1 - t): it blows up at t = 1.
    with pytest.raises(ConvergenceError):
        solve_ode_numeric(lambda t, y: y**2, (0.0, 2.0), [1.0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "Euler"},
        {"rtol": 0.0},
        {"rtol": -1e-6},
        {"atol": -1.0},
    ],
)
def test_invalid_options(kwargs):
    with pytest.raises(InvalidInputError):
        solve_ode_numeric(lambda t, y: y, (0.0, 1.0), [1.0], **kwargs)


@pytest.mark.parametrize(
    ("t_span", "y0"),
    [
        ((0.0, 0.0), [1.0]),
        ((0.0, float("nan")), [1.0]),
        ((0.0, 1.0), [float("nan")]),
        ((0.0, 1.0), []),
    ],
)
def test_invalid_problem_definition(t_span, y0):
    with pytest.raises(InvalidInputError):
        solve_ode_numeric(lambda t, y: y, t_span, y0)


def test_non_callable_right_hand_side():
    with pytest.raises(InvalidInputError):
        solve_ode_numeric("y", (0.0, 1.0), [1.0])  # type: ignore[arg-type]


def test_t_span_must_be_a_pair():
    with pytest.raises(InvalidInputError):
        solve_ode_numeric(lambda t, y: y, (0.0, 1.0, 2.0), [1.0])  # type: ignore[arg-type]


def test_non_callable_acceleration():
    with pytest.raises(InvalidInputError):
        solve_second_order_numeric(5, (0.0, 1.0), 0.0, 0.0)  # type: ignore[arg-type]
