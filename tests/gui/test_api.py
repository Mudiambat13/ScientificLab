import json

import pytest

from scientificlab.gui import api


def defaults(kind):
    return {p.name: p.default for p in api.SIMULATION_FORMS[kind][1]}


def test_schema_describes_every_simulation():
    schema = api.schema({})["simulations"]
    assert set(schema) == set(api.SIMULATION_FORMS)
    assert all(s["params"] for s in schema.values())


@pytest.mark.parametrize("kind", list(api.SIMULATION_FORMS))
def test_every_simulation_runs_with_its_defaults(kind):
    result = api.simulate({"kind": kind, "values": defaults(kind)})
    assert "error" not in result, result
    assert result["figures"] and result["method"] and result["notes"]
    assert result["figures"][0]["data"].lstrip().startswith(("<?xml", "<svg"))
    json.dumps(result, allow_nan=False)  # fully JSON-safe


def test_projectile_summary_matches_library():
    values = {**defaults("projectile"), "speed": 20, "angle": 45}
    result = api.simulate({"kind": "projectile", "values": values})
    assert result["summary"]["range"] == pytest.approx(400 / 9.80665, rel=1e-12)


def test_cooling_is_reported_in_celsius():
    values = {"t0": 90, "ambient": 20, "k_per_min": 0.05, "minutes": 600}
    result = api.simulate({"kind": "cooling", "values": values})
    assert result["summary"]["final_temperature_c"] == pytest.approx(20.0, abs=0.01)
    assert "final_temperature" not in result["summary"]


def test_below_absolute_zero_is_reported_not_crashed():
    values = {"t0": -300, "ambient": 20, "k_per_min": 0.05, "minutes": 60}
    result = api.simulate({"kind": "cooling", "values": values})
    assert result["type"] == "PhysicalInconsistencyError"


def test_wave_shapes():
    values = {**defaults("wave"), "shape": "mode", "mode": 2}
    result = api.simulate({"kind": "wave", "values": values})
    assert result["summary"]["energy_relative_drift"] < 1e-9


@pytest.mark.parametrize("fmt", ["png", "pdf", "svg"])
def test_export_formats(fmt):
    result = api.simulate({"kind": "rc", "values": defaults("rc"), "format": fmt})
    assert result["figures"][0]["format"] == fmt


def test_simulation_errors_are_messages():
    bad = {**defaults("projectile"), "angle": 120}
    result = api.simulate({"kind": "projectile", "values": bad})
    assert result["type"] == "InvalidInputError"
    assert "angle" in result["error"]
    assert "error" in api.simulate({"kind": "nope"})
    assert "error" in api.simulate({"kind": "rc", "values": {"source_voltage": "x"}})
    assert "error" in api.simulate({"kind": "rc", "values": 3})
    assert "error" in api.simulate(
        {"kind": "rc", "values": defaults("rc"), "format": "gif"}
    )


def test_convert():
    r = api.convert_units({"value": 100, "from": "km/h", "to": "m/s"})
    assert r["value"] == pytest.approx(100 / 3.6)
    assert api.convert_units({"value": 1, "from": "m", "to": "s"})["type"] == (
        "IncompatibleUnitsError"
    )
    assert api.convert_units({"value": 1, "from": "foo", "to": "m"})["type"] == (
        "UnknownUnitError"
    )
    assert "error" in api.convert_units({"value": "1", "from": "m", "to": "km"})


def test_constants_listing():
    listing = api.constants({})["constants"]
    c = next(c for c in listing if c["key"] == "speed_of_light")
    assert c["value"] == 299792458.0 and c["exact"] is True


def test_solve_equation_and_flags():
    r = api.solve({"equations": ["x**2 - 4 = 0"]})
    assert [s["numeric"] for s in r["solutions"]] == [-2.0, 2.0]
    assert (r["exact"], r["complete"], r["verified"]) == (True, True, True)
    r = api.solve({"equations": ["exp(x) = 2"]})
    assert r["complete"] is False and r["warnings"]
    r = api.solve({"equations": ["cos(x) = x"], "initial_guess": 0.7})
    assert r["method"] == "numeric" and r["exact"] is False
    r = api.solve(
        {
            "equations": ["v = v0 + a*t"],
            "variable": "t",
            "values": {"v": 20, "v0": 0, "a": 2},
        }
    )
    assert r["solutions"][0]["numeric"] == 10.0


def test_solve_system_and_errors():
    r = api.solve({"equations": ["x + y = 3", "x - y = 1"]})
    assert r["kind"] == "system"
    assert r["solutions"][0]["x"]["numeric"] == 2.0
    assert api.solve({"equations": ["x**2 + 1 = 0"]})["type"] == "NoSolutionError"
    assert "error" in api.solve({"equations": []})
    assert "error" in api.solve({"equations": "x = 1"})
    assert "error" in api.solve({"equations": ["x = 1"] * 11})
    assert "error" in api.solve({"equations": ["x = 1"], "values": [1]})


@pytest.mark.parametrize(
    "text",
    [
        "__import__('os').system('id')",
        "x.__class__",
        "import os",
        "eval('1')",
        "open('f')",
        "x" * 501,
        "",
        5,
    ],
)
def test_dangerous_or_invalid_text_is_rejected_before_parsing(text):
    assert "error" in api.solve({"equations": [text]})
    assert "error" in api.plot_expressions(
        {"expressions": [text], "xmin": 0, "xmax": 1}
    )


def test_plot_expressions():
    r = api.plot_expressions(
        {"expressions": ["sin(x)", "1/x"], "xmin": -3, "xmax": 3, "ymin": -5, "ymax": 5}
    )
    assert r["format"] == "svg" and "<svg" in r["data"]
    assert "error" in api.plot_expressions({"expressions": ["x"], "xmin": 1, "xmax": 0})
    assert "error" in api.plot_expressions(
        {"expressions": ["x"], "xmin": "a", "xmax": 1}
    )


def test_non_object_body():
    assert api.solve([1, 2])["type"] == "Bad"  # type: ignore[arg-type]
