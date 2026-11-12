"""Request handlers of the local web interface (no HTTP code in this module).

Every handler takes a plain ``dict`` (decoded JSON) and returns a JSON-safe
``dict``. Library errors (:class:`ScientificLabError`) become
``{"error": message, "type": ClassName}``; they are reported to the user, never
swallowed. This keeps the logic testable without a server or a browser.
"""

import base64
import dataclasses
import io
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import sympy as sp
from matplotlib.figure import Figure

from scientificlab.constants import list_constants
from scientificlab.simulations import (
    SimulationResult,
    simulate_cooling,
    simulate_free_fall,
    simulate_harmonic_oscillator,
    simulate_projectile,
    simulate_rc_circuit,
    simulate_string_wave,
)
from scientificlab.solvers import EquationSolver
from scientificlab.units import convert
from scientificlab.utils.exceptions import InvalidInputError, ScientificLabError
from scientificlab.visualization import (
    plot_functions,
    plot_simulation,
    plot_trajectory,
    plot_wave_snapshots,
)

MAX_TEXT = 500
MAX_LINES = 10
_FORMATS = ("svg", "png", "pdf")


@dataclass(frozen=True)
class Param:
    """Description of one input of a simulation form."""

    name: str
    label: str
    default: float | str
    minimum: float = 0.0
    maximum: float = 1.0
    step: float = 0.1
    unit: str = ""
    options: tuple[tuple[str, str], ...] = ()  # (value, label) for selects


# --------------------------------------------------------------------- helpers
def _json_safe(value: Any) -> Any:
    """Recursively convert NumPy/SymPy values to JSON-safe ones (NaN/inf -> None)."""
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, bool | str) or value is None:
        return value
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    return str(value)


def _check_text(text: object, name: str) -> str:
    """Basic input hygiene for strings that SymPy will parse.

    Defense in depth only: the real protection is that the server listens on the
    loopback interface and requires a per-session token (see ``server``).
    """
    if not isinstance(text, str) or not text.strip():
        raise InvalidInputError(f"{name} must be a non-empty text.")
    if len(text) > MAX_TEXT:
        raise InvalidInputError(f"{name} is too long (max {MAX_TEXT} characters).")
    lowered = text.lower()
    if "__" in text or any(w in lowered for w in ("import", "exec", "eval", "open(")):
        raise InvalidInputError(f"{name} contains a forbidden construct.")
    return text


def _number(params: Mapping[str, Any], name: str) -> float:
    value = params.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvalidInputError(f"{name} must be a number.")
    return float(value)


def _render(fig: Figure, fmt: str) -> dict[str, str]:
    if fmt not in _FORMATS:
        raise InvalidInputError(f"format must be one of {_FORMATS}.")
    buffer = io.BytesIO()
    fig.savefig(buffer, format=fmt, dpi=150)
    raw = buffer.getvalue()
    if fmt == "svg":
        return {"format": fmt, "data": raw.decode("utf-8")}
    return {"format": fmt, "data": base64.b64encode(raw).decode("ascii")}


def _guard(
    handler: Callable[[Mapping[str, Any]], dict[str, Any]],
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    def wrapper(params: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(params, Mapping):
            return {"error": "The request body must be a JSON object.", "type": "Bad"}
        try:
            return _json_safe(handler(params))  # type: ignore[no-any-return]
        except ScientificLabError as error:
            return {"error": str(error), "type": type(error).__name__}

    wrapper.__name__ = handler.__name__
    wrapper.__doc__ = handler.__doc__
    return wrapper


# ----------------------------------------------------------------- simulations
_G = 9.80665

SIMULATION_FORMS: dict[str, tuple[str, tuple[Param, ...]]] = {
    "projectile": (
        "Projectile",
        (
            Param("speed", "Vitesse initiale", 20, 1, 200, 1, "m/s"),
            Param("angle", "Angle de tir", 45, 0, 90, 1, "°"),
            Param("height", "Hauteur de départ", 0, 0, 100, 1, "m"),
            Param("gravity", "Gravité", _G, 0.5, 30, 0.01, "m/s²"),
            Param("drag", "Traînée quadratique c", 0, 0, 0.1, 0.001, "1/m"),
        ),
    ),
    "free_fall": (
        "Chute libre",
        (
            Param("height", "Hauteur", 100, 1, 5000, 1, "m"),
            Param("gravity", "Gravité", _G, 0.5, 30, 0.01, "m/s²"),
            Param("drag", "Traînée quadratique c", 0, 0, 0.1, 0.001, "1/m"),
        ),
    ),
    "oscillator": (
        "Oscillateur harmonique",
        (
            Param("mass", "Masse", 1, 0.1, 20, 0.1, "kg"),
            Param("stiffness", "Raideur du ressort", 100, 1, 1000, 1, "N/m"),
            Param("damping", "Amortissement c", 2, 0, 60, 0.1, "kg/s"),
            Param("x0", "Position initiale", 1, -2, 2, 0.1, "m"),
            Param("v0", "Vitesse initiale", 0, -10, 10, 0.1, "m/s"),
        ),
    ),
    "rc": (
        "Circuit RC",
        (
            Param("source_voltage", "Tension de la source", 5, 0, 24, 0.1, "V"),
            Param("resistance", "Résistance", 1000, 10, 100000, 10, "Ω"),
            Param("capacitance_mf", "Capacité", 1, 0.01, 10, 0.01, "mF"),
            Param(
                "initial_voltage",
                "Tension initiale du condensateur",
                0,
                0,
                24,
                0.1,
                "V",
            ),
        ),
    ),
    "cooling": (
        "Refroidissement (loi de Newton)",
        (
            Param("t0", "Température initiale", 90, -50, 300, 1, "°C"),
            Param("ambient", "Température ambiante", 20, -50, 100, 1, "°C"),
            Param(
                "k_per_min",
                "Constante de refroidissement k",
                0.05,
                0.001,
                1,
                0.001,
                "1/min",
            ),
            Param("minutes", "Durée simulée", 60, 1, 600, 1, "min"),
        ),
    ),
    "wave": (
        "Onde sur une corde",
        (
            Param("length", "Longueur de la corde", 1, 0.1, 10, 0.1, "m"),
            Param("speed", "Vitesse de l'onde", 100, 1, 1000, 1, "m/s"),
            Param(
                "shape",
                "Forme initiale",
                "pluck",
                options=(
                    ("pluck", "Pincement gaussien"),
                    ("mode", "Mode propre"),
                ),
            ),
            Param(
                "position",
                "Position du pincement (fraction)",
                0.3,
                0.05,
                0.95,
                0.01,
                "",
            ),
            Param("mode", "Numéro du mode", 3, 1, 10, 1, ""),
            Param("duration_ms", "Durée simulée", 20, 1, 200, 1, "ms"),
        ),
    ),
}


def _run_simulation(
    kind: str, p: Mapping[str, Any]
) -> tuple[list[tuple[str, Figure]], dict[str, Any], tuple[str, ...], str]:
    """Run ``kind`` and return (titled figures, summary, notes, method)."""

    def num(name: str) -> float:
        return _number(p, name)

    result: SimulationResult
    if kind == "projectile":
        result = simulate_projectile(
            num("speed"),
            num("angle"),
            height=num("height"),
            gravity=num("gravity"),
            drag=num("drag"),
        )
        figures = [
            ("Trajectoire", plot_trajectory(result)),
            ("Vitesses", plot_simulation(result, ["vx (m/s)", "vy (m/s)"])),
        ]
    elif kind == "free_fall":
        result = simulate_free_fall(
            num("height"), gravity=num("gravity"), drag=num("drag")
        )
        figures = [("Chute", plot_simulation(result))]
    elif kind == "oscillator":
        result = simulate_harmonic_oscillator(
            num("mass"),
            num("stiffness"),
            damping=num("damping"),
            initial_position=num("x0"),
            initial_velocity=num("v0"),
        )
        figures = [("Mouvement", plot_simulation(result))]
    elif kind == "rc":
        result = simulate_rc_circuit(
            num("source_voltage"),
            num("resistance"),
            num("capacitance_mf") * 1e-3,
            initial_voltage=num("initial_voltage"),
        )
        figures = [
            ("Circuit", plot_simulation(result, ["voltage (V)", "current (A)"])),
            (
                "Énergie",
                plot_simulation(
                    result,
                    [
                        "stored energy (J)",
                        "dissipated energy (J)",
                        "supplied energy (J)",
                    ],
                ),
            ),
        ]
    elif kind == "cooling":
        t0 = float(convert(num("t0"), "degC", "K").magnitude)
        ta = float(convert(num("ambient"), "degC", "K").magnitude)
        k = num("k_per_min") / 60.0
        kelvin = simulate_cooling(t0, ta, k, duration=num("minutes") * 60.0)
        celsius = kelvin.series["temperature (K)"] - 273.15
        summary = {
            key: val
            for key, val in kelvin.summary.items()
            if key != "final_temperature"
        }
        summary["final_temperature_c"] = float(celsius[-1])
        result = dataclasses.replace(
            kelvin, series={"temperature (°C)": celsius}, summary=summary
        )
        figures = [("Température", plot_simulation(result))]
    elif kind == "wave":
        length = num("length")
        speed = num("speed")
        if p.get("shape") == "mode":
            n = int(num("mode"))

            def shape(x: np.ndarray) -> np.ndarray:
                return np.sin(n * np.pi * x / length)
        else:
            centre, width = num("position") * length, 0.05 * length

            def shape(x: np.ndarray) -> np.ndarray:
                # Subtract the (tiny) end values so the data vanishes at both ends.
                g = np.exp(-(((x - centre) / width) ** 2))
                return np.asarray(g - g[0] * (1 - x / length) - g[-1] * (x / length))

        wave = simulate_string_wave(
            length, speed, shape, duration=num("duration_ms") * 1e-3, nx=201
        )
        drift = float((wave.energy.max() - wave.energy.min()) / wave.energy.mean())
        summary = {**wave.summary, "energy_relative_drift": drift}
        return (
            [("Profils", plot_wave_snapshots(wave))],
            summary,
            wave.notes,
            wave.method,
        )
    else:
        raise InvalidInputError(f"Unknown simulation {kind!r}.")
    return figures, dict(result.summary), result.notes, result.method


@_guard
def simulate(params: Mapping[str, Any]) -> dict[str, Any]:
    """Run a simulation and return titled figures, summary, notes and method."""
    kind = params.get("kind")
    if kind not in SIMULATION_FORMS:
        raise InvalidInputError(f"Unknown simulation {kind!r}.")
    values = params.get("values", {})
    if not isinstance(values, Mapping):
        raise InvalidInputError("values must be an object.")
    fmt = str(params.get("format", "svg"))
    figures, summary, notes, method = _run_simulation(str(kind), values)
    return {
        "figures": [{"title": t, **_render(f, fmt)} for t, f in figures],
        "summary": summary,
        "notes": list(notes),
        "method": method,
    }


# ------------------------------------------------------------------ the others
@_guard
def schema(_params: Mapping[str, Any]) -> dict[str, Any]:
    """Describe the simulation forms so the page can build them."""
    return {
        "simulations": {
            key: {"title": title, "params": [dataclasses.asdict(p) for p in plist]}
            for key, (title, plist) in SIMULATION_FORMS.items()
        }
    }


@_guard
def constants(_params: Mapping[str, Any]) -> dict[str, Any]:
    """List all physical constants with unit, uncertainty and source."""
    return {
        "constants": [
            {
                "key": c.key,
                "name": c.name,
                "symbol": c.symbol,
                "value": c.value,
                "unit": c.unit,
                "uncertainty": c.uncertainty,
                "exact": c.exact,
                "edition": c.edition,
            }
            for c in list_constants()
        ]
    }


@_guard
def convert_units(params: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a value between units (checked dimensions)."""
    result = convert(
        _number(params, "value"),
        _check_text(params.get("from"), "from"),
        _check_text(params.get("to"), "to"),
    )
    return {"value": float(result.magnitude), "text": str(result)}


def _solution_view(s: sp.Expr) -> dict[str, Any]:
    try:
        number = complex(sp.N(s))
        numeric: Any = number.real if number.imag == 0 else str(number)
    except TypeError:
        numeric = None
    return {"exact": str(s), "numeric": numeric}


@_guard
def solve(params: Mapping[str, Any]) -> dict[str, Any]:
    """Solve one equation (one line) or a system (several lines)."""
    raw = params.get("equations")
    if not isinstance(raw, Sequence) or isinstance(raw, str) or not raw:
        raise InvalidInputError("equations must be a non-empty list of texts.")
    if len(raw) > MAX_LINES:
        raise InvalidInputError(f"At most {MAX_LINES} equations.")
    equations = [_check_text(e, "equation") for e in raw]
    values = params.get("values") or {}
    if not isinstance(values, Mapping):
        raise InvalidInputError("values must be an object.")
    solver = EquationSolver(domain=str(params.get("domain", "real")))
    variable = params.get("variable") or None
    guess = params.get("initial_guess")
    if len(equations) == 1:
        single = solver.solve(
            equations[0],
            None if variable is None else str(variable),
            values=values,
            initial_guess=None if guess in (None, "") else float(guess),
        )
        return {
            "kind": "equation",
            "variable": single.variable,
            "solutions": [_solution_view(s) for s in single.solutions],
            "method": single.method,
            "exact": single.exact,
            "complete": single.complete,
            "verified": single.verified,
            "warnings": list(single.warnings),
        }
    system = solver.solve_system(equations, values=values)
    return {
        "kind": "system",
        "variables": list(system.variables),
        "solutions": [
            {name: _solution_view(v) for name, v in sol.items()}
            for sol in system.solutions
        ],
        "method": system.method,
        "exact": system.exact,
        "complete": system.complete,
        "verified": system.verified,
        "warnings": list(system.warnings),
    }


@_guard
def plot_expressions(params: Mapping[str, Any]) -> dict[str, Any]:
    """Plot one or more expressions in ``x`` over a range."""
    raw = params.get("expressions")
    if not isinstance(raw, Sequence) or isinstance(raw, str) or not raw:
        raise InvalidInputError("expressions must be a non-empty list of texts.")
    if len(raw) > MAX_LINES:
        raise InvalidInputError(f"At most {MAX_LINES} expressions.")
    expressions = [_check_text(e, "expression") for e in raw]
    limits = None
    if params.get("ymin") not in (None, "") and params.get("ymax") not in (None, ""):
        limits = (_number(params, "ymin"), _number(params, "ymax"))
    fig = plot_functions(
        expressions, (_number(params, "xmin"), _number(params, "xmax")), y_limits=limits
    )
    return _render(fig, str(params.get("format", "svg")))


HANDLERS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "schema": schema,
    "constants": constants,
    "convert": convert_units,
    "solve": solve,
    "plot": plot_expressions,
    "simulate": simulate,
}
