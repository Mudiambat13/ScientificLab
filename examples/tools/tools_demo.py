"""Constants, units, equation solver and plots (Phase 4).

Run: python examples/tools/tools_demo.py   (writes PNG/SVG/PDF in this folder)
"""

from pathlib import Path

from scientificlab.constants import get_constant
from scientificlab.solvers import EquationSolver, solve_equation
from scientificlab.units import Q_, convert
from scientificlab.units import physics as up
from scientificlab.utils.exceptions import IncompatibleUnitsError
from scientificlab.visualization import (
    plot_functions,
    plot_isotherms,
    plot_rc_charging,
)

out = Path(__file__).parent

# --- Constants with provenance -------------------------------------------
G = get_constant("G")
print(G)  # ... = 6.6743e-11 ... [+/- 1.5e-15; CODATA 2022]
print(get_constant("speed of light").quantity)  # 299792458.0 m/s

# --- Units ---------------------------------------------------------------
print(convert(100, "km/h", "m/s"))  # 27.77777777777778 m/s
print(convert(25, "degC", "K"))  # 298.15 K
print(up.kinetic_energy(Q_(2, "g"), Q_(36, "km/h")))  # 0.1 J
try:
    convert(1, "m", "s")
except IncompatibleUnitsError as error:
    print("refused:", error)

# --- Equation solver -----------------------------------------------------
r = solve_equation("v = v0 + a*t", "t", values={"v": 20, "v0": 0, "a": 2})
print(r.solutions, r.exact, r.complete, r.verified)  # (10,) True True True

r = solve_equation("exp(x) = 2")
print(r.solutions, r.complete)  # (log(2),) False  <- honest: may be incomplete

r = solve_equation("cos(x) = x", initial_guess=0.7)
print(r.values, r.method, r.exact)  # (0.739085...,) numeric False

print(EquationSolver().solve_system(["x**2 + y**2 = 25", "x - y = 1"]).solutions)

# --- Plots (PNG / SVG / PDF chosen by extension) ---------------------------
plot_functions(["sin(x)/x", "cos(x)"], (-10, 10), save_path=out / "functions.png")
plot_rc_charging(5.0, 1000.0, 1e-3, save_path=out / "rc.svg")
plot_isotherms(1.0, [250, 300, 400], (0.01, 0.1), save_path=out / "isotherms.pdf")
print("figures written to", out)
