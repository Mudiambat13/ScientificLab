"""Scientific plots (Matplotlib object-oriented API, PNG/SVG/PDF export)."""

from scientificlab.visualization._common import save_figure
from scientificlab.visualization.functions import (
    plot_data,
    plot_function,
    plot_functions,
)
from scientificlab.visualization.physics_plots import (
    plot_isotherms,
    plot_position_vs_time,
    plot_rc_charging,
)
from scientificlab.visualization.simulation_plots import (
    plot_simulation,
    plot_trajectory,
    plot_wave_snapshots,
)
from scientificlab.visualization.statistics_plots import plot_histogram, plot_regression

__all__ = [
    "plot_data",
    "plot_function",
    "plot_functions",
    "plot_histogram",
    "plot_isotherms",
    "plot_position_vs_time",
    "plot_rc_charging",
    "plot_regression",
    "plot_simulation",
    "plot_trajectory",
    "plot_wave_snapshots",
    "save_figure",
]
