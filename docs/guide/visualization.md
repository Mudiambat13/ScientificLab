# Plots

Figures are built with Matplotlib's object-oriented API (no global `pyplot`
state). Every function returns a `Figure` and can save it to **PNG, SVG or PDF**
according to the file extension.

```python
from scientificlab.visualization import plot_functions

figure = plot_functions(
    ["sin(x)/x", "1/x"], (-10, 10), y_limits=(-2, 2), title="Two functions"
)
print(type(figure).__name__)  # Figure
```

Points where a function is undefined, not real, or crosses a pole are left as
**gaps** and never joined by a line. Use `plot_data` for measured points with
error bars, `plot_simulation` / `plot_trajectory` / `plot_wave_snapshots` for
simulation results, and `plot_histogram` and `plot_regression` (data with its least-squares line),
`plot_rc_charging`, `plot_position_vs_time`,
`plot_isotherms` for ready-made physics plots.
