# Graphical interface

```bash
scientificlab-gui            # opens your browser on a local page
scientificlab-gui --port 8000 --no-browser
python -m scientificlab.gui  # equivalent
```

The page has four tabs: **Simulations** (live sliders, plots exportable to PNG,
SVG and PDF), **Équations** (equation or system, with the exact / complete /
verified flags), **Fonctions** (plots of expressions in `x`) and **Unités et
constantes**. The interface language is French.

It uses only the Python standard library, so it needs no extra dependency and no
Tkinter.

!!! warning "Local use only"
    The solver parses text with SymPy (`eval`). The server therefore listens on
    `127.0.0.1` only, requires a random per-session token, and checks the `Host`
    and `Origin` headers. Do not expose it through a reverse proxy or on another
    network interface.
