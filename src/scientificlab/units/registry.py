"""The single shared Pint unit registry.

All quantities created by ScientificLab use this registry. Mixing quantities
from different registries raises an error in Pint, so always import ``ureg``
and ``Q_`` from here (or from :mod:`scientificlab.units`).
"""

import pint

ureg: pint.UnitRegistry = pint.UnitRegistry()
"""The shared registry (default Pint unit definitions)."""

ureg.formatter.default_format = "~P"  # short, pretty: "27.8 m/s"

Q_ = ureg.Quantity
"""Quantity constructor: ``Q_(5, "m/s")`` or ``Q_("5 m/s")``."""
