"""Physical constants.

Phase 1 only provides the constant needed by the mechanics module. The full
constants system (name, symbol, unit, source, uncertainty, CODATA edition) is
planned for Phase 4.
"""

STANDARD_GRAVITY: float = 9.80665
"""Standard acceleration of gravity ``g_n`` in m/s**2.

Exact by definition (3rd CGPM, 1901); it is not a measured quantity.
"""

__all__ = ["STANDARD_GRAVITY"]
