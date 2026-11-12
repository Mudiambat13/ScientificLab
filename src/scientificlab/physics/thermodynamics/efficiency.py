"""Heat-engine efficiency, checked against the first and second laws.

Efficiencies are dimensionless fractions in [0, 1]. Heat and work are in J,
temperatures in kelvin.
"""

from scientificlab.utils.exceptions import InvalidInputError, PhysicalInconsistencyError
from scientificlab.utils.validation import (
    require_non_negative,
    require_positive,
)

_CARNOT_TOLERANCE = 1e-9


def carnot_efficiency(hot_temperature: float, cold_temperature: float) -> float:
    """Maximum efficiency of any engine between two reservoirs: ``1 - T_c/T_h``.

    Args:
        hot_temperature: Hot reservoir temperature in K (> 0).
        cold_temperature: Cold reservoir temperature in K (> 0).

    Raises:
        PhysicalInconsistencyError: If ``T_h <= T_c`` (no engine can run).
    """
    t_hot = require_positive(hot_temperature, "hot_temperature")
    t_cold = require_positive(cold_temperature, "cold_temperature")
    if t_hot <= t_cold:
        raise PhysicalInconsistencyError(
            "The hot reservoir must be hotter than the cold one "
            f"(got {t_hot} K and {t_cold} K)."
        )
    return 1.0 - t_cold / t_hot


def thermal_efficiency(
    work_output: float,
    heat_input: float,
    hot_temperature: float | None = None,
    cold_temperature: float | None = None,
) -> float:
    """Thermal efficiency of a heat engine: ``eta = W / Q_h``.

    Args:
        work_output: Net work produced per cycle, in J (>= 0).
        heat_input: Heat absorbed from the hot reservoir per cycle, in J (> 0).
        hot_temperature: Optional hot reservoir temperature in K. Give both
            temperatures to also check the second law.
        cold_temperature: Optional cold reservoir temperature in K.

    Raises:
        InvalidInputError: If only one temperature is given, or on invalid values.
        PhysicalInconsistencyError: If ``W > Q_h`` (violates energy
            conservation) or, with temperatures, if ``eta`` exceeds the Carnot
            efficiency (violates the second law).
    """
    work = require_non_negative(work_output, "work_output")
    heat = require_positive(heat_input, "heat_input")
    if (hot_temperature is None) != (cold_temperature is None):
        raise InvalidInputError(
            "Give both hot_temperature and cold_temperature, or neither."
        )
    efficiency = work / heat
    if efficiency > 1.0:
        raise PhysicalInconsistencyError(
            f"Work output ({work} J) exceeds heat input ({heat} J): "
            "energy would not be conserved."
        )
    if hot_temperature is not None and cold_temperature is not None:
        limit = carnot_efficiency(hot_temperature, cold_temperature)
        if efficiency > limit * (1.0 + _CARNOT_TOLERANCE):
            raise PhysicalInconsistencyError(
                f"Efficiency {efficiency:.4g} exceeds the Carnot limit "
                f"{limit:.4g}: the second law forbids it."
            )
    return efficiency


def efficiency_from_heats(heat_input: float, heat_rejected: float) -> float:
    """Efficiency from the heats exchanged: ``eta = 1 - Q_c / Q_h``.

    Only energy conservation is checked here. Whether the efficiency is allowed
    by the second law depends on the reservoir temperatures: use
    :func:`thermal_efficiency` with both temperatures for that check.

    Args:
        heat_input: Heat absorbed from the hot reservoir, in J (> 0).
        heat_rejected: Heat released to the cold reservoir, in J (>= 0,
            magnitude).

    Raises:
        PhysicalInconsistencyError: If ``Q_c > Q_h`` (negative work output).
    """
    q_hot = require_positive(heat_input, "heat_input")
    q_cold = require_non_negative(heat_rejected, "heat_rejected")
    if q_cold > q_hot:
        raise PhysicalInconsistencyError(
            "Rejected heat exceeds absorbed heat: the engine would consume work."
        )
    return 1.0 - q_cold / q_hot
