# Constants and units

## Constants with provenance

```python
from scientificlab.constants import get_constant, SPEED_OF_LIGHT

G = get_constant("G")
print(G.value, G.unit, G.uncertainty, G.exact, G.edition)
# 6.6743e-11 m ** 3 / (kg * s ** 2) 1.5e-15 False CODATA 2022
print(SPEED_OF_LIGHT)  # 299792458.0
print(get_constant("speed of light").quantity)  # 299792458.0 m/s
```

Eighteen constants are provided; exact ones (SI 2019, CGPM) have uncertainty 0.
The tests compare every value with SciPy's independent CODATA table.

## Units (Pint)

```python
from scientificlab.units import convert, make_quantity

print(convert(25, "degC", "K"))  # 298.15 K
print(convert(1, "atm", "Pa"))  # 101325.0 Pa
```

- Dimensions are checked: `convert(1, "m", "s")` raises `IncompatibleUnitsError`.
- `degC` and `degF` are offset scales and convert correctly; `make_quantity(-300, "degC")`
  raises `PhysicalInconsistencyError` (below absolute zero).
- Unit-aware physics: `scientificlab.units.physics` (`kinetic_energy`,
  `heat_energy`, ...). An absolute `degC` is refused where a temperature
  *difference* is expected.
- Offset-scale conversions go through floating point: compare with a tolerance
  (`32 degF -> degC` gives about 5.7e-14, not exactly 0).
