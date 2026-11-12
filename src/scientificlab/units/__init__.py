"""Unit management built on Pint: conversion, validation, unit-aware physics.

* :data:`Q_` / :data:`ureg`: the shared quantity constructor and registry.
* :func:`convert`: checked unit conversion (offset temperatures supported).
* :mod:`scientificlab.units.physics`: unit-aware wrappers of physics functions.
"""

from scientificlab.units.conversion import (
    check_dimension,
    convert,
    convert_quantity,
    make_quantity,
    parse_unit,
)
from scientificlab.units.registry import Q_, ureg

__all__ = [
    "Q_",
    "check_dimension",
    "convert",
    "convert_quantity",
    "make_quantity",
    "parse_unit",
    "ureg",
]
