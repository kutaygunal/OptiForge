"""Automatic selection of a starting-point lens form from the user's specs.

A small, transparent heuristic maps (field of view, f-number, focal length,
optional explicit target) to the classical form that is most likely to be a
good starting point for that operating point.  The engineer can always override
it with an explicit lens_type.
"""
from __future__ import annotations

from .catalog import CATALOG

# explicit lens_type (UI labels) -> prototype id
_ALIAS = {
    "auto": None,
    "double_gauss": "double_gauss",
    "camera": "double_gauss",
    "cooke_triplet": "cooke_triplet",
    "cooke": "cooke_triplet",
    "telephoto": "telephoto",
    "retrofocus": "retrofocus",
    "wide": "retrofocus",
    "wide_angle": "retrofocus",
    "petzval": "petzval",
    "projector": "petzval",
    "microscope": "microscope",
    "microscope_objective": "microscope",
    "collimator": "collimator",
}


def resolve_id(lens_type: str) -> str:
    key = lens_type.strip().lower().replace(" ", "_")
    if key in _ALIAS:
        return _ALIAS[key]
    if key in CATALOG:
        return key
    raise KeyError(f"unknown lens type '{lens_type}'")


def auto_select(half_field_deg: float, fno: float, efl: float) -> str:
    """Pick a prototype id from the operating point.

    half_field_deg is the half angle in degrees, fno the target f-number.
    """
    if half_field_deg >= 55:
        return "retrofocus"          # very wide angle needs negative front group
    if half_field_deg >= 30:
        # medium-wide: double Gauss when reasonably fast, triplet when slow/simple
        return "double_gauss" if fno <= 4.0 else "cooke_triplet"
    if half_field_deg >= 12:
        # normal to medium-narrow
        if efl is not None and efl >= 90 and fno <= 6.0:
            return "telephoto"
        return "double_gauss"
    # narrow field
    if fno is not None and fno <= 2.5:
        return "collimator"
    if fno is not None and fno <= 4.0:
        return "petzval"
    return "collimator"


def select(lens_type: str, half_field_deg: float, fno: float,
           efl: float, finite: bool = False) -> str:
    """Return the prototype id for the request (honoring explicit type)."""
    if finite or lens_type.lower() in ("microscope", "microscope_objective"):
        return "microscope"
    resolved = resolve_id(lens_type) if lens_type and lens_type.lower() != "auto" else None
    if resolved is not None:
        return resolved
    return auto_select(half_field_deg, fno, efl)
