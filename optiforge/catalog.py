"""Library of normalized classical starting-point lens prescriptions.

Each entry is a well-known lens *form* with real Schott glasses, stored at a
reference focal length f0 (roughly 100 mm).  The generator scales the whole
design to the requested EFL and then locally optimizes the shape while holding
the EFL fixed, so these are genuinely *sensible* starting points rather than
arbitrary glass stacks.

Stored geometry is validated by optiforge/seqparse against the .seq
dialect.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .design import Design


@dataclass
class Prototype:
    id: str
    name: str
    description: str
    f0: float                    # reference focal length
    radius: List[float]          # surface radii (idx 0 = object, last = image)
    thick: List[float]           # gaps after each surface (len = len(radius)-1)
    glass: List[str]             # media after each surface ('' = air)
    stop: int
    finite: bool = False
    obj_dist: float = float("inf")
    obj_height: float = 0.0
    suggested_na: float = 0.0    # object-side NA for finite conjugates
    fov_range: Tuple[float, float] = (0.0, 0.0)   # half-field angle range
    fno_range: Tuple[float, float] = (0.0, 0.0)
    # Target back focal length / EFL.  Left at 0 it is measured from the
    # prescription itself by _resolve_bfl_ratios() below, which keeps the
    # optimizer from fighting the form's own natural back focus.
    bfl_ratio: float = 0.0


def build(p: Prototype) -> Design:
    """Return the prototype as a Design object."""
    return Design(radius=list(p.radius), thick=list(p.thick),
                  glass=list(p.glass), stop=p.stop, title=p.name,
                  efl=p.f0, obj_type="finite" if p.finite else "infinite",
                  obj_dist=p.obj_dist)


def _gl(ids: str) -> str:
    """Attach the SCHOTT catalog suffix to a bare glass name."""
    base = ids if ids.endswith("_SCHOTT") else ids + "_SCHOTT"
    return base


# ===========================================================================
# 1. Double Gauss  (photographic standard; values from a published prescription)
# ===========================================================================
double_gauss = Prototype(
    id="double_gauss",
    name="Double Gauss",
    description="Symmetric six-element photographic objective. Wide field "
                "aberration balance; workhorse for 30-60 deg fields, F/2-F/4.",
    f0=100.0,
    radius=[float("inf"),
            56.20141, 152.28332, 37.68333, 0.0, 24.23147, 0.0,
            -28.37722, 0.0, -37.92544, 177.41524, -79.40703, float("inf")],
    thick=[1e13,
           8.75, 0.5, 12.5, 3.8, 16.36944, 13.74796,
           3.8, 11.0, 0.5, 7.0, 61.49],
    glass=["", "NSSK2_SCHOTT", "", "NSK2_SCHOTT", "F5_SCHOTT", "", "",
           "F5_SCHOTT", "NSK16_SCHOTT", "", "NSK16_SCHOTT", ""],
    stop=6,
    fov_range=(30.0, 60.0),
    fno_range=(1.4, 4.0),
)

# ===========================================================================
# 2. Cooke triplet
# ===========================================================================
cooke_triplet = Prototype(
    id="cooke_triplet",
    name="Cooke Triplet",
    description="Three-element (+ - +) photographic lens. Simple, cheap, "
                "light. Moderate fields at F/3.5-F/8.",
    f0=100.0,
    radius=[1e13, 77.83, -102.89, -123.12, 123.12, 101.14, -76.95, 1e13],
    thick=[1e13, 4.0, 2.5, 1.2, 10.0, 4.0, 88.0],
    glass=["", "NBK7_SCHOTT", "", "NSF5_SCHOTT", "", "NBK7_SCHOTT", ""],
    stop=4,
    fov_range=(30.0, 55.0),
    fno_range=(3.5, 8.0),
)

# ===========================================================================
# 3. Telephoto
# ===========================================================================
telephoto = Prototype(
    id="telephoto",
    name="Telephoto",
    description="Positive front group + negative rear group compress the tube "
                "(EFL/overall > 1). Medium-narrow fields, long focal lengths.",
    f0=100.0,
    radius=[1e13, 24.85, -15.69, -40.54, -18.31, -45.77, 1e13],
    thick=[1e13, 5.5, 2.2, 13.0, 1.8, 45.0],
    glass=["", _gl("NBK7"), _gl("NSF5"), "", _gl("NSF10"), ""],
    stop=1,
    fov_range=(10.0, 35.0),
    fno_range=(2.8, 5.6),
)

# ===========================================================================
# 4. Retrofocus / inverted telephoto (wide angle)
# ===========================================================================
retrofocus = Prototype(
    id="retrofocus",
    name="Retrofocus (wide angle)",
    description="Negative front + positive rear places the entrance pupil deep "
                "(long back focal). Standard for wide-angle / mirrorless.",
    f0=100.0,
    radius=[1e13, -269.6, 168.5, 151.6, -117.9, 336.9, -263.6, 1e13],
    thick=[1e13, 4.0, 14.0, 6.0, 22.0, 5.0, 60.0],
    glass=["", "NSF5_SCHOTT", "", "NSF6_SCHOTT", "", "NBK7_SCHOTT", ""],
    stop=5,
    fov_range=(55.0, 110.0),
    fno_range=(2.8, 5.6),
)

# ===========================================================================
# 5. Petzval (projector / relay)
# ===========================================================================
petzval = Prototype(
    id="petzval",
    name="Petzval (projector)",
    description="Two positive groups gives high speed at small field. Common "
                "for projectors, relays and fast front sections.",
    f0=100.0,
    radius=[1e13, 94.7, -75.7, -189.3, 69.4, -56.8, 110.5, 1e13],
    thick=[1e13, 6.0, 2.0, 26.0, 5.0, 2.0, 40.0],
    glass=["", "NBK7_SCHOTT", "NSF5_SCHOTT", "", "NSF5_SCHOTT", "NBK7_SCHOTT", ""],
    stop=3,
    fov_range=(5.0, 25.0),
    fno_range=(1.4, 2.8),
)

# ===========================================================================
# 6. Microscope objective (finite conjugate)
# ===========================================================================
microscope = Prototype(
    id="microscope",
    name="Microscope objective",
    description="Finite-conjugate high-aperture objective (object at working "
                "distance, image near tube). Achromatic; NA ~0.1-0.5.",
    f0=100.0,
    radius=[1e13, 12.0, -30.0, -12.0, 30.0, 8.0, -40.0, 1e13],
    thick=[1.0, 3.0, 4.0, 3.0, 1.2, 2.5, 120.0],
    glass=["", "NBK7_SCHOTT", "NSF5_SCHOTT", "NBK7_SCHOTT", "NSF10_SCHOTT", "", ""],
    stop=5,
    finite=True,
    obj_dist=1.0,
    suggested_na=0.25,
    fov_range=(0.0, 2.0),
    fno_range=(1.0, 4.0),
)

# ===========================================================================
# 7. Collimator / projector objective
# ===========================================================================
collimator = Prototype(
    id="collimator",
    name="Collimator / projector",
    description="Collimates a small source; optimized for large aperture "
                "(low F/#) at narrow field.",
    f0=100.0,
    radius=[1e13, 182.7, -261.0, 130.5, 293.6, -228.4, 1e13],
    thick=[1e13, 8.0, 1.0, 6.0, 3.0, 70.0],
    glass=["", "NBK7_SCHOTT", "", "NSF5_SCHOTT", "NBK7_SCHOTT", ""],
    stop=2,
    fov_range=(0.5, 8.0),
    fno_range=(1.4, 3.0),
)

CATALOG: dict = {
    p.id: p for p in [double_gauss, cooke_triplet, telephoto, retrofocus,
                      petzval, microscope, collimator]
}


def get(pid: str) -> Prototype:
    if pid not in CATALOG:
        raise KeyError(f"unknown prototype '{pid}'")
    return CATALOG[pid]


def _resolve_bfl_ratios() -> None:
    """Measure each prototype's own back-focal ratio from its prescription.

    The optimizer pulls the back focus toward bfl_ratio * EFL.  Deriving that
    ratio from the stored prescription (rather than declaring it by hand) means
    the target can never drift out of step with the geometry: the form keeps
    the back focus it was designed with, and the user's image-clearance goal is
    what moves it.
    """
    from . import optics

    for p in CATALOG.values():
        if p.bfl_ratio:
            continue
        if p.finite:
            p.bfl_ratio = 1.0
            continue
        d = build(p)
        try:
            rt = optics.trace(d, epd=p.f0 / 4.0, field_angle_deg=1.0)
            if rt.efl > 0 and rt.bfl > 0:
                p.bfl_ratio = rt.bfl / rt.efl
            else:
                p.bfl_ratio = 1.0
        except Exception:
            p.bfl_ratio = 1.0


_resolve_bfl_ratios()
