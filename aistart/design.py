"""Data model for an optical design (a CODE V lens sequence in memory)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


def _r_to_c(r: float) -> float:
    """Convert a radius to a curvature. INF/0 -> plane."""
    if r == 0.0 or not np.isfinite(r):
        return 0.0
    return 1.0 / r


@dataclass
class Design:
    """A sequential optical system.

    Surfaces are indexed 0..N-1 where surface 0 is the object (dummy, R=INF)
    and the last surface is the image surface (dummy). Surfaces 1..N-2 are the
    real refracting surfaces.

    Attributes:
        radius: N radii (RDY).  Surface 0 = object (INF). Last = image (INF).
        thick:  N-1 gaps; thick[k] = distance from surface k to k+1. The final
                element (thick[-2]... actually) uses thick between real
                surfaces; the object gap (thick[0]) is the object distance.
                The last gap is the image distance (back focal).
        glass:  N-1 media AFTER each surface k (glass code or '' for air).
                glass[k] applies to the gap from surface k to k+1.
        stop:   index of the aperture stop surface (1..N-2).
        title:  lens title.
        unit:   length unit label used when exporting ('MM').
        efl:    nominal effective focal length of the design.
        fnum:   working f-number (None until evaluated).
        obj_type: 'infinite' or 'finite' object conjugate.
    """

    radius: List[float] = field(default_factory=list)
    thick: List[float] = field(default_factory=list)
    glass: List[str] = field(default_factory=list)
    stop: int = 1
    title: str = ""
    unit: str = "MM"
    efl: float = 0.0
    fno: Optional[float] = None
    obj_type: str = "infinite"
    obj_dist: float = np.inf

    # --- convenience helpers ------------------------------------------------
    @property
    def n_real(self) -> int:
        """Number of real (non-object, non-image) surfaces."""
        return len(self.radius) - 2

    @property
    def curv(self) -> np.ndarray:
        return np.array([_r_to_c(r) for r in self.radius], dtype=float)

    @property
    def thi(self) -> np.ndarray:
        return np.array(self.thick, dtype=float)

    def copy(self) -> "Design":
        return Design(
            radius=list(self.radius),
            thick=list(self.thick),
            glass=list(self.glass),
            stop=self.stop,
            title=self.title,
            unit=self.unit,
            efl=self.efl,
            fno=self.fno,
            obj_type=self.obj_type,
            obj_dist=self.obj_dist,
        )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        parts = [f"Design({self.title!r} stop=S{self.stop} efl={self.efl:.3f}"]
        parts.append(f"n_surfaces={len(self.radius)})")
        return "".join(parts)


def from_arrays(radius, thick, glass, stop=1, title="", efl=0.0, **kw) -> Design:
    """Build a Design from surface arrays."""
    d = Design(radius=list(radius), thick=list(thick), glass=list(glass),
               stop=stop, title=title, efl=efl, **kw)
    return d
