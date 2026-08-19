"""User specification model and aperture/field conversions."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

# Supported lens-type targets (mapped in autotype / catalog).
LENS_TYPES = [
    "auto",
    "double_gauss",
    "cooke_triplet",
    "telephoto",
    "retrofocus",
    "petzval",
    "microscope",
    "collimator",
]

LENS_TYPE_LABELS = {
    "auto": "Auto (recommended)",
    "double_gauss": "Camera / double Gauss",
    "cooke_triplet": "Cooke triplet",
    "telephoto": "Telephoto",
    "retrofocus": "Retrofocus (wide angle)",
    "petzval": "Petzval (projector)",
    "microscope": "Microscope objective",
    "collimator": "Collimator",
}


@dataclass
class Spec:
    """The small set of inputs the engineer provides.

    Aperture is given as one of EPD / F# / NA (image-side for infinite
    conjugates, object-side NA for finite / microscope).  The field is the
    full (diagonal or vertical) field of view in degrees.
    """
    efl: float = 50.0            # effective focal length [mm]
    fov_deg: float = 40.0        # full field of view [degrees]
    aperture: str = "fno"        # 'epd' | 'fno' | 'na'
    aperture_value: float = 2.8  # value of the chosen aperture
    lens_type: str = "auto"      # one of LENS_TYPES
    # optional constraints
    distortion_max: float = 3.0  # allowed distortion [%] (0 = as-small-as-possible)
    min_image_clearance: float = 0.0  # minimum back focal length [mm]
    object_distance: float = math.inf  # finite conjugate object distance (microscope)
    object_height: float = 0.0        # finite conjugate field height
    # performance bookkeeping (filled by the generator)
    half_field_deg: float = 0.0
    epd: float = 0.0
    fno: float = 0.0
    na: float = 0.0
    finite: bool = False

    # --- derived quantities ------------------------------------------------
    def compute(self) -> "Spec":
        efl = self.efl
        if efl <= 0:
            raise ValueError("efl must be positive")
        a = self.aperture
        v = self.aperture_value
        if a == "epd":
            self.epd = v
            self.fno = efl / v if v else 0.0
            self.na = 1.0 / (2.0 * self.fno) if self.fno else 0.0
        elif a == "fno":
            self.fno = v
            self.epd = efl / v if v else 0.0
            self.na = 1.0 / (2.0 * v) if v else 0.0
        elif a == "na":
            self.na = v
            self.fno = 1.0 / (2.0 * v) if v else 0.0
            self.epd = efl / self.fno if self.fno else 0.0
        else:
            raise ValueError(f"unknown aperture kind '{a}'")
        self.half_field_deg = self.fov_deg / 2.0
        self.finite = (self.lens_type == "microscope" or
                       (math.isfinite(self.object_distance) and self.object_distance > 0))
        return self

    def validate(self) -> List[str]:
        problems = []
        if self.efl <= 0:
            problems.append("focal length must be positive")
        if self.fov_deg < 0 or self.fov_deg > 180:
            problems.append("field of view out of range (0..180 deg)")
        if self.aperture_value <= 0:
            problems.append("aperture value must be positive")
        if self.lens_type not in LENS_TYPES:
            problems.append(f"lens type must be one of {LENS_TYPES}")
        return problems

    @property
    def aperture_kind_label(self) -> str:
        return {"epd": "EPD", "fno": "F/#", "na": "NA"}.get(self.aperture, self.aperture)
