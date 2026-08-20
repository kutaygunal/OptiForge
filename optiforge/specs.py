"""User specification model and aperture/field conversions.

The specification mirrors the OptiForge dialog: a handful of *required*
first-order targets (units, focal length, pupil, semi-field angle, wavelength
range) plus a set of *optional* goals that can each be switched on or off
(package length, image clearance, distortion, chief ray angle at the image,
element count).  Everything the generator ranks its candidate systems against
lives here.
"""
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

# Desired system units -> (label, DIM token, millimetres per unit)
UNITS = {
    "mm": ("Millimeters", "MM", 1.0),
    "cm": ("Centimeters", "CM", 10.0),
    "in": ("Inches", "IN", 25.4),
}


@dataclass
class Spec:
    """The set of inputs the engineer provides.

    Aperture ("pupil specification") is given as one of EPD / F# / NA
    (image-side for infinite conjugates, object-side NA for finite /
    microscope conjugates).  The field may be entered either as the full field
    of view (fov_deg) or as the semi-field angle (semi_field_deg); the two are
    kept consistent by :meth:`compute`.
    """
    efl: float = 50.0            # effective focal length [system units]
    fov_deg: float = 40.0        # full field of view [degrees]
    semi_field_deg: Optional[float] = None   # semi-field angle [deg] (wins if set)
    aperture: str = "fno"        # 'epd' | 'fno' | 'na'
    aperture_value: float = 2.8  # value of the chosen aperture
    lens_type: str = "auto"      # one of LENS_TYPES
    units: str = "mm"            # 'mm' | 'cm' | 'in'

    # wavelength range (nm)
    wl_short: float = 486.1
    wl_long: float = 656.3

    # --- optional goals (each with an on/off flag, like the dialog) --------
    use_package_length: bool = False
    package_length_max: float = 0.0     # maximum package length
    use_image_clearance: bool = False
    min_image_clearance: float = 0.0    # minimum image clearance (back focal)
    use_distortion: bool = True
    distortion_max: float = 3.0         # maximum distortion [%, +/-]
    use_cra: bool = False
    cra_target: float = 0.0             # chief ray angle at image, max field [deg]
    cra_tolerance: float = 10.0         # +/- tolerance [deg]
    use_elem_min: bool = False
    elem_min: int = 3
    use_elem_max: bool = False
    elem_max: int = 8

    # finite conjugates (microscope)
    object_distance: float = math.inf
    object_height: float = 0.0

    # --- generation options ----------------------------------------------
    n_systems: int = 10          # number of desired systems
    base_name: str = "Lens"      # base file name for the generated systems

    # performance bookkeeping (filled by compute())
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
        # field: semi-field angle wins when supplied
        if self.semi_field_deg is not None:
            self.fov_deg = 2.0 * self.semi_field_deg
        self.half_field_deg = self.fov_deg / 2.0
        self.semi_field_deg = self.half_field_deg
        if self.wl_short > self.wl_long:      # tolerate swapped entry
            self.wl_short, self.wl_long = self.wl_long, self.wl_short
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
        if self.units not in UNITS:
            problems.append(f"units must be one of {sorted(UNITS)}")
        if self.wl_short <= 0 or self.wl_long <= 0:
            problems.append("wavelengths must be positive")
        if self.use_package_length and self.package_length_max <= 0:
            problems.append("maximum package length must be positive")
        if self.use_cra and self.cra_tolerance < 0:
            problems.append("chief ray angle tolerance must be >= 0")
        lo, hi = self.elem_range()
        if lo > hi:
            problems.append("minimum element count exceeds the maximum")
        return problems

    # --- convenience -------------------------------------------------------
    @property
    def aperture_kind_label(self) -> str:
        return {"epd": "EPD", "fno": "F/#", "na": "NA"}.get(self.aperture, self.aperture)

    @property
    def unit_label(self) -> str:
        return UNITS.get(self.units, UNITS["mm"])[0]

    @property
    def dim_token(self) -> str:
        return UNITS.get(self.units, UNITS["mm"])[1]

    @property
    def wl_ref(self) -> float:
        """Reference (central) wavelength of the requested range."""
        return 0.5 * (self.wl_short + self.wl_long)

    def wavelengths(self) -> List[float]:
        """The three wavelengths written to the .seq: long, reference, short."""
        return [self.wl_long, self.wl_ref, self.wl_short]

    def elem_range(self) -> tuple:
        """Allowed element count as a (min, max) pair; unbounded -> (1, 99)."""
        lo = int(self.elem_min) if self.use_elem_min else 1
        hi = int(self.elem_max) if self.use_elem_max else 99
        return lo, hi

    def goals(self) -> dict:
        """The enabled goals, as a plain dict (used by the summary table)."""
        g = {
            "epd": self.epd,
            "semi_field_deg": self.half_field_deg,
            "efl": self.efl,
            "wl_short": self.wl_short,
            "wl_long": self.wl_long,
        }
        g["package_length"] = self.package_length_max if self.use_package_length else None
        g["image_clearance"] = self.min_image_clearance if self.use_image_clearance else None
        g["distortion_pct"] = self.distortion_max if self.use_distortion else None
        g["cra_deg"] = self.cra_target if self.use_cra else None
        g["cra_tolerance"] = self.cra_tolerance if self.use_cra else None
        lo, hi = self.elem_range()
        g["elem_min"] = lo if self.use_elem_min else None
        g["elem_max"] = hi if self.use_elem_max else None
        return g
