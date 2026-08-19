"""Optical glass catalog subset (Schott) used for CODE V starting-point designs.

Each entry stores the d-line refractive index and the Abbe number so the local
paraxial / third-order engine can evaluate both monochromatic and chromatic
performance. The NAME here is the CODE V glass string, e.g. "NBK7_SCHOTT".
"""
from __future__ import annotations

from dataclasses import dataclass

# nd: d-line (587.5618 nm) index.  Vd: Abbe number (Vd = (nd-1)/(nF-nC)).
CATALOG: dict[str, tuple[float, float]] = {
    # Fluoride / low-dispersion crowns
    "NFK51_SCHOTT":  (1.48664, 75.47),   # H-FK51
    "NPK51_SCHOTT":  (1.52425, 76.98),   # N-PK51
    # Crowns (low dispersion, moderate/low index)
    "NBK7_SCHOTT":   (1.51680, 64.17),   # N-BK7
    "NK5_SCHOTT":    (1.52247, 59.48),   # N-K5
    "NSK2_SCHOTT":   (1.60738, 56.46),   # N-SK2
    "NSK11_SCHOTT":  (1.59684, 60.76),   # N-SK11
    "NSK16_SCHOTT":  (1.62041, 60.34),   # N-SK16
    "NLAK22_SCHOTT": (1.65113, 56.04),   # N-LAK22
    "NLAK8_SCHOTT":  (1.71246, 53.89),   # N-LAK8
    "NSSK2_SCHOTT":  (1.65830, 50.88),   # N-SSK2
    "NSSK5_SCHOTT":  (1.65835, 50.85),   # N-SSK5
    # Flint /other dispersion
    "F2_SCHOTT":     (1.62004, 36.37),   # F2
    "F5_SCHOTT":     (1.60342, 38.03),   # F5
    "NSF1_SCHOTT":   (1.71736, 29.51),   # N-SF1
    "NSF5_SCHOTT":   (1.67270, 29.50),   # N-SF5
    "NSF6_SCHOTT":   (1.80518, 25.42),   # N-SF6
    "NSF10_SCHOTT":  (1.72814, 28.29),   # N-SF10
    "NSF11_SCHOTT":  (1.78472, 25.71),   # N-SF11
    "NSF15_SCHOTT":  (1.69895, 30.05),   # N-SF15
    "NSF57_SCHOTT":  (1.84666, 23.78),   # N-SF57
    "NSF56_SCHOTT":  (1.86829, 25.37),   # N-SF56
    "NLF9_SCHOTT":   (1.54348, 52.39),   # N-LF9
    "NLAF2_SCHOTT":  (1.74250, 35.18),   # N-LAF2
    "NLAF9_SCHOTT":  (1.85025, 32.21),   # N-LAF9
    "NCASF3_SCHOTT": (2.00272, 19.32),   # N-CASF3
    "NCASF10_SCHOTT":(1.79545, 30.10),   # N-CASF10
}

# Mapping of short prescription roles to concrete Schott glasses.
CROWN = ["NBK7_SCHOTT", "NK5_SCHOTT", "NSK2_SCHOTT", "NSK16_SCHOTT"]
FLINT = ["NSF5_SCHOTT", "NSF6_SCHOTT", "NSF10_SCHOTT", "NSF15_SCHOTT", "F2_SCHOTT"]
HIGH_LOW = ["NFK51_SCHOTT", "NPK51_SCHOTT"]   # anomalous partial dispersion, low cost


@dataclass
class Glass:
    code: str
    nd: float
    vd: float


def get(code: str) -> Glass:
    """Return a Glass for a CODE V glass string (case-insensitive)."""
    key = code.strip().upper()
    if key not in CATALOG:
        raise KeyError(f"Unknown glass '{code}'. Add it to aistart/glass.py CATALOG.")
    nd, vd = CATALOG[key]
    return Glass(key, nd, vd)


def index(code: str) -> float:
    return get(code).nd


def abbe(code: str) -> float:
    return get(code).vd
