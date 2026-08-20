"""Optical glass catalog subset (Schott) used for starting-point designs.

Each entry stores the d-line refractive index and the Abbe number so the local
paraxial / third-order engine can evaluate both monochromatic and chromatic
performance. The NAME here is the glass catalog string, e.g. "NBK7_SCHOTT".
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
    """Return a Glass for a glass catalog string (case-insensitive)."""
    key = code.strip().upper()
    if key not in CATALOG:
        raise KeyError(f"Unknown glass '{code}'. Add it to optiforge/glass.py CATALOG.")
    nd, vd = CATALOG[key]
    return Glass(key, nd, vd)


def index(code: str) -> float:
    return get(code).nd


def abbe(code: str) -> float:
    return get(code).vd


# ---------------------------------------------------------------------------
# Dispersion model
# ---------------------------------------------------------------------------
# The catalog stores only (nd, Vd), which is exactly two pieces of information,
# so the dispersion curve is reconstructed with a two-term Cauchy law
#     n(lam) = A + B / lam^2        (lam in micrometers)
# fitted so that n(lam_d) = nd and n(lam_F) - n(lam_C) = (nd - 1) / Vd.
# That is the standard first-order reconstruction and is accurate enough for
# the paraxial / third-order starting-point engine.  It lets the user's own
# wavelength range (the "Short"/"Long" fields of the dialog) drive the
# chromatic balance instead of the fixed F/d/C lines.
LAM_D = 0.5875618   # d line   [um]
LAM_F = 0.4861327   # F line   [um]
LAM_C = 0.6562725   # C line   [um]


def cauchy(code: str) -> tuple:
    """Return the (A, B) Cauchy coefficients for a catalog glass."""
    g = get(code)
    dn = (g.nd - 1.0) / g.vd if g.vd else 0.0
    B = dn / (1.0 / LAM_F ** 2 - 1.0 / LAM_C ** 2)
    A = g.nd - B / LAM_D ** 2
    return A, B


def index_at(code: str, wl_nm: float) -> float:
    """Refractive index of `code` at a wavelength given in nanometres."""
    if wl_nm is None or wl_nm <= 0:
        return index(code)
    A, B = cauchy(code)
    lam = wl_nm / 1000.0
    return A + B / (lam * lam)


def v_number(code: str, wl_short: float, wl_long: float,
             wl_ref: float = None) -> float:
    """Effective Abbe number over the user's own wavelength range.

    V = (n_ref - 1) / (n_short - n_long).  Falls back to the catalog Vd when
    the range is degenerate.
    """
    if not wl_short or not wl_long or abs(wl_short - wl_long) < 1e-6:
        return abbe(code)
    ref = wl_ref if wl_ref else 0.5 * (wl_short + wl_long)
    ns = index_at(code, wl_short)
    nl = index_at(code, wl_long)
    nr = index_at(code, ref)
    denom = ns - nl
    if abs(denom) < 1e-9:
        return abbe(code)
    return (nr - 1.0) / denom
