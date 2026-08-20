"""Performance metrics for a candidate system.

These are exactly the quantities the summary table reports for every generated
starting point, so both the ranking engine and the UI read from one place:

    package length      front vertex -> image plane (total track)
    image clearance     last optical surface -> image plane (back working dist)
    chief ray angle     chief ray incidence angle at the image, at max field
    distortion          third-order distortion at max field [%]
    rel. illumination   relative illumination at max field [%] (estimate)
    avg. spot diameter  RMS spot diameter averaged over the field (estimate)
    element count       number of glass elements

The aberration quantities are third-order (Seidel) estimates from the local
paraxial engine, not real-ray results: they are there to *rank and screen*
starting points before a full ray-trace analysis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import glass, optics
from .design import Design

# Relative field heights (and weights) used for the field-averaged spot size,
# matching the usual 3-zone sampling of a starting-point evaluation.
FIELD_ZONES = ((0.0, 1.0), (0.7, 1.0), (1.0, 1.0))

# Pupil sampling for the RMS spot integration.
_N_RHO = 6
_N_THETA = 12


@dataclass
class Metrics:
    """Summary-table quantities for one system."""
    efl: float = 0.0
    fno: float = 0.0
    epd: float = 0.0
    bfl: float = 0.0
    image_height: float = 0.0
    package_length: float = 0.0
    image_clearance: float = 0.0
    cra_deg: float = 0.0
    distortion_pct: float = 0.0
    rel_illum_pct: float = 0.0
    avg_spot_diam: float = 0.0
    elem_count: int = 0
    n_surfaces: int = 0
    min_radius_ratio: float = 0.0   # min |R| / clear semi-aperture (>1 buildable)
    seidel: Optional[object] = None
    raytrace: Optional[object] = None

    def as_row(self) -> dict:
        """The plain-number form used by the summary table / JSON payload."""
        return {
            "efl": self.efl,
            "fno": self.fno,
            "epd": self.epd,
            "bfl": self.bfl,
            "image_height": self.image_height,
            "package_length": self.package_length,
            "image_clearance": self.image_clearance,
            "cra_deg": self.cra_deg,
            "distortion_pct": self.distortion_pct,
            "rel_illum_pct": self.rel_illum_pct,
            "avg_spot_diam": self.avg_spot_diam,
            "elem_count": self.elem_count,
        }


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def element_count(design: Design) -> int:
    """Number of glass elements (a run of one glass code = one element)."""
    n = 0
    prev = ""
    for g in design.glass:
        code = (g or "").strip()
        if code and code != prev:
            n += 1
        prev = code
    return n


def package_length(design: Design) -> float:
    """Front vertex to image plane (total track), excluding the object gap."""
    if len(design.thick) < 2:
        return 0.0
    return float(sum(design.thick[1:]))


def image_clearance(design: Design) -> float:
    """Last optical surface to the image plane (back working distance)."""
    if not design.thick:
        return 0.0
    return float(design.thick[-1])


# Fraction of the paraxial chief-ray height that an element actually has to
# pass.  Taking the full |y| + |y_bar| would size every element for an
# unvignetted corner beam - a real fast wide-field lens vignettes roughly half
# of it away, and sizing to the unvignetted value rejects prescriptions (the
# stock double Gauss among them) that are manifestly buildable.
VIGNETTE_FACTOR = 0.5


def clear_semi_aperture(rt, i: int) -> float:
    """Clear semi-aperture required at surface i.

    Marginal ray height plus a vignetted share of the chief ray height - the
    single definition used by both the buildability metrics and the
    optimizer's manufacturability penalties.
    """
    if i < 0 or i >= len(rt.y):
        return 0.0
    return abs(float(rt.y[i])) + VIGNETTE_FACTOR * abs(float(rt.y_b[i]))


def min_radius_ratio(design: Design, rt) -> float:
    """Smallest ratio of |radius| to the clear semi-aperture it must cover.

    A spherical surface steeper than a hemisphere over its own aperture cannot
    be made, and paraxial optics gives no hint of it, so this is the cheap
    buildability check the ranking uses.  Values above 1 are buildable.
    """
    worst = float("inf")
    for i in range(1, len(design.radius) - 1):
        r = design.radius[i]
        if not math.isfinite(r) or r == 0.0:
            continue
        sa = clear_semi_aperture(rt, i)
        if sa <= 0:
            continue
        worst = min(worst, abs(r) / sa)
    return worst if math.isfinite(worst) else 99.0


def chief_ray_angle(rt) -> float:
    """Chief ray angle at the image, at maximum field [degrees].

    The chief ray's reduced angle in the last (image space) gap is its slope
    in air, so the incidence angle on the sensor is atan of it.  This is the
    quantity sensor makers constrain (the "CRA" of the lens).
    """
    v = rt.V_c
    if v is None or len(v) < 2:
        return 0.0
    return abs(math.degrees(math.atan(float(v[-1]))))


# ---------------------------------------------------------------------------
# Image quality (third-order estimates)
# ---------------------------------------------------------------------------
def distortion_pct(sd, rt) -> float:
    """Third-order distortion at maximum field, in percent.

    The transverse distortion of the full-field chief ray is S5 / (2 n'u'),
    and the paraxial image height is -H / (n'u'), so the *fractional*
    distortion is simply S5 / (2H) - independent of the image-space cone, as
    it must be.
    """
    H = rt.optical_invariant
    if abs(H) < 1e-12:
        return 0.0
    return 100.0 * abs(sd.s5) / (2.0 * abs(H))


def relative_illumination_pct(rt, design: Design) -> float:
    """Relative illumination at maximum field, in percent (estimate).

    Uses the pupil-corrected cosine-fourth law
        RI = cos^2(theta_object) * cos^2(theta_image)
    where theta_object is the field angle in object space and theta_image the
    chief ray angle in image space.  For a system whose exit pupil sits far
    from the image (a large image-space chief angle) this correctly predicts
    the strong roll-off; a near-telecentric design keeps most of its
    illumination.  Vignetting by finite clear apertures is NOT included, so
    treat this as an upper bound.
    """
    if rt.finite:
        # finite conjugate: the object-space obliquity is small, use the
        # image-space cone only
        th_obj = 0.0
    else:
        th_obj = math.radians(rt.field_angle)
    th_img = math.atan(abs(float(rt.V_c[-1]))) if len(rt.V_c) else 0.0
    ri = (math.cos(th_obj) ** 2) * (math.cos(th_img) ** 2)
    return 100.0 * max(0.0, min(1.0, ri))


def _transverse(sd, nu, f, rho, theta):
    """Third-order transverse ray aberration (ex, ey) at relative field f.

    Classical Seidel transverse aberration (Welford):
        ey = 1/(2 n'u') [ S1 r^3 cos t + S2 f r^2 (2 + cos 2t)
                          + (3 S3 + S4) f^2 r cos t + S5 f^3 ]
        ex = 1/(2 n'u') [ S1 r^3 sin t + S2 f r^2 sin 2t
                          + (S3 + S4) f^2 r sin t ]
    The Seidel sums are evaluated for the full aperture and full field, so the
    relative field f enters as the usual power of the field height.
    """
    c, s = math.cos(theta), math.sin(theta)
    c2, s2 = math.cos(2 * theta), math.sin(2 * theta)
    k = 1.0 / (2.0 * nu)
    ey = k * (sd.s1 * rho ** 3 * c
              + sd.s2 * f * rho ** 2 * (2.0 + c2)
              + (3.0 * sd.s3 + sd.s4) * f ** 2 * rho * c
              + sd.s5 * f ** 3)
    ex = k * (sd.s1 * rho ** 3 * s
              + sd.s2 * f * rho ** 2 * s2
              + (sd.s3 + sd.s4) * f ** 2 * rho * s)
    return ex, ey


def rms_spot_radius(sd, rt, f: float) -> float:
    """RMS spot radius at relative field f (monochromatic + chromatic blur)."""
    nu = float(rt.V_m[-1])
    if abs(nu) < 1e-12:
        return 0.0
    xs: List[float] = []
    ys: List[float] = []
    # ring sampling of the pupil, weighted so each ring carries equal area
    for i in range(1, _N_RHO + 1):
        rho = math.sqrt(i / _N_RHO)
        for j in range(_N_THETA):
            th = 2.0 * math.pi * j / _N_THETA
            ex, ey = _transverse(sd, nu, f, rho, th)
            xs.append(ex)
            ys.append(ey)
    if not xs:
        return 0.0
    ax = np.array(xs)
    ay = np.array(ys)
    ax -= ax.mean()          # remove the centroid (distortion is not blur)
    ay -= ay.mean()
    mono = float(np.sqrt(np.mean(ax ** 2 + ay ** 2)))
    # chromatic blur: axial color over the whole aperture, lateral color
    # growing with field
    chr_ax = abs(sd.lch) / (2.0 * abs(nu))
    chr_lat = abs(sd.tch) * f / (2.0 * abs(nu))
    return math.sqrt(mono ** 2 + chr_ax ** 2 + chr_lat ** 2)


def avg_spot_diameter(sd, rt) -> float:
    """RMS spot diameter averaged over the field zones."""
    tot = wsum = 0.0
    for f, w in FIELD_ZONES:
        tot += w * 2.0 * rms_spot_radius(sd, rt, f)
        wsum += w
    return tot / wsum if wsum else 0.0


# ---------------------------------------------------------------------------
# Chromatic terms over the user's wavelength range
# ---------------------------------------------------------------------------
def rescale_chromatic(sd, design: Design, rt, wl_short: float, wl_long: float):
    """Recompute the chromatic sums using the requested wavelength range.

    optics.seidel() uses the catalog Abbe number (F/C lines).  A design whose
    waveband is wider or narrower than F..C has proportionally more or less
    colour, so the sums are rescaled by the ratio of the effective V numbers,
    element by element, weighted by that element's contribution.
    """
    if not wl_short or not wl_long:
        return sd
    num = den = 0.0
    for g in design.glass:
        code = (g or "").strip()
        if not code:
            continue
        try:
            vd = glass.abbe(code)
            veff = glass.v_number(code, wl_short, wl_long,
                                  0.5 * (wl_short + wl_long))
        except KeyError:
            continue
        if veff:
            num += vd / veff
            den += 1.0
    if den == 0:
        return sd
    k = num / den
    sd.lch *= k
    sd.tch *= k
    return sd


# ---------------------------------------------------------------------------
# One-stop evaluation
# ---------------------------------------------------------------------------
def evaluate(design: Design, spec) -> Metrics:
    """Compute every summary-table quantity for `design` under `spec`."""
    rt = optics.trace(design, epd=spec.epd,
                      field_angle_deg=spec.half_field_deg,
                      na_obj=spec.na, obj_height=spec.object_height,
                      finite=spec.finite)
    sd = optics.seidel(rt, design)
    sd = rescale_chromatic(sd, design, rt, spec.wl_short, spec.wl_long)

    if spec.finite:
        img_h = spec.object_height
    else:
        img_h = abs(spec.efl * math.tan(math.radians(spec.half_field_deg)))

    return Metrics(
        efl=rt.efl if not spec.finite else (design.efl or rt.efl),
        fno=rt.fno,
        epd=rt.epd,
        bfl=rt.bfl,
        image_height=img_h,
        package_length=package_length(design),
        image_clearance=image_clearance(design),
        cra_deg=chief_ray_angle(rt),
        distortion_pct=distortion_pct(sd, rt),
        rel_illum_pct=relative_illumination_pct(rt, design),
        avg_spot_diam=avg_spot_diameter(sd, rt),
        elem_count=element_count(design),
        n_surfaces=len(design.radius) - 2,
        min_radius_ratio=min_radius_ratio(design, rt),
        seidel=sd,
        raytrace=rt,
    )
