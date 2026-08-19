"""Paraxial optics engine: ray tracing, first-order data, and third-order
(Seidel) aberration coefficients for an on-axis starting-point lens.

Works entirely with the design's *radii* (RDY), thicknesses and media, using
the classical reduced-angle (y, V = n*u) paraxial formalism.  Dependency-light
(numpy only) so the whole tool runs fully on the operator's machine.

Seidel coefficients follow the classical formulation (units of length):
  S1 spherical, S2 coma, S3 astigmatism, S4 Petzval/field, S5 distortion.
Chromatic aberration uses the thin-element dispersion law over each element:
  axial  ~ sum h^2 * phi / V,   lateral ~ sum h*hbar * phi / V.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import glass
from .design import Design


def _med(design: Design, k: int) -> float:
    """Refractive index in the gap AFTER surface k (air when no glass)."""
    if k < 0 or k >= len(design.glass):
        return 1.0
    g = design.glass[k]
    if not g or g.strip() == "":
        return 1.0
    return glass.index(g)


def _ref_arrays(design: Design):
    N = len(design.radius)
    curv = design.curv
    n_before = np.zeros(N)
    n_after = np.zeros(N)
    for i in range(N):
        n_before[i] = _med(design, i - 1) if i > 0 else 1.0
        n_after[i] = _med(design, i) if i < N - 1 else (_med(design, N - 2) if N > 2 else 1.0)
    return curv, n_before, n_after


def _fwd_trace(design, start, y0, U0):
    """Forward paraxial trace from surface `start` with (y0, U0) BEFORE start.

    Returns (y[], V[]) arrays over all surfaces. V is the reduced angle
    (n*u) in the medium before each surface.
    """
    N = len(design.radius)
    curv, n_before, n_after = _ref_arrays(design)
    y = np.zeros(N)
    V = np.zeros(N)
    y[start] = y0
    V[start] = U0
    for i in range(start, N):
        nb = n_before[i]
        na = n_after[i]
        Ua = V[i] - y[i] * (na - nb) * curv[i]
        if i < N - 1:
            ua = Ua / na
            y[i + 1] = y[i] + design.thick[i] * ua
            V[i + 1] = Ua
        else:
            V[i] = Ua          # post-refraction reduced angle at the image
    return y, V


@dataclass
class RayTrace:
    """Per-surface paraxial ray data plus first-order properties."""
    y: np.ndarray        # marginal ray height at each surface
    V_m: np.ndarray      # marginal reduced angle (n*u) before each surface
    y_b: np.ndarray      # chief ray height at each surface
    V_c: np.ndarray      # chief reduced angle before each surface
    curv: np.ndarray
    n_before: np.ndarray
    n_after: np.ndarray
    efl: float
    bfl: float
    epd: float
    fno: float
    optical_invariant: float
    marginal_img_angle: float
    stop: int
    finite: bool
    obj_height: float = 0.0
    field_angle: float = 0.0


def trace(design: Design, epd: float = 0.0, field_angle_deg: float = 0.0,
          na_obj: float = 0.0, obj_height: float = 0.0,
          finite: bool = False) -> RayTrace:
    """Build the marginal and chief rays of a design.

    Infinite conjugate uses `epd` (entrance pupil diameter) to fix the
    aperture and `field_angle_deg` (half-field angle) for the field.
    Finite conjugate (e.g. microscope) uses `na_obj` (object-side NA) and
    `obj_height` (object height).
    """
    N = len(design.radius)
    stop = design.stop

    if not finite:
        yA, VA = _fwd_trace(design, 1, 1.0, 0.0)   # marginal basis (parallel)
        yB, VB = _fwd_trace(design, 1, 0.0, 1.0)   # chief basis (unit angle)
        if epd is None or epd <= 0:
            raise ValueError("epd required for infinite conjugate")
        h1 = yA[stop]
        if abs(h1) < 1e-12:
            raise ValueError("stop has zero marginal height (invalid design)")
        k = (epd / 2.0) / h1
        ym, Vm = k * yA, k * VA
        h2 = yB[stop]
        fac = h2 / h1
        th = math.radians(field_angle_deg)
        yb = th * (yB - fac * yA)
        Vb = th * (VB - fac * VA)
        epd_ = epd
    else:
        yA, VA = _fwd_trace(design, 0, 0.0, 1.0)   # (y=0, U=1)
        yB, VB = _fwd_trace(design, 0, 1.0, 0.0)   # (y=1, U=0)
        ya_s = yA[stop]
        if na_obj and na_obj > 0:
            ym, Vm = na_obj * yA, na_obj * VA
            epd_ = abs(2.0 * ym[stop])
        else:
            ym = (epd / 2.0) / ya_s * yA
            Vm = (epd / 2.0) / ya_s * VA
            epd_ = epd
        if ya_s != 0:
            A_coef = -obj_height * yB[stop] / ya_s
        else:
            A_coef = 0.0
        yb = A_coef * yA + obj_height * yB
        Vb = A_coef * VA + obj_height * VB

    V_img = Vm[N - 1]
    y_last = ym[N - 2]          # height at the LAST REAL surface
    bfl = -y_last / V_img if abs(V_img) > 1e-12 else 0.0
    efl = -ym[1] / V_img if abs(V_img) > 1e-12 else 0.0

    nb1 = _med(design, 0) if N > 1 else 1.0
    H = nb1 * (ym[1] * Vb[1] - yb[1] * Vm[1])
    fno = efl / epd_ if epd_ else 0.0

    return RayTrace(
        y=ym, V_m=Vm, y_b=yb, V_c=Vb,
        curv=design.curv, n_before=_order_b(design), n_after=_order_a(design),
        efl=efl, bfl=bfl, epd=epd_, fno=fno, optical_invariant=H,
        marginal_img_angle=Vm[N - 1], stop=stop,
        finite=finite, obj_height=obj_height, field_angle=field_angle_deg,
    )


def _order_a(design):
    N = len(design.radius)
    return np.array([_med(design, i) if i < N - 1 else (_med(design, N - 2) if N > 2 else 1.0) for i in range(N)])


def _order_b(design):
    return np.array([_med(design, i - 1) if i > 0 else 1.0 for i in range(len(design.radius))])


@dataclass
class Seidel:
    """Third-order (Seidel) coefficients and chromatic terms."""
    s1: float  # spherical
    s2: float  # coma
    s3: float  # astigmatism
    s4: float  # Petzval / field curvature
    s5: float  # distortion
    lch: float # axial chromatic
    tch: float # lateral chromatic

    def array(self) -> np.ndarray:
        return np.array([self.s1, self.s2, self.s3, self.s4, self.s5])


def seidel(rt: RayTrace, design: Design) -> Seidel:
    """Compute third-order + chromatic coefficients from a RayTrace."""
    y = rt.y
    yb = rt.y_b
    V = rt.V_m
    Vb = rt.V_c
    curv = rt.curv
    nb = rt.n_before
    na = rt.n_after
    H = rt.optical_invariant

    s1 = s2 = s3 = s4 = s5 = 0.0
    nreal = len(y)
    for i in range(1, nreal - 1):
        h = y[i]
        hbar = yb[i]
        c = curv[i]
        if abs(c) < 1e-12:
            continue
        u = V[i]
        ub = Vb[i]
        A = u + h * c
        Ab = ub + hbar * c
        dnu = -h * (na[i] - nb[i]) * c
        if abs(A) < 1e-12:
            continue
        s1 += A * A * h * dnu
        s2 += A * Ab * h * dnu
        s3 += Ab * Ab * h * dnu
        petz = H * H * c * (1.0 / nb[i] - 1.0 / na[i]) if nb[i] != na[i] else 0.0
        s4 += petz
        s5 += (Ab * Ab * Ab / A) * h * dnu + Ab * petz

    lch, tch = _chromatic(design, rt)

    return Seidel(s1=s1, s2=s2, s3=s3, s4=s4, s5=s5, lch=lch, tch=tch)


def _chromatic(design: Design, rt: RayTrace) -> tuple:
    """Element-based axial (lch) and lateral (tch) chromatic coefficients."""
    lch = tch = 0.0
    n = len(design.radius)
    k = 0
    while k < n - 1:
        g = design.glass[k] if k < len(design.glass) else ""
        # find an element (a run of the same non-air glass)
        if g and g.strip():
            gcode = g
            ng = glass.index(gcode)
            vd = glass.abbe(gcode)
            k2 = k
            while k2 < len(design.glass) and design.glass[k2] == gcode:
                k2 += 1
            # element spans surfaces k .. k2 (k2 is the exit surface)
            if 1 <= k and k2 < n:
                c1 = rt.curv[k]
                c2 = rt.curv[k2] if k2 < n else 0.0
                phi = (ng - 1.0) * (c1 - c2)
                h_avg = 0.5 * (abs(rt.y[k]) + abs(rt.y[k2]))
                hb_avg = 0.5 * (abs(rt.y_b[k]) + abs(rt.y_b[k2]))
                if vd:
                    lch += h_avg * h_avg * phi / vd
                    tch += h_avg * hb_avg * phi / vd
            k = k2
        else:
            k += 1
    return lch, tch


def analyze(design: Design, epd: float, field_angle_deg: float = 0.0,
            na_obj: float = 0.0, obj_height: float = 0.0,
            finite: bool = False) -> dict:
    """One-stop evaluation: return a dict of performance quantities.

    Used by the merit function and the report generator.
    """
    rt = trace(design, epd=epd, field_angle_deg=field_angle_deg,
               na_obj=na_obj, obj_height=obj_height, finite=finite)
    sd = seidel(rt, design)
    return {
        "raytrace": rt,
        "seidel": sd,
        "efl": rt.efl,
        "bfl": rt.bfl,
        "epd": rt.epd,
        "fno": rt.fno,
        "S1": sd.s1, "S2": sd.s2, "S3": sd.s3, "S4": sd.s4, "S5": sd.s5,
        "LCH": sd.lch, "TCH": sd.tch,
        "H": rt.optical_invariant,
    }
