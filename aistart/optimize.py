"""Local merit-function optimizer for starting-point lens design.

Holds the target EFL exactly (by scaling all curvatures each evaluation) and
refocuses the image plane, then shapes the lens to reduce third-order and
chromatic aberrations subject to physical feasibility (minimum thicknesses,
minimum image clearance).  This is the "built-in intelligence for lens shape
and sensitivity" — it turns a scaled classical form into an optically sensible
starting point ready for CODE V refinement.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.optimize import minimize

from . import optics
from .design import Design
from .specs import Spec

# default merit weights (relative importance of each error term)
WEIGHTS = {
    "S1": 1.0, "S2": 0.85, "S3": 0.7, "S4": 0.55, "S5": 0.45,
    "LCH": 0.5, "TCH": 0.35,
    "EFL": 1e4, "BFL": 2.0,
}


class Optimizer:
    """Optimize a starting-point design for the requested spec."""

    def __init__(self, base: Design, efl_target: float, epd: float,
                 field_angle_deg: float, na_obj: float = 0.0,
                 obj_height: float = 0.0, finite: bool = False,
                 min_bfl: float = 0.0, obj_dist: float = None,
                 bfl_ratio: float = 1.0,
                 weights: dict | None = None,
                 max_iter: int = 500):
        self.base = base
        self.efl = efl_target
        self.epd = epd
        self.field_deg = field_angle_deg
        self.na_obj = na_obj
        self.obj_height = obj_height
        self.finite = finite
        self.min_bfl = min_bfl
        self.obj_dist = obj_dist if obj_dist is not None else base.obj_dist
        self.bfl_ratio = bfl_ratio
        self.bfl_target = bfl_ratio * efl_target
        self.w = dict(WEIGHTS)
        if weights:
            self.w.update(weights)
        self.max_iter = max_iter

        self._build_vars()
        self._norm = self._initial_norms()
        self.best = None
        self.best_x = None
        self.history = []

    # ------------------------------------------------------------------
    def _build_vars(self):
        n = len(self.base.radius)
        # real surfaces = 1 .. n-2 ; gaps free = 1 .. n-3 (interior only)
        self.var_surfaces = list(range(1, n - 1))          # curvature surfaces
        self.var_gaps = list(range(1, n - 2))              # interior gaps
        cmax = 10.0 / max(self.efl, 1e-6)
        self.c_bound = (-cmax, cmax)
        self.g_bound = (0.006 * self.efl, 1.0 * self.efl)
        self._x0 = self._pack(self.base)
        self._low = np.array([self.c_bound[0]] * len(self.var_surfaces) +
                             [self.g_bound[0]] * len(self.var_gaps))
        self._high = np.array([self.c_bound[1]] * len(self.var_surfaces) +
                              [self.g_bound[1]] * len(self.var_gaps))

    def _pack(self, d: Design) -> np.ndarray:
        xs = [1.0 / d.radius[i] if d.radius[i] and np.isfinite(d.radius[i]) else 0.0
              for i in self.var_surfaces]
        xt = [d.thick[i] for i in self.var_gaps]
        return np.array(xs + xt, dtype=float)

    def _unpack(self, x: np.ndarray) -> Design:
        d = self.base.copy()
        nc = len(self.var_surfaces)
        for j, i in enumerate(self.var_surfaces):
            c = float(x[j])
            d.radius[i] = (1.0 / c) if abs(c) > 1e-12 else float("inf")
        for j, i in enumerate(self.var_gaps):
            d.thick[i] = float(x[nc + j])
        # object gap
        if self.finite:
            d.thick[0] = self.obj_dist
        else:
            d.thick[0] = 1e13
        return d

    def _initial_norms(self) -> dict:
        d = self._unpack(self._x0)
        d = self._normalize_efl(d)
        rt = optics.trace(d, epd=self.epd, field_angle_deg=self.field_deg,
                          na_obj=self.na_obj, obj_height=self.obj_height,
                          finite=self.finite)
        sd = optics.seidel(rt, d)
        norm = {}
        for key, val in zip(["S1", "S2", "S3", "S4", "S5"],
                            [sd.s1, sd.s2, sd.s3, sd.s4, sd.s5]):
            norm[key] = max(abs(val), 1e-6)
        norm["LCH"] = max(abs(sd.lch), 1e-4)
        norm["TCH"] = max(abs(sd.tch), 1e-4)
        norm["EFL"] = self.efl
        norm["BFL"] = max(self.efl, 1e-6)
        return norm

    def _evaluate_efl(self, d: Design) -> float:
        rt = optics.trace(d, epd=self.epd, field_angle_deg=self.field_deg,
                          na_obj=self.na_obj, obj_height=self.obj_height,
                          finite=self.finite)
        return rt.efl

    def _normalize_efl(self, d: Design) -> Design:
        """Homothetically scale radii+thicknesses so the EFL equals target.

        EFL scales linearly with a pure homothety (all radii AND all gaps
        scaled by one factor), so this drives the EFL exactly to the target
        while preserving the relative shape.  Then refocus the image plane.
        For finite conjugates (microscope) we instead scale to the requested
        object distance and do not force an EFL.
        """
        if self.finite:
            # scale so the working (object) distance matches the request
            if math.isfinite(self.obj_dist) and self.obj_dist > 0:
                cur = d.thick[0] if d.thick else self.obj_dist
                if cur > 0:
                    k = self.obj_dist / cur
                    for i in range(1, len(d.radius) - 1):
                        if d.radius[i] and np.isfinite(d.radius[i]):
                            d.radius[i] *= k
                    for i in range(len(d.thick)):
                        d.thick[i] *= k
                    d.thick[0] = self.obj_dist
            return d
        rt = optics.trace(d, epd=self.epd, field_angle_deg=self.field_deg,
                          na_obj=self.na_obj, obj_height=self.obj_height,
                          finite=self.finite)
        e0 = rt.efl
        if abs(e0) < 1e-9:
            return d
        scale = self.efl / e0
        for i in range(1, len(d.radius) - 1):
            if d.radius[i] and np.isfinite(d.radius[i]):
                d.radius[i] = d.radius[i] * scale
        for i in range(len(d.thick)):
            d.thick[i] = d.thick[i] * scale
        # refocus: put the image at the paraxial focus
        rt2 = optics.trace(d, epd=self.epd, field_angle_deg=self.field_deg,
                           na_obj=self.na_obj, obj_height=self.obj_height,
                           finite=self.finite)
        if len(d.thick) > 0:
            d.thick[-1] = rt2.bfl
        return d

    def _objective(self, x: np.ndarray) -> float:
        d = self._unpack(x)
        try:
            d = self._normalize_efl(d)
        except Exception:
            return 1e9
        rt = optics.trace(d, epd=self.epd, field_angle_deg=self.field_deg,
                          na_obj=self.na_obj, obj_height=self.obj_height,
                          finite=self.finite)
        sd = optics.seidel(rt, d)
        n = self._norm
        m = 0.0
        if not self.finite:
            m += self.w["EFL"] * ((rt.efl - self.efl) / n["EFL"]) ** 2
        for key in ["S1", "S2", "S3", "S4", "S5"]:
            m += self.w[key] * (getattr(sd, key.lower()) / n[key]) ** 2
        m += self.w["LCH"] * (sd.lch / n["LCH"]) ** 2
        m += self.w["TCH"] * (sd.tch / n["TCH"]) ** 2
        if not self.finite and self.bfl_target > 0:
            bfl_err = (rt.bfl - self.bfl_target) / n["BFL"]
            m += self.w["BFL"] * bfl_err ** 2
        # Manufacturability on the FINAL (post-scale) geometry.
        # EFL-normalization homothetically scales gaps each evaluation, so we
        # must penalize the final thicknesses directly rather than rely on the
        # (pre-scale) bounds.
        e = self.efl
        for i in range(1, len(d.thick) - 1):
            is_glass = (i < len(d.glass)) and d.glass[i] and d.glass[i].strip()
            lo = (0.008 * e) if is_glass else (0.004 * e)
            hi = (0.9 * e) if is_glass else (1.2 * e)
            t = d.thick[i]
            if t < lo:
                m += 40.0 * ((lo - t) / lo) ** 2
            elif t > hi:
                m += 40.0 * ((t - hi) / hi) ** 2
        # total track length cap (front element vertex to image plane)
        oal = sum(d.thick[1:]) if len(d.thick) > 1 else 0.0
        oal_hi = 2.8 * e
        if oal > oal_hi:
            m += 30.0 * ((oal - oal_hi) / oal_hi) ** 2
        # light regularization: keep shape near the prototype (scaled to target EFL)
        base_c = self.base.curv
        scale0 = self.base.efl / self.efl if self.base.efl else 1.0
        if len(base_c) == len(rt.curv):
            for j, i in enumerate(self.var_surfaces):
                c0 = base_c[i] * scale0
                if abs(c0) > 1e-9:
                    m += 0.02 * ((rt.curv[i] - c0) / c0) ** 2
        # negative/thin thickness guard (already bounded, but double-check)
        if np.any(np.array(d.thick) < 0):
            m += 1e3
        # record best
        if self.best is None or m < self.best:
            self.best = m
            self.best_x = x.copy()
        self.history.append(m)
        return m

    # property alias
    @property
    def norm(self):
        return self._norm

    def optimize(self) -> Design:
        x0 = np.clip(self._x0, self._low, self._high)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(self._objective, x0, method="SLSQP",
                           bounds=list(zip(self._low, self._high)),
                           options={"maxiter": self.max_iter,
                                    "ftol": 1e-9, "disp": False})
        # best found design
        if self.best_x is None:
            self.best_x = x0
        d = self._unpack(np.array(self.best_x))
        d = self._normalize_efl(d)
        if not self.finite:
            d.efl = self.efl
            d.fno = self.efl / self.epd if self.epd else 0.0
        d.title = self.base.title
        return d

