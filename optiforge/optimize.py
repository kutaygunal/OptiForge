"""Local merit-function optimizer for starting-point lens design.

Holds the target EFL exactly (by scaling all curvatures each evaluation) and
refocuses the image plane, then shapes the lens to reduce third-order and
chromatic aberrations subject to physical feasibility (minimum thicknesses,
minimum image clearance).  This is the "built-in intelligence for lens shape
and sensitivity" — it turns a scaled classical form into an optically sensible
starting point ready for detailed optimization.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.optimize import minimize

from . import metrics, optics
from .design import Design
from .specs import Spec

# default merit weights (relative importance of each error term)
WEIGHTS = {
    "S1": 1.0, "S2": 0.85, "S3": 0.7, "S4": 0.55, "S5": 0.45,
    "LCH": 0.5, "TCH": 0.35,
    "EFL": 1e4, "BFL": 2.0,
    # goal terms: only active when the corresponding goal is switched on, and
    # weighted high enough to actually bind against the aberration balance
    "PKG": 60.0, "CLR": 60.0, "DIST": 40.0, "CRA": 40.0,
    "SAG": 100.0,
}

# A spherical surface cannot be steeper than a hemisphere over its own clear
# aperture: |R| must exceed the semi-aperture it has to cover, with margin for
# the edge thickness.  Paraxial optics never notices this, so it has to be an
# explicit constraint or the optimizer will happily return unbuildable glass.
MIN_RADIUS_FACTOR = 1.10

# Minimum centre thickness, air space and edge thickness, as a fraction of the
# clear semi-aperture at that gap - and a maximum, because an element thicker
# than its own clear aperture is a glass rod, not a lens.  Real elements run
# about 0.15-0.35 of their diameter; 1.0 x semi-aperture (0.5 x diameter)
# leaves plenty of room while ruling out the degenerate solutions the merit
# function would otherwise reach for.
MIN_CENTER_RATIO = 0.10
MAX_CENTER_RATIO = 1.00
MIN_AIR_RATIO = 0.02
MIN_EDGE_RATIO = 0.02


def _sag(c: float, h: float) -> float:
    """Sag of a spherical surface of curvature c at height h."""
    if abs(c) < 1e-12:
        return 0.0
    disc = 1.0 - (c * h) ** 2
    if disc <= 0.0:
        # steeper than a hemisphere: clamp at the widest point the sphere has
        return 1.0 / c
    return c * h * h / (1.0 + math.sqrt(disc))


def _edge_thickness(r1: float, r2: float, t: float, h: float) -> float:
    """Thickness of an element at height h (its edge, given a clear radius h)."""
    c1 = 1.0 / r1 if r1 and np.isfinite(r1) else 0.0
    c2 = 1.0 / r2 if r2 and np.isfinite(r2) else 0.0
    return t + _sag(c2, h) - _sag(c1, h)

# A one-sided penalty has zero gradient at the boundary, so the optimizer
# settles a hair on the wrong side of an inequality goal.  Aiming this far
# inside the limit lands the result safely on the right side of it.
GOAL_MARGIN = 0.005


class Optimizer:
    """Optimize a starting-point design for the requested spec."""

    def __init__(self, base: Design, efl_target: float, epd: float,
                 field_angle_deg: float, na_obj: float = 0.0,
                 obj_height: float = 0.0, finite: bool = False,
                 min_bfl: float = 0.0, obj_dist: float = None,
                 bfl_ratio: float = 1.0,
                 weights: dict | None = None,
                 max_iter: int = 500,
                 pkg_max: float = None, clearance_min: float = None,
                 distortion_max: float = None,
                 cra_target: float = None, cra_tol: float = 0.0):
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
        # optional goals (None = not requested by the user)
        self.pkg_max = pkg_max
        self.clearance_min = clearance_min if clearance_min else (min_bfl or None)
        self.distortion_max = distortion_max
        self.cra_target = cra_target
        self.cra_tol = cra_tol or 0.0
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
        m += self._goal_penalty(d, rt, sd)
        # Manufacturability on the FINAL (post-scale) geometry.
        # EFL-normalization homothetically scales gaps each evaluation, so we
        # must penalize the final thicknesses directly rather than rely on the
        # (pre-scale) bounds.
        # What a glass blank can actually be is set by its *aperture*, not by
        # the focal length: a 0.3 mm centre thickness is fine on a 2 mm lens
        # and impossible on a 15 mm one.  Both the centre and the edge are
        # checked, so the optimizer cannot pay for aberration correction with
        # knife-edged elements.
        e = self.efl
        for i in range(1, len(d.thick) - 1):
            is_glass = (i < len(d.glass)) and d.glass[i] and d.glass[i].strip()
            sa = max(metrics.clear_semi_aperture(rt, i),
                     metrics.clear_semi_aperture(rt, i + 1))
            t = d.thick[i]
            if is_glass:
                lo = max(MIN_CENTER_RATIO * sa, 0.01 * e)
                hi = min(MAX_CENTER_RATIO * sa, 0.9 * e) if sa > 0 else 0.9 * e
                hi = max(hi, lo * 1.5)
            else:
                lo = max(MIN_AIR_RATIO * sa, 0.004 * e)
                hi = 1.2 * e
            if t < lo:
                m += 40.0 * ((lo - t) / lo) ** 2
            elif t > hi:
                m += 40.0 * ((t - hi) / hi) ** 2
            if is_glass and sa > 0:
                et = _edge_thickness(d.radius[i], d.radius[i + 1], t, sa)
                lo_e = max(MIN_EDGE_RATIO * sa, 0.006 * e)
                if et < lo_e:
                    m += 40.0 * ((lo_e - et) / lo_e) ** 2
        # surface steepness: every radius must cover its own clear aperture
        for i in range(1, len(d.radius) - 1):
            R = d.radius[i]
            if not np.isfinite(R) or R == 0.0:
                continue
            sa = metrics.clear_semi_aperture(rt, i)
            if sa <= 0:
                continue
            need = MIN_RADIUS_FACTOR * sa
            if abs(R) < need:
                m += self.w["SAG"] * ((need - abs(R)) / need) ** 2
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

    # ------------------------------------------------------------------
    def _goal_penalty(self, d: Design, rt, sd) -> float:
        """One-sided penalties for the user's optional goals.

        Each term is a hinge: zero while the goal is met, growing
        quadratically once it is violated, and normalized by the goal itself
        so the weights mean the same thing at any scale.  A goal the user did
        not switch on contributes nothing.
        """
        from . import metrics

        p = 0.0
        if self.pkg_max and self.pkg_max > 0:
            limit = self.pkg_max * (1.0 - GOAL_MARGIN)
            L = metrics.package_length(d)
            if L > limit:
                p += self.w["PKG"] * ((L - limit) / limit) ** 2
        if self.clearance_min and self.clearance_min > 0:
            limit = self.clearance_min * (1.0 + GOAL_MARGIN)
            c = metrics.image_clearance(d)
            if c < limit:
                p += self.w["CLR"] * ((limit - c) / limit) ** 2
        if self.distortion_max is not None and self.distortion_max > 0:
            limit = self.distortion_max * (1.0 - GOAL_MARGIN)
            dist = metrics.distortion_pct(sd, rt)
            if dist > limit:
                p += self.w["DIST"] * ((dist - limit) / limit) ** 2
        if self.cra_target is not None:
            cra = metrics.chief_ray_angle(rt)
            tol = self.cra_tol * (1.0 - GOAL_MARGIN)
            excess = abs(cra - self.cra_target) - tol
            if excess > 0:
                scale = max(tol, 1.0)
                p += self.w["CRA"] * (excess / scale) ** 2
        return p

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

