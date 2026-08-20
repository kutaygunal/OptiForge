"""End-to-end generator: specs -> candidate population -> optimize -> rank.

Two entry points:

    generate(spec)          one starting point (the classic single answer)
    generate_systems(spec)  a ranked *set* of starting points: many viable
                            configurations that trade the goals off
                            differently, presented in a summary table for the
                            designer to choose from.

Everything runs on the local machine (numpy + scipy), matching the
"on-premise, no internet" requirement, and every system is emitted as a lens sequence
.seq file.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import (autotype, catalog, seqexport, metrics, optics, optimize,
               seqparse, variants)
from .design import Design
from .metrics import Metrics
from .specs import UNITS, Spec

# How hard the search works before it gives up looking for more distinct forms.
CANDIDATE_CAP = 60          # hard ceiling on candidates evaluated
GOAL_TOL = 1e-3             # relative slack when deciding "goal met"
SCREEN_ITERS = 110          # optimizer iterations during the screening pass
REFINE_ITERS = 320          # iterations for the systems that make the cut


@dataclass
class Result:
    spec: Spec
    design: Design
    prototype_id: str
    prototype_name: str
    perf: dict = field(default_factory=dict)
    seq_text: str = ""
    warnings: List[str] = field(default_factory=list)
    seq_valid: bool = True


@dataclass
class System:
    """One generated starting point, as shown in one summary-table row."""
    name: str
    index: int
    design: Design
    prototype_id: str
    prototype_name: str
    description: str
    metrics: Metrics
    score: float
    goal_status: Dict[str, dict] = field(default_factory=dict)
    seq_text: str = ""
    seq_valid: bool = True
    warnings: List[str] = field(default_factory=list)

    @property
    def meets_goals(self) -> bool:
        return all(g["met"] for g in self.goal_status.values())

    @property
    def filename(self) -> str:
        return f"{self.name}.seq"


@dataclass
class SystemSet:
    """The population returned to the summary table."""
    spec: Spec
    goals: dict
    systems: List[System] = field(default_factory=list)
    considered: int = 0
    elapsed: float = 0.0

    def __len__(self):
        return len(self.systems)

    def __iter__(self):
        return iter(self.systems)


# ===========================================================================
# Single starting point (unchanged public behaviour)
# ===========================================================================
def generate(spec: Spec, optimize_iters: int = 300,
             validate_seq: bool = True) -> Result:
    """Generate a starting-point design for the given specs."""
    spec.compute()
    problems = spec.validate()
    if problems:
        raise ValueError("; ".join(problems))

    finite = spec.finite
    pid = autotype.select(spec.lens_type, spec.half_field_deg, spec.fno,
                          spec.efl, finite)
    proto = catalog.get(pid)
    base = catalog.build(proto)

    design = _optimize_candidate(base, proto, spec, optimize_iters)
    m = metrics.evaluate(design, spec)

    perf = {
        "efl": m.efl, "bfl": m.bfl, "epd": m.epd, "fno": m.fno,
        "H": m.raytrace.optical_invariant,
        "S1": m.seidel.s1, "S2": m.seidel.s2, "S3": m.seidel.s3,
        "S4": m.seidel.s4, "S5": m.seidel.s5,
        "LCH": m.seidel.lch, "TCH": m.seidel.tch,
        "image_height": m.image_height,
        "distortion_pct": m.distortion_pct,
        "package_length": m.package_length,
        "image_clearance": m.image_clearance,
        "cra_deg": m.cra_deg,
        "rel_illum_pct": m.rel_illum_pct,
        "avg_spot_diam": m.avg_spot_diam,
        "n_surfaces": m.n_surfaces,
        "n_elements": m.elem_count,
    }

    seq_text = _seq_for(design, spec, title=design.title)

    warnings: List[str] = []
    seq_valid = True
    if validate_seq:
        r = seqparse.validate(seq_text,
                              expected_efl=spec.efl if not finite else None)
        seq_valid = r.ok
        warnings.extend(r.warnings)

    return Result(spec=spec, design=design, prototype_id=pid,
                  prototype_name=proto.name, perf=perf, seq_text=seq_text,
                  warnings=warnings, seq_valid=seq_valid)


# ===========================================================================
# Population of starting points
# ===========================================================================
def generate_systems(spec: Spec, validate_seq: bool = True,
                     progress=None) -> SystemSet:
    """Generate, rank and name a set of viable starting points.

    The search runs in three stages:

    1. *Enumerate* - classical forms compatible with the operating point are
       crossed with the structural moves in `variants` (element added or
       removed, cemented interface split, glass substituted) and with
       randomised restarts, giving a population of distinct configurations.
    2. *Screen* - every candidate is optimized briefly against the goals and
       scored on how well it meets them.
    3. *Refine* - the survivors are re-optimized properly, de-duplicated and
       returned in rank order.
    """
    t0 = time.time()
    spec.compute()
    problems = spec.validate()
    if problems:
        raise ValueError("; ".join(problems))

    want = max(1, int(spec.n_systems))
    cands = _enumerate_candidates(spec, want)

    screened = []
    for i, c in enumerate(cands):
        if progress:
            progress(i, len(cands), "screening")
        try:
            d = _optimize_candidate(c["design"], c["proto"], spec, SCREEN_ITERS)
            m = metrics.evaluate(d, spec)
        except Exception:
            continue
        if not _sane(d, m, spec):
            continue
        sc, status = _score(m, spec)
        screened.append({**c, "design": d, "metrics": m, "score": sc,
                         "status": status})

    screened.sort(key=lambda r: r["score"])

    # refine the best ones, de-duplicating as we go
    systems: List[System] = []
    seen = set()
    for r in screened:
        if len(systems) >= want:
            break
        if progress:
            progress(len(systems), want, "refining")
        try:
            d = _optimize_candidate(r["src_design"], r["proto"], spec,
                                    REFINE_ITERS)
            m = metrics.evaluate(d, spec)
            if not _sane(d, m, spec):
                d, m = r["design"], r["metrics"]
            elif _score(m, spec)[0] > r["score"]:
                d, m = r["design"], r["metrics"]   # keep the better of the two
        except Exception:
            d, m = r["design"], r["metrics"]
        sig = _signature(m, spec)
        if sig in seen:
            continue
        seen.add(sig)
        sc, status = _score(m, spec)
        systems.append(System(
            name="", index=0, design=d, prototype_id=r["pid"],
            prototype_name=r["proto"].name, description=r["desc"],
            metrics=m, score=sc, goal_status=status))

    systems.sort(key=lambda s: s.score)
    for i, s in enumerate(systems, start=1):
        s.index = i
        s.name = f"{spec.base_name}_{i:02d}"
        s.design.title = s.name
        s.seq_text = _seq_for(s.design, spec, title=s.name)
        if validate_seq:
            v = seqparse.validate(s.seq_text,
                                  expected_efl=spec.efl if not spec.finite else None)
            s.seq_valid = v.ok
            s.warnings = list(v.warnings) + list(v.errors)

    return SystemSet(spec=spec, goals=spec.goals(), systems=systems,
                     considered=len(cands), elapsed=time.time() - t0)


# ---------------------------------------------------------------------------
# Candidate enumeration
# ---------------------------------------------------------------------------
def _prototype_order(spec: Spec) -> List[str]:
    """Classical forms to draw from, best-suited first."""
    if spec.finite:
        return ["microscope"]
    if spec.lens_type and spec.lens_type.lower() != "auto":
        pid = autotype.resolve_id(spec.lens_type)
        if pid:
            return [pid]
    primary = autotype.auto_select(spec.half_field_deg, spec.fno, spec.efl)
    others = []
    for pid, p in catalog.CATALOG.items():
        if pid == primary or p.finite:
            continue
        others.append((_form_distance(p, spec), pid))
    others.sort()
    return [primary] + [pid for _, pid in others]


def _form_distance(proto, spec: Spec) -> float:
    """How far the operating point sits outside a form's comfort zone."""
    d = 0.0
    lo, hi = proto.fov_range
    if hi > 0:
        if spec.half_field_deg < lo:
            d += (lo - spec.half_field_deg) / max(lo, 1.0)
        elif spec.half_field_deg > hi:
            d += (spec.half_field_deg - hi) / max(hi, 1.0)
    lo, hi = proto.fno_range
    if hi > 0:
        if spec.fno < lo:
            d += (lo - spec.fno) / max(lo, 1.0)
        elif spec.fno > hi:
            d += (spec.fno - hi) / max(hi, 1.0)
    return d


def _chains(moves):
    """Single moves plus a few useful combinations."""
    by = {m.name: m for m in moves}
    out = [[m] for m in moves]
    out += [
        [by["field flattener"], by["split doublet"]],
        [by["front corrector"], by["high-index glass"]],
        [by["field flattener"], by["front corrector"]],
        [by["reduced element"], by["high-index glass"]],
        [by["split doublet"], by["alt. glass"]],
    ]
    return out


def _describe(proto_name: str, chain) -> str:
    tags = [m.name for m in chain if m.name != "as-drawn"]
    return proto_name + (" + " + " + ".join(tags) if tags else "")


def _enumerate_candidates(spec: Spec, want: int) -> List[dict]:
    """Build the candidate population (structural pass, then restarts)."""
    lo, hi = spec.elem_range()
    moves = variants.base_moves()
    chains = _chains(moves)

    structural: List[dict] = []
    for pid in _prototype_order(spec):
        proto = catalog.get(pid)
        base = catalog.build(proto)
        for chain in chains:
            d = variants.apply_moves(base, proto.f0, chain)
            if d is None:
                continue
            ec = metrics.element_count(d)
            if not (lo <= ec <= hi):
                continue
            if not _traceable(d, proto, spec):
                continue
            structural.append({
                "pid": pid, "proto": proto, "design": d, "src_design": d,
                "desc": _describe(proto.name, chain),
            })

    # The structural pass is where the diversity comes from, so it is never
    # trimmed for a small request: asking for 4 systems should search as hard
    # as asking for 10, it just returns fewer of them.
    cap = min(CANDIDATE_CAP, max(len(structural), 4 * want))

    out = list(structural)
    # randomised restarts: same structure, different local minimum
    seed = 0
    while len(out) < cap and structural and seed < 8:
        seed += 1
        for c in structural:
            if len(out) >= cap:
                break
            d = variants.perturb(c["design"], c["proto"].f0, seed=seed)
            if d is None or not _traceable(d, c["proto"], spec):
                continue
            out.append({**c, "design": d, "src_design": d,
                        "desc": f"{c['desc']} (restart {seed})"})
    return out[:cap]


# ---------------------------------------------------------------------------
# Optimization / viability
# ---------------------------------------------------------------------------
def _optimize_candidate(base: Design, proto, spec: Spec, iters: int) -> Design:
    """Run the goal-aware optimizer on one candidate."""
    obj_dist = spec.object_distance if spec.finite else None
    opt = optimize.Optimizer(
        base, spec.efl, spec.epd, spec.half_field_deg,
        na_obj=spec.na, obj_height=spec.object_height, finite=spec.finite,
        min_bfl=spec.min_image_clearance if spec.use_image_clearance else 0.0,
        obj_dist=obj_dist,
        bfl_ratio=getattr(proto, "bfl_ratio", 1.0),
        max_iter=iters,
        pkg_max=spec.package_length_max if spec.use_package_length else None,
        clearance_min=spec.min_image_clearance if spec.use_image_clearance else None,
        distortion_max=spec.distortion_max if spec.use_distortion else None,
        cra_target=spec.cra_target if spec.use_cra else None,
        cra_tol=spec.cra_tolerance if spec.use_cra else 0.0)
    d = opt.optimize()
    d.title = base.title or proto.name
    return d


def _traceable(d: Design, proto, spec: Spec) -> bool:
    """Reject a candidate that cannot be scaled to the requested first order."""
    epd_proto = (proto.f0 / spec.fno) if spec.fno else spec.epd
    try:
        rt = optics.trace(d, epd=epd_proto,
                          field_angle_deg=spec.half_field_deg,
                          na_obj=spec.na, obj_height=spec.object_height,
                          finite=spec.finite)
    except Exception:
        return False
    if spec.finite:
        return True
    if not math.isfinite(rt.efl) or rt.efl <= 0:
        return False
    # the homothety that sets the EFL must not be an absurd rescale
    ratio = rt.efl / proto.f0
    if ratio < 0.05 or ratio > 20.0:
        return False
    if not math.isfinite(rt.bfl) or rt.bfl <= 0:
        return False
    return all(t > 0 for t in d.thick[1:])


def _sane(d: Design, m: Metrics, spec: Spec) -> bool:
    """Reject an optimized design that is not physically usable."""
    if any((t is None or not math.isfinite(t) or t <= 0) for t in d.thick[1:]):
        return False
    if not math.isfinite(m.efl) or not math.isfinite(m.avg_spot_diam):
        return False
    if m.image_clearance <= 0 or m.package_length <= 0:
        return False
    # a surface steeper than a hemisphere over its own aperture cannot be made
    if m.min_radius_ratio < 1.0:
        return False
    if not spec.finite:
        if abs(m.efl - spec.efl) > 0.05 * spec.efl:
            return False
        if m.efl <= 0:
            return False
    return True


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _diffraction_spot(spec: Spec) -> float:
    """Airy disc diameter (2.44 lambda F/#) in system units."""
    mm_per_unit = UNITS.get(spec.units, UNITS["mm"])[2]
    lam = spec.wl_ref * 1e-6 / mm_per_unit      # nm -> mm -> system units
    fno = spec.fno if spec.fno else 1.0
    return 2.44 * lam * fno


def _score(m: Metrics, spec: Spec):
    """Rank a system: goal violations first, then image quality.

    Returns (score, goal_status).  Lower scores are better.  Goal violations
    are normalized by the goal itself so one weight set works at any scale.
    """
    status: Dict[str, dict] = {}
    viol = 0.0

    def add(key, value, target, met, excess=0.0, kind="max"):
        nonlocal viol
        status[key] = {"value": value, "target": target, "met": bool(met),
                       "kind": kind}
        viol += max(0.0, excess)

    # first-order targets are always goals
    if not spec.finite:
        e = abs(m.efl - spec.efl) / spec.efl if spec.efl else 0.0
        add("efl", m.efl, spec.efl, e <= 0.01, e, "target")
        if spec.fno:
            f = abs(m.fno - spec.fno) / spec.fno
            add("fno", m.fno, spec.fno, f <= 0.02, f, "target")

    if spec.use_package_length and spec.package_length_max > 0:
        x = (m.package_length - spec.package_length_max) / spec.package_length_max
        add("package_length", m.package_length, spec.package_length_max,
            x <= GOAL_TOL, x, "max")
    if spec.use_image_clearance and spec.min_image_clearance > 0:
        x = (spec.min_image_clearance - m.image_clearance) / spec.min_image_clearance
        add("image_clearance", m.image_clearance, spec.min_image_clearance,
            x <= GOAL_TOL, x, "min")
    if spec.use_distortion and spec.distortion_max > 0:
        x = (m.distortion_pct - spec.distortion_max) / spec.distortion_max
        add("distortion_pct", m.distortion_pct, spec.distortion_max,
            x <= GOAL_TOL, x, "max")
    if spec.use_cra:
        excess = abs(m.cra_deg - spec.cra_target) - spec.cra_tolerance
        add("cra_deg", m.cra_deg, spec.cra_target, excess <= GOAL_TOL,
            excess / max(spec.cra_tolerance, 1.0), "tolerance")
    lo, hi = spec.elem_range()
    if spec.use_elem_min or spec.use_elem_max:
        ok = lo <= m.elem_count <= hi
        short = max(lo - m.elem_count, m.elem_count - hi, 0)
        add("elem_count", m.elem_count, (lo, hi), ok, short * 0.5, "range")

    # image quality: how many times the diffraction limit the spot is
    diff = _diffraction_spot(spec)
    q = m.avg_spot_diam / diff if diff > 0 else m.avg_spot_diam
    quality = 8.0 * math.log10(1.0 + max(q, 0.0))
    illum = -0.4 * (m.rel_illum_pct / 100.0)      # small tie-breaker
    simple = 0.15 * m.elem_count                  # prefer the simpler system

    return 100.0 * viol + quality + illum + simple, status


def _signature(m: Metrics, spec: Spec):
    """Coarse fingerprint used to drop near-identical systems."""
    e = spec.efl if spec.efl else 1.0
    return (m.elem_count,
            round(m.package_length / e, 2),
            round(m.image_clearance / e, 2),
            round(m.cra_deg, 1),
            round(m.avg_spot_diam / e, 4))


# ---------------------------------------------------------------------------
# Export helper
# ---------------------------------------------------------------------------
def _seq_for(design: Design, spec: Spec, title: str = None) -> str:
    return seqexport.seq_string(
        design, epd=spec.epd, field_angle_deg=spec.half_field_deg,
        na_obj=spec.na, obj_height=spec.object_height, finite=spec.finite,
        wavelengths=spec.wavelengths(), reference=spec.wl_ref,
        dim=spec.dim_token, title=title)
