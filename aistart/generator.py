"""End-to-end generator: specs -> prototype selection -> optimize -> export.

This is the public entry point ("generate a starting-point lens design").  It
runs entirely on the local machine (numpy + scipy), matching the "on-premise,
no internet" requirement, and emits a CODE V .seq file plus a design report.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from . import autotype, catalog, codevexport, optics, optimize, seqparse
from .design import Design
from .specs import Spec


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


def generate(spec: Spec, optimize_iters: int = 300,
             validate_seq: bool = True) -> Result:
    """Generate a starting-point design for the given specs."""
    spec.compute()
    problems = spec.validate()
    if problems:
        raise ValueError("; ".join(problems))

    finite = spec.finite
    # select a prototype form
    pid = autotype.select(spec.lens_type, spec.half_field_deg, spec.fno,
                          spec.efl, finite)
    proto = catalog.get(pid)
    base = catalog.build(proto)

    obj_dist = spec.object_distance if finite else None
    opt = optimize.Optimizer(
        base, spec.efl, spec.epd, spec.half_field_deg,
        na_obj=spec.na, obj_height=spec.object_height, finite=finite,
        min_bfl=spec.min_image_clearance, obj_dist=obj_dist,
        bfl_ratio=proto.bfl_ratio,
        max_iter=optimize_iters)
    design = opt.optimize()

    # final performance evaluation
    rt = optics.trace(design, epd=spec.epd,
                      field_angle_deg=spec.half_field_deg,
                      na_obj=spec.na, obj_height=spec.object_height,
                      finite=finite)
    sd = optics.seidel(rt, design)
    design.efl = rt.efl if not finite else design.efl
    design.fno = rt.fno if not finite else design.fno

    perf = {
        "efl": rt.efl, "bfl": rt.bfl, "epd": rt.epd, "fno": rt.fno,
        "H": rt.optical_invariant,
        "S1": sd.s1, "S2": sd.s2, "S3": sd.s3, "S4": sd.s4, "S5": sd.s5,
        "LCH": sd.lch, "TCH": sd.tch,
        "image_height": _image_height(spec, rt, finite),
        "distortion_pct": _distortion_pct(sd.s5, rt),
        "n_surfaces": len(design.radius) - 2,
        "n_elements": _count_elements(design),
        "merit": opt.best if opt.best else 0.0,
    }

    seq_text = codevexport.seq_string(
        design, epd=spec.epd, field_angle_deg=spec.half_field_deg,
        na_obj=spec.na, obj_height=spec.object_height, finite=finite)

    warnings = []
    seq_valid = True
    if validate_seq:
        r = seqparse.validate(seq_text, expected_efl=spec.efl if not finite else None)
        seq_valid = r.ok
        warnings.extend(r.warnings)

    return Result(spec=spec, design=design, prototype_id=pid,
                  prototype_name=proto.name, perf=perf, seq_text=seq_text,
                  warnings=warnings, seq_valid=seq_valid)


def _count_elements(design):
    n = 0
    prev = False
    for g in design.glass:
        is_glass = g and g.strip() != ""
        if is_glass and not prev:
            n += 1
        prev = is_glass
    return n


def _image_height(spec, rt, finite):
    if finite:
        return spec.object_height
    return abs(spec.efl * math.tan(math.radians(spec.half_field_deg)))


def _distortion_pct(s5, rt):
    """Rough third-order distortion estimate (%)."""
    if abs(rt.optical_invariant) < 1e-12 or abs(rt.efl) < 1e-9:
        return 0.0
    return 100.0 * abs(s5) / (abs(rt.optical_invariant) * abs(rt.efl))
