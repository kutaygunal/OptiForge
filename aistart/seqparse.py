"""CODE V .seq validator.

Parses a CODE V lens-sequence file with the same rules as the rayoptics
"cmdproc" reader (which reads real CODE V .seq files), so the generator's
output can be round-trip validated.  It reconstructs the Design from the file
and checks it against the in-memory model (geometry, EFL, F/#).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import List

from .design import Design


def _isnum(t):
    try:
        float(t)
        return True
    except (TypeError, ValueError):
        return False


def tokenize(line: str) -> List[str]:
    tkns = re.findall(r"[^'\"]\S*|\".+?\"|'.+?'", line)
    out = []
    for t in tkns:
        if t[:1] in ('"', "'"):
            out.append(t[1:-1])
        else:
            out.append(t)
    return out


@dataclass
class SeqParse:
    ok: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    design: Design = None
    title: str = ""
    epd: float = 0.0
    efl: float = 0.0
    fno: float = 0.0
    na: float = 0.0
    field_angle: float = 0.0
    finite: bool = False
    obj_dist: float = float("inf")
    obj_height: float = 0.0
    stop: int = 1
    wavelengths: List[float] = field(default_factory=list)

    def add_error(self, msg):
        self.ok = False
        self.errors.append(msg)


def parse_seq(text: str) -> SeqParse:
    """Parse CODE V .seq text and reconstruct the Design."""
    r = SeqParse()
    radius: List[float] = [0.0]     # object surface (plane)
    thick: List[float] = []
    glass: List[str] = []
    stop = 0
    real = 0
    obj_dist = float("inf")

    for raw in text.splitlines():
        line = raw.split("!", 1)[0]
        for seg in line.split(";"):
            seg = seg.strip()
            if not seg:
                continue
            toks = tokenize(seg)
            if not toks:
                continue
            cmd = toks[0].upper()
            rest = toks[1:]
            if cmd == "TIT":
                r.title = toks[1] if len(toks) > 1 else ""
            elif cmd == "WL":
                r.wavelengths = [float(t) for t in rest if _isnum(t)]
            elif cmd in ("EPD", "FNO", "NA", "NAO"):
                v = float(rest[0]) if rest and _isnum(rest[0]) else 0.0
                if cmd == "EPD":
                    r.epd = v
                elif cmd == "FNO":
                    r.fno = v
                elif cmd == "NA":
                    r.na = v
                elif cmd == "NAO":
                    r.na = v
                    r.finite = True
            elif cmd == "YAN":
                r.field_angle = max([float(t) for t in rest if _isnum(t)] + [0.0])
            elif cmd == "YOB":
                r.obj_height = max([float(t) for t in rest if _isnum(t)] + [0.0])
                r.finite = True
            elif cmd == "SO":
                if rest and _isnum(rest[-1]):
                    obj_dist = float(rest[-1])
            elif cmd == "SI":
                radius.append(0.0)   # image surface (plane)
            elif cmd == "STO":
                stop = real
            elif cmd == "S":
                if len(rest) >= 2 and _isnum(rest[0]) and _isnum(rest[1]):
                    rdy = float(rest[0])
                    thi = float(rest[1])
                    gla = (rest[2] if len(rest) > 2 else "").strip()
                    radius.append(rdy)
                    thick.append(thi)
                    glass.append(gla)
                    real += 1
                else:
                    r.add_error(f"malformed surface line: {seg!r}")

    # insert the object gap in front; radius already has object(0) + real + image
    thick.insert(0, obj_dist)
    glass.insert(0, "")
    if len(radius) - 1 != len(thick):
        r.add_error(f"surface/thickness mismatch ({len(radius)} surf, {len(thick)} gaps)")

    if stop == 0:
        stop = max(1, real // 2)

    r.design = Design(radius=radius, thick=thick, glass=glass, stop=stop,
                      title=r.title, obj_dist=obj_dist,
                      obj_type="finite" if r.finite else "infinite")
    return r


def validate(text: str, expected_efl: float = None,
             expected_epd: float = None) -> SeqParse:
    """Parse a .seq block and (optionally) verify EFL / EPD from its geometry."""
    from . import optics
    r = parse_seq(text)
    if not r.ok:
        return r
    d = r.design
    epd = r.epd if r.epd else (expected_epd or 10.0)
    try:
        rt = optics.trace(d, epd=epd, field_angle_deg=r.field_angle,
                          na_obj=r.na, obj_height=r.obj_height,
                          finite=r.finite)
        d.efl = rt.efl
        d.fno = rt.fno
        r.efl = rt.efl
        r.fno = rt.fno
        if expected_efl:
            if abs(rt.efl - expected_efl) > max(1e-3, 0.01 * abs(expected_efl)):
                r.add_error(f"EFL mismatch: parsed {rt.efl:.4f} != expected {expected_efl:.4f}")
    except Exception as e:  # noqa
        r.add_error(f"trace of parsed design failed: {e}")
    return r


# delayed import to avoid a cycle at module load
def _lazy():
    from . import optics
    return optics
