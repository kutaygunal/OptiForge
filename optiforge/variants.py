"""Structural transforms that turn one classical form into many candidates.

OptiForge does not return a single answer: it returns a *population* of
viable starting points that trade the goals off against each other in
different ways.  This module supplies the structural moves that build that
population from the catalog forms:

    add_rear_element     field flattener near the image (+1 element)
    add_front_element    weak meniscus corrector at the front (+1 element)
    split_doublet        break a cemented interface into an air space
    remove_element       drop the weakest singlet (-1 element)
    swap_glasses         substitute crowns / flints from the catalog
    perturb              randomised curvature + airspace kick (new local minimum)

Each move returns a NEW Design (the input is never mutated) or None when it
does not apply.  Element count is what the "Number of Elements" goal filters
on, so every move reports how it changes that count.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from . import glass as glasslib
from .design import Design


@dataclass
class Move:
    """One structural transform, with the element-count delta it produces."""
    name: str          # short tag used in the system description
    fn: Callable       # (Design, efl) -> Optional[Design]
    d_elements: int    # change in element count


def _n_real(d: Design) -> int:
    return len(d.radius) - 2


def _is_glass(code) -> bool:
    return bool(code) and bool(str(code).strip())


# ---------------------------------------------------------------------------
# Element insertion
# ---------------------------------------------------------------------------
def add_rear_element(d: Design, efl: float,
                     code: str = "NSF10_SCHOTT") -> Optional[Design]:
    """Insert a negative field flattener just in front of the image plane.

    A thick, strongly curved negative element close to the image bends the
    Petzval sum without disturbing the aperture-dependent aberrations (the
    marginal ray is nearly at the axis there), so it buys field flatness for
    one extra element.
    """
    n = _n_real(d)
    if n < 2:
        return None
    bfl = d.thick[n]
    t_el = max(0.02 * efl, 0.4)
    if bfl < 3.0 * t_el:
        return None
    air1 = 0.55 * bfl
    air2 = bfl - air1 - t_el
    if air2 < 0.15 * bfl:
        return None
    # plano / convex-to-image negative meniscus: R1 = INF, R2 > 0
    r1 = float("inf")
    r2 = 0.55 * efl
    out = d.copy()
    out.radius = d.radius[:n + 1] + [r1, r2] + d.radius[n + 1:]
    out.thick = d.thick[:n] + [air1, t_el, air2] + d.thick[n + 1:]
    out.glass = d.glass[:n] + ["", code, ""] + d.glass[n + 1:]
    return out


def add_front_element(d: Design, efl: float,
                      code: str = "NBK7_SCHOTT") -> Optional[Design]:
    """Add a weak positive meniscus ahead of the first surface.

    An extra front element takes some of the ray bending off the original
    group, which lowers the surface incidence angles and therefore the
    higher-order residuals - at the cost of package length.
    """
    n = _n_real(d)
    if n < 2:
        return None
    t_el = max(0.035 * efl, 0.5)
    t_air = max(0.02 * efl, 0.3)
    r1 = 0.85 * efl
    r2 = 1.35 * efl
    out = d.copy()
    out.radius = d.radius[:1] + [r1, r2] + d.radius[1:]
    out.thick = d.thick[:1] + [t_el, t_air] + d.thick[1:]
    out.glass = d.glass[:1] + [code, ""] + d.glass[1:]
    out.stop = d.stop + 2
    return out


# ---------------------------------------------------------------------------
# Element restructuring
# ---------------------------------------------------------------------------
def split_doublet(d: Design, efl: float) -> Optional[Design]:
    """Break the first cemented interface into a narrow air space.

    Same element count, two more degrees of freedom (the two faces are no
    longer forced to share a radius), which usually buys spherical / coma
    balance at the cost of an extra air-glass boundary.
    """
    n = _n_real(d)
    for k in range(1, n):
        g1 = d.glass[k] if k < len(d.glass) else ""
        g2 = d.glass[k + 1] if k + 1 < len(d.glass) else ""
        if _is_glass(g1) and _is_glass(g2) and str(g1).strip() != str(g2).strip():
            gap = max(0.004 * efl, 0.05)
            out = d.copy()
            out.radius = d.radius[:k + 2] + [d.radius[k + 1]] + d.radius[k + 2:]
            out.thick = d.thick[:k + 1] + [gap] + d.thick[k + 1:]
            out.glass = d.glass[:k + 1] + [""] + d.glass[k + 1:]
            if d.stop > k + 1:
                out.stop = d.stop + 1
            return out
    return None


def _element_runs(d: Design) -> List[tuple]:
    """Return [(first_surface, last_surface, code), ...] for each element."""
    runs = []
    n = _n_real(d)
    k = 1
    while k <= n:
        code = d.glass[k] if k < len(d.glass) else ""
        if _is_glass(code):
            k2 = k
            while k2 + 1 <= n and str(d.glass[k2 + 1]).strip() == str(code).strip():
                k2 += 1
            runs.append((k, k2 + 1, str(code).strip()))
            k = k2 + 1
        else:
            k += 1
    return runs


def _air_spaced_groups(d: Design) -> List[tuple]:
    """Groups of surfaces bounded by air on both sides, with their power.

    A group is one air-spaced element or one cemented set - i.e. exactly what
    can be lifted out of the stack without leaving a glass-to-glass interface
    hanging.  Returns [(first_surface, last_surface, n_elements, |power|)].
    """
    n = _n_real(d)
    groups = []
    k = 1
    while k <= n:
        if not _is_glass(d.glass[k] if k < len(d.glass) else ""):
            k += 1
            continue
        k1 = k
        n_el = 0
        prev = ""
        while k <= n and _is_glass(d.glass[k] if k < len(d.glass) else ""):
            code = str(d.glass[k]).strip()
            if code != prev:
                n_el += 1
            prev = code
            k += 1
        k2 = k                      # exit surface (its medium is air)
        # thin-lens power of the whole group
        phi = 0.0
        for i in range(k1, k2):
            code = str(d.glass[i]).strip()
            try:
                ng = glasslib.index(code)
            except KeyError:
                continue
            c1 = 1.0 / d.radius[i] if np.isfinite(d.radius[i]) and d.radius[i] else 0.0
            c2 = (1.0 / d.radius[i + 1]
                  if np.isfinite(d.radius[i + 1]) and d.radius[i + 1] else 0.0)
            phi += (ng - 1.0) * (c1 - c2)
        groups.append((k1, k2, n_el, abs(phi)))
    return groups


def remove_element(d: Design, efl: float) -> Optional[Design]:
    """Remove the group doing the least first-order work.

    Picks the air-spaced element (or cemented set) whose surface powers most
    nearly cancel and merges the surrounding air spaces, so the rest of the
    prescription keeps its spacing.  A simpler lens is often the better
    starting point when the goals are loose, and it is the only way to reach
    the low end of a "number of elements" goal.
    """
    n = _n_real(d)
    groups = [g for g in _air_spaced_groups(d)]
    if len(groups) < 2:             # never dissolve the whole lens
        return None
    k1, k2, _n_el, _phi = min(groups, key=lambda g: g[3])
    span = k2 - k1 + 1              # surfaces removed (k1 .. k2)

    out = d.copy()
    out.radius = d.radius[:k1] + d.radius[k2 + 1:]
    if k1 == 1:
        # leading group: the object gap must survive untouched
        out.thick = [d.thick[0]] + d.thick[k2 + 1:]
        out.glass = [""] + d.glass[k2 + 1:]
    else:
        merged = sum(d.thick[k1 - 1:k2 + 1])
        out.thick = d.thick[:k1 - 1] + [merged] + d.thick[k2 + 1:]
        out.glass = d.glass[:k1 - 1] + [""] + d.glass[k2 + 1:]

    if d.stop > k2:
        out.stop = d.stop - span
    elif k1 <= d.stop <= k2:
        out.stop = max(1, k1 - 1)
    out.stop = max(1, min(out.stop, len(out.radius) - 2))
    if len(out.radius) < 4:
        return None
    return out


# ---------------------------------------------------------------------------
# Glass substitution
# ---------------------------------------------------------------------------
def swap_glasses(d: Design, efl: float, shift: int = 1) -> Optional[Design]:
    """Substitute every glass with its neighbour in the crown / flint lists.

    Moving up the crown list raises the index (flatter surfaces for the same
    power, less spherical aberration); moving along the flint list changes the
    dispersion available for the colour correction.
    """
    out = d.copy()
    changed = False
    for i, code in enumerate(out.glass):
        c = str(code or "").strip().upper()
        if not c:
            continue
        for pool in (glasslib.CROWN, glasslib.FLINT):
            if c in pool:
                j = (pool.index(c) + shift) % len(pool)
                if pool[j] != c:
                    out.glass[i] = pool[j]
                    changed = True
                break
        else:
            # not in either pool: map by dispersion to the nearest pool entry
            try:
                vd = glasslib.abbe(c)
            except KeyError:
                continue
            pool = glasslib.CROWN if vd >= 45 else glasslib.FLINT
            j = shift % len(pool)
            if pool[j] != c:
                out.glass[i] = pool[j]
                changed = True
    return out if changed else None


# ---------------------------------------------------------------------------
# Randomised restart
# ---------------------------------------------------------------------------
def perturb(d: Design, efl: float, seed: int = 0,
            curv_amp: float = 0.10, gap_amp: float = 0.12) -> Optional[Design]:
    """Kick curvatures and air spaces to land the optimizer elsewhere.

    The merit surface of a lens is riddled with local minima; restarting from
    a perturbed prescription is the standard way to reach a genuinely
    different solution rather than a cosmetic variation of the same one.
    """
    rng = np.random.default_rng(seed)
    n = _n_real(d)
    out = d.copy()
    for i in range(1, n + 1):
        r = out.radius[i]
        if r and np.isfinite(r):
            c = 1.0 / r
            c *= 1.0 + curv_amp * float(rng.normal())
            out.radius[i] = 1.0 / c if abs(c) > 1e-12 else float("inf")
    for i in range(1, len(out.thick) - 1):
        code = out.glass[i] if i < len(out.glass) else ""
        if _is_glass(code):
            continue                     # keep element thicknesses sane
        t = out.thick[i] * (1.0 + gap_amp * float(rng.normal()))
        out.thick[i] = max(t, 0.004 * efl)
    return out


# ---------------------------------------------------------------------------
# The move set used by the generator
# ---------------------------------------------------------------------------
def base_moves() -> List[Move]:
    """Structural moves, in the order the generator tries them."""
    return [
        Move("as-drawn", lambda d, f: d.copy(), 0),
        Move("split doublet", split_doublet, 0),
        Move("high-index glass", lambda d, f: swap_glasses(d, f, 1), 0),
        Move("field flattener", add_rear_element, +1),
        Move("front corrector", add_front_element, +1),
        Move("reduced element", remove_element, -1),
        Move("alt. glass", lambda d, f: swap_glasses(d, f, 2), 0),
    ]


def apply_moves(d: Design, efl: float, moves: List[Move]) -> Optional[Design]:
    """Apply a chain of moves; returns None if any of them does not apply."""
    cur = d
    for mv in moves:
        cur = mv.fn(cur, efl)
        if cur is None:
            return None
    return cur
