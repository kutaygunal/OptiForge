"""Human-readable design report + matplotlib lens drawing."""
from __future__ import annotations

import math
import os

import numpy as np

from . import optics
from .design import Design


def _sag(c, y):
    """Sag (axial deviation) of a spherical surface at height y."""
    y = np.asarray(y, dtype=float)
    if abs(c) < 1e-12:
        return np.zeros_like(y)
    z = c * y ** 2 / (1.0 + np.sqrt(np.maximum(1e-12, 1.0 - c ** 2 * y ** 2)))
    return z


def draw_layout(design: Design, epd: float, field_angle_deg: float = 0.0,
                output: str = "lens_layout.png", width=8.0, dpi=110) -> str:
    """Render a schematic cross-section of the lens and save to `output`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rt = optics.trace(design, epd=epd, field_angle_deg=field_angle_deg)
    ym = rt.y                     # marginal heights per surface
    n = len(design.radius) - 2    # real surfaces

    # axial positions of the real surface vertices (object gap ignored)
    zr = [0.0] * (n + 2)
    for i in range(2, n + 2):
        zr[i] = zr[i - 1] + design.thick[i - 1]

    real_radius = [design.radius[i] for i in range(1, n + 1)]

    fig, ax = plt.subplots(figsize=(width, width * 0.5), dpi=dpi)
    ax.set_facecolor("white")
    ax.axis("off")

    hmax = 0.0
    # draw each element (glass body between two consecutive surfaces)
    i = 1
    while i <= n:
        g = design.glass[i] if i < len(design.glass) else ""
        if g and g.strip():
            i2 = i + 1
            if i2 > n + 1:
                break
            R1 = design.radius[i]
            R2 = design.radius[i2] if i2 <= n else design.radius[n]
            c1 = 1.0 / R1 if R1 and math.isfinite(R1) else 0.0
            c2 = 1.0 / R2 if R2 and math.isfinite(R2) else 0.0
            z1, z2 = zr[i], zr[i2]
            h = max(abs(ym[i]), abs(ym[i2]), 0.2) * 1.12
            hmax = max(hmax, h)
            ylin = np.linspace(-h, h, 200)
            left = z1 + _sag(c1, ylin)
            right = z2 + _sag(c2, ylin)
            poly = np.column_stack([np.r_[left, right[::-1]],
                                    np.r_[ylin, ylin[::-1]]])
            ax.fill(poly[:, 0], poly[:, 1], color="#cfe0f0",
                    edgecolor="#24406b", lw=1.0)
            ax.plot(left, ylin, color="#24406b", lw=1.0)
            ax.plot(right, ylin, color="#24406b", lw=1.0)
            ax.text((z1 + z2) / 2, h * 1.15, g, ha="center", fontsize=8,
                    color="#24406b")
            i = i2
        else:
            i += 1

    # stop
    if 1 <= design.stop <= n + 1:
        ax.axvline(zr[design.stop], color="red", lw=1.0, ls="--", alpha=0.8)
        ax.text(zr[design.stop], hmax * 1.15, "STOP", ha="center", fontsize=7,
                color="red")

    # image plane
    zimg = zr[n] + design.thick[n]
    ax.axvline(zimg, color="#333", lw=1.4)
    ax.text(zimg, hmax * 1.15, "Image", ha="center", fontsize=9, color="#333")

    ax.plot([zr[1], zimg + 2], [0, 0], color="#bbb", lw=0.8)
    ax.set_xlim(zr[1] - 1, zimg + 3)
    ax.set_ylim(-hmax * 1.7, hmax * 1.7)
    ax.set_aspect("equal")
    ax.set_title(design.title or "Lens layout", fontsize=13)
    out = os.path.abspath(output)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------
def _lbl(v) -> str:
    return f"{v:+.3e}"


def _QS(v):
    a = abs(v)
    if a < 0.1:
        return "{:.2e}  (well corrected)".format(v)
    if a < 1.0:
        return "{:.2e}  (small)".format(v)
    if a < 5.0:
        return "{:.2e}  (moderate)".format(v)
    return "{:.2e}  (large - refine)".format(v)


def text_report(spec, result) -> str:
    p = result.perf
    L = []
    L.append(f"AI Start Expert  -  {result.prototype_name}")
    L.append("=" * 52)
    L.append(f"Spec: EFL={spec.efl:g} mm  FOV={spec.fov_deg:g} deg  "
             f"{spec.aperture_kind_label}={spec.aperture_value:g}")
    L.append("")
    L.append("First-order:")
    L.append(f"  EFL             {p['efl']:.4f} mm")
    L.append(f"  F/#             {p['fno']:.4f}")
    L.append(f"  Entrance pupil  {p['epd']:.4f} mm")
    L.append(f"  Back focal      {p['bfl']:.4f} mm")
    L.append(f"  Image height    {p['image_height']:.4f} mm")
    L.append(f"  Elements        {p['n_elements']}")
    L.append("")
    L.append("Third-order (Seidel) coefficients:")
    for key, lab in [("S1", "spherical"), ("S2", "coma"), ("S3", "astigmatism"),
                     ("S4", "field curvature"), ("S5", "distortion")]:
        L.append(f"  {key} {lab:<16s} {_QS(p[key])}")
    L.append(f"  Axial color        {_QS(p['LCH'])}")
    L.append(f"  Lateral color      {_QS(p['TCH'])}")
    L.append("")
    L.append(f"Third-order distortion ~ {p['distortion_pct']:.2f} %")
    L.append(f"CODE V .seq round-trip: {'OK' if result.seq_valid else 'FAILED'}")
    return "\n".join(L)
