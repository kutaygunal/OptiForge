"""Human-readable design report, lens drawings and the summary table."""
from __future__ import annotations

import io
import math
import os

import numpy as np

from . import optics
from .design import Design

# Ray-bundle colours per relative field (axis, zone, full field)
FIELD_COLORS = ("#1f77b4", "#2ca02c", "#d62728", "#9467bd")
PUPIL_ZONES = (-1.0, -0.5, 0.0, 0.5, 1.0)


def _sag(c, y):
    """Sag (axial deviation) of a spherical surface at height y."""
    y = np.asarray(y, dtype=float)
    if abs(c) < 1e-12:
        return np.zeros_like(y)
    z = c * y ** 2 / (1.0 + np.sqrt(np.maximum(1e-12, 1.0 - c ** 2 * y ** 2)))
    return z


def _vertices(design: Design, n: int):
    """Axial position of every surface vertex (object gap ignored)."""
    zr = [0.0] * (n + 2)
    for i in range(2, n + 2):
        zr[i] = zr[i - 1] + design.thick[i - 1]
    return zr


def _draw_rays(ax, design, rt, zr, n, fields=(0.0, 0.7, 1.0), lw=0.6):
    """Draw paraxial ray bundles for a few relative field heights.

    Paraxial optics is linear, so every ray is a combination of the marginal
    and chief rays: y = pupil_zone * y_marginal + field * y_chief.  These are
    paraxial rays - they show where the light goes, not the real aberrated
    intercepts.
    """
    zimg = zr[n] + design.thick[n]
    for fi, f in enumerate(fields):
        color = FIELD_COLORS[fi % len(FIELD_COLORS)]
        for zone in PUPIL_ZONES:
            zs, ys = [], []
            for i in range(1, n + 1):
                zs.append(zr[i])
                ys.append(zone * rt.y[i] + f * rt.y_b[i])
            # extend to the image plane with the last slope
            if len(zs) >= 2 and abs(zimg - zs[-1]) > 0:
                nu = zone * rt.V_m[-1] + f * rt.V_c[-1]
                zs.append(zimg)
                ys.append(ys[-1] + nu * (zimg - zr[n]))
            ax.plot(zs, ys, color=color, lw=lw, alpha=0.75, zorder=1)


def _render(design: Design, epd: float, field_angle_deg: float = 0.0,
            width=8.0, dpi=110, thumb: bool = False, rays: bool = True):
    """Build the matplotlib figure for a lens cross-section."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rt = optics.trace(design, epd=epd, field_angle_deg=field_angle_deg)
    ym = rt.y
    n = len(design.radius) - 2          # real surfaces
    zr = _vertices(design, n)

    fig, ax = plt.subplots(figsize=(width, width * 0.5), dpi=dpi)
    ax.set_facecolor("white")
    ax.axis("off")

    hmax = 0.0
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
                    edgecolor="#24406b", lw=0.7 if thumb else 1.0, zorder=2)
            ax.plot(left, ylin, color="#24406b", lw=0.7 if thumb else 1.0, zorder=3)
            ax.plot(right, ylin, color="#24406b", lw=0.7 if thumb else 1.0, zorder=3)
            if not thumb:
                ax.text((z1 + z2) / 2, h * 1.15, g, ha="center", fontsize=8,
                        color="#24406b")
            i = i2
        else:
            i += 1

    if hmax <= 0:
        hmax = max(abs(epd) * 0.5, 1.0)

    if rays:
        _draw_rays(ax, design, rt, zr, n, lw=0.5 if thumb else 0.7)

    zimg = zr[n] + design.thick[n]
    if not thumb:
        if 1 <= design.stop <= n + 1:
            ax.axvline(zr[design.stop], color="red", lw=1.0, ls="--", alpha=0.8)
            ax.text(zr[design.stop], hmax * 1.15, "STOP", ha="center",
                    fontsize=7, color="red")
        ax.text(zimg, hmax * 1.15, "Image", ha="center", fontsize=9, color="#333")
        ax.set_title(design.title or "Lens layout", fontsize=13)
    ax.axvline(zimg, color="#333", lw=1.0 if thumb else 1.4, zorder=4)
    ax.plot([zr[1], zimg], [0, 0], color="#bbb", lw=0.6, zorder=0)

    span = max(zimg - zr[1], 1e-6)
    pad = 0.03 * span
    ax.set_xlim(zr[1] - pad, zimg + pad)
    ax.set_ylim(-hmax * 1.45, hmax * 1.45)
    ax.set_aspect("equal")
    return fig


def draw_layout(design: Design, epd: float, field_angle_deg: float = 0.0,
                output: str = "lens_layout.png", width=8.0, dpi=110) -> str:
    """Render a schematic cross-section of the lens and save to `output`."""
    import matplotlib.pyplot as plt
    fig = _render(design, epd, field_angle_deg, width=width, dpi=dpi)
    out = os.path.abspath(output)
    d = os.path.dirname(out)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return out


def layout_png(design: Design, epd: float, field_angle_deg: float = 0.0,
               width=8.0, dpi=110, thumb: bool = False) -> bytes:
    """Render a lens cross-section straight to PNG bytes (no temp file)."""
    import matplotlib.pyplot as plt
    fig = _render(design, epd, field_angle_deg, width=width, dpi=dpi,
                  thumb=thumb)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi,
                transparent=False)
    plt.close(fig)
    return buf.getvalue()


def thumbnail_png(design: Design, epd: float, field_angle_deg: float = 0.0) -> bytes:
    """Small summary-table thumbnail of the system."""
    return layout_png(design, epd, field_angle_deg, width=2.6, dpi=100,
                      thumb=True)


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
    L.append(f"OptiForge  -  {result.prototype_name}")
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
    L.append(f".seq round-trip: {'OK' if result.seq_valid else 'FAILED'}")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Summary table (the OptiForge result view, as text)
# ---------------------------------------------------------------------------
SUMMARY_COLUMNS = [
    ("name",            "Name",              12, None),
    ("epd",             "Ent.Pupil",          9, "{:.4f}"),
    ("semi_field",      "SemiFld",            8, "{:.4f}"),
    ("efl",             "EFL",                9, "{:.4f}"),
    ("package_length",  "Package",           10, "{:.4f}"),
    ("image_clearance", "ImgClear",          10, "{:.4f}"),
    ("cra_deg",         "ChiefRay",          10, "{:.4f}"),
    ("distortion_pct",  "Distort%",          10, "{:.4f}"),
    ("rel_illum_pct",   "RelIllum%",         10, "{:.4f}"),
    ("avg_spot_diam",   "AvgSpot",           10, "{:.4f}"),
    ("elem_count",      "Elems",              6, "{:d}"),
]


def summary_table(sysset) -> str:
    """Render a SystemSet as the OptiForge summary table (text)."""
    spec = sysset.spec
    g = sysset.goals
    lines = []
    lines.append("OptiForge Summary")
    lines.append("=" * 108)

    header = "".join(lab.rjust(w) for _, lab, w, _ in SUMMARY_COLUMNS)
    lines.append(header)
    lines.append("-" * len(header))

    # goals row
    goal_row = {
        "name": "Goals",
        "epd": g["epd"], "semi_field": g["semi_field_deg"], "efl": g["efl"],
        "package_length": g["package_length"],
        "image_clearance": g["image_clearance"],
        "cra_deg": g["cra_deg"],
        "distortion_pct": g["distortion_pct"],
        "rel_illum_pct": None,
        "avg_spot_diam": None,
        "elem_count": None,
    }
    cells = []
    for key, _lab, w, fmt in SUMMARY_COLUMNS:
        v = goal_row.get(key)
        if key == "elem_count":
            lo, hi = g.get("elem_min"), g.get("elem_max")
            s = f"[{lo},{hi}]" if (lo or hi) else "-"
        elif v is None:
            s = "-"
        elif isinstance(v, str):
            s = v
        else:
            s = fmt.format(v) if fmt and fmt != "{:d}" else str(v)
        cells.append(s.rjust(w))
    lines.append("".join(cells))
    lines.append("-" * len(header))

    for s in sysset.systems:
        m = s.metrics
        row = {
            "name": s.name, "epd": m.epd, "semi_field": spec.half_field_deg,
            "efl": m.efl, "package_length": m.package_length,
            "image_clearance": m.image_clearance, "cra_deg": m.cra_deg,
            "distortion_pct": m.distortion_pct,
            "rel_illum_pct": m.rel_illum_pct,
            "avg_spot_diam": m.avg_spot_diam, "elem_count": m.elem_count,
        }
        cells = []
        for key, _lab, w, fmt in SUMMARY_COLUMNS:
            v = row[key]
            if isinstance(v, str):
                s_ = v
            elif fmt == "{:d}":
                s_ = str(int(v))
            else:
                s_ = fmt.format(v)
            cells.append(s_.rjust(w))
        flag = "" if s.meets_goals else "  *"
        lines.append("".join(cells) + flag)

    lines.append("-" * len(header))
    lines.append(f"{len(sysset.systems)} systems from {sysset.considered} "
                 f"candidates.  * = one or more goals not met.")
    return "\n".join(lines)
