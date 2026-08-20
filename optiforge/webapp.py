"""OptiForge - local web application (Flask).

Run entirely on the operator's machine (on-premise, no internet needed):
    python -m optiforge.webapp            # serves http://127.0.0.1:5000

The UI mirrors the OptiForge workflow: a specification dialog of
first-order targets plus optional goals, and a summary table of the generated
starting points that can be sorted, inspected and exported as lens sequence
(.seq) files.
"""
from __future__ import annotations

import base64
import io
import math
import os
import zipfile

from flask import Flask, Response, jsonify, render_template, request

from . import generate, generator, metrics, report
from .specs import LENS_TYPE_LABELS, UNITS, Spec

app = Flask(__name__)

# The most recent run, kept in memory so the detail / download endpoints do not
# have to regenerate anything.  Single local user, single run at a time.
_LAST: dict = {"set": None}


def _b64png(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode()


def _f(data, key, default=0.0):
    v = data.get(key, default)
    if v is None or v == "":
        return default
    return float(v)


def _b(data, key, default=False):
    v = data.get(key, default)
    if isinstance(v, str):
        return v.lower() in ("1", "true", "yes", "on")
    return bool(v)


def _spec_from_json(data: dict) -> Spec:
    """Build a Spec from the dialog payload."""
    obj_dist = _f(data, "object_distance", 0.0)
    spec = Spec(
        efl=_f(data, "efl", 50.0),
        semi_field_deg=_f(data, "semi_field_deg", 20.0),
        aperture=data.get("aperture", "fno"),
        aperture_value=_f(data, "aperture_value", 2.8),
        lens_type=data.get("lens_type", "auto"),
        units=data.get("units", "mm"),
        wl_short=_f(data, "wl_short", 486.1),
        wl_long=_f(data, "wl_long", 656.3),
        use_package_length=_b(data, "use_package_length"),
        package_length_max=_f(data, "package_length_max", 0.0),
        use_image_clearance=_b(data, "use_image_clearance"),
        min_image_clearance=_f(data, "min_image_clearance", 0.0),
        use_distortion=_b(data, "use_distortion"),
        distortion_max=_f(data, "distortion_max", 3.0),
        use_cra=_b(data, "use_cra"),
        cra_target=_f(data, "cra_target", 0.0),
        cra_tolerance=_f(data, "cra_tolerance", 10.0),
        use_elem_min=_b(data, "use_elem_min"),
        elem_min=int(_f(data, "elem_min", 3)),
        use_elem_max=_b(data, "use_elem_max"),
        elem_max=int(_f(data, "elem_max", 8)),
        object_distance=obj_dist if obj_dist > 0 else float("inf"),
        object_height=_f(data, "object_height", 0.0),
        n_systems=int(_f(data, "n_systems", 10)),
        base_name=(data.get("base_name") or "Lens").strip() or "Lens",
    )
    return spec


def _row(sysm) -> dict:
    """One summary-table row."""
    m = sysm.metrics
    thumb = report.thumbnail_png(sysm.design, sysm.metrics.epd,
                                 _LAST["spec"].half_field_deg)
    return {
        "index": sysm.index,
        "name": sysm.name,
        "description": sysm.description,
        "prototype": sysm.prototype_name,
        "thumb": _b64png(thumb),
        "metrics": {k: _clean(v) for k, v in m.as_row().items()},
        "semi_field_deg": _LAST["spec"].half_field_deg,
        "wl_short": _LAST["spec"].wl_short,
        "wl_long": _LAST["spec"].wl_long,
        "goal_status": {k: {"met": v["met"], "kind": v["kind"]}
                        for k, v in sysm.goal_status.items()},
        "meets_goals": sysm.meets_goals,
        "seq_valid": sysm.seq_valid,
        "score": _clean(sysm.score),
    }


def _clean(v):
    """JSON-safe number."""
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if math.isnan(f) or math.isinf(f):
        return None
    return f


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html",
                           lens_types=LENS_TYPE_LABELS,
                           units={k: v[0] for k, v in UNITS.items()})


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
@app.route("/api/generate_systems", methods=["POST"])
def api_generate_systems():
    """Run the search and return the summary table."""
    data = request.get_json(force=True) or {}
    try:
        spec = _spec_from_json(data)
    except (TypeError, ValueError) as e:
        return jsonify({"ok": False, "error": f"invalid input: {e}"}), 400

    try:
        sysset = generator.generate_systems(spec)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    _LAST["set"] = sysset
    _LAST["spec"] = sysset.spec

    if not sysset.systems:
        return jsonify({"ok": False,
                        "error": "no viable system met the specification - "
                                 "try relaxing a goal or widening the element "
                                 "count range"}), 400

    return jsonify({
        "ok": True,
        "goals": {k: _clean(v) for k, v in sysset.goals.items()},
        "unit_label": sysset.spec.unit_label,
        "units": sysset.spec.units,
        "rows": [_row(s) for s in sysset.systems],
        "considered": sysset.considered,
        "elapsed": round(sysset.elapsed, 2),
    })


@app.route("/api/detail/<int:index>")
def api_detail(index: int):
    """Full detail for one generated system."""
    sysset = _LAST.get("set")
    if not sysset:
        return jsonify({"ok": False, "error": "nothing generated yet"}), 404
    match = [s for s in sysset.systems if s.index == index]
    if not match:
        return jsonify({"ok": False, "error": f"no system {index}"}), 404
    s = match[0]
    spec = sysset.spec
    m = s.metrics
    layout = report.layout_png(s.design, m.epd, spec.half_field_deg,
                               width=8.0, dpi=110)

    surfaces = []
    n = len(s.design.radius) - 2
    for k in range(1, n + 1):
        r = s.design.radius[k]
        surfaces.append({
            "i": k,
            "radius": None if not math.isfinite(r) else round(r, 6),
            "thick": round(s.design.thick[k], 6),
            "glass": (s.design.glass[k] or "").strip() or "air",
            "stop": k == s.design.stop,
        })

    sd = m.seidel
    return jsonify({
        "ok": True,
        "name": s.name,
        "description": s.description,
        "prototype": s.prototype_name,
        "layout": _b64png(layout),
        "seq": s.seq_text,
        "seq_valid": s.seq_valid,
        "warnings": s.warnings,
        "metrics": {k: _clean(v) for k, v in m.as_row().items()},
        "seidel": {"S1": _clean(sd.s1), "S2": _clean(sd.s2), "S3": _clean(sd.s3),
                   "S4": _clean(sd.s4), "S5": _clean(sd.s5),
                   "LCH": _clean(sd.lch), "TCH": _clean(sd.tch)},
        "surfaces": surfaces,
        "goal_status": {k: {"value": _clean(v["value"]) if not isinstance(v["value"], tuple) else list(v["value"]),
                            "target": list(v["target"]) if isinstance(v["target"], tuple) else _clean(v["target"]),
                            "met": v["met"], "kind": v["kind"]}
                        for k, v in s.goal_status.items()},
    })


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
@app.route("/api/seq/<int:index>")
def api_seq(index: int):
    sysset = _LAST.get("set")
    if not sysset:
        return "nothing generated yet", 404
    match = [s for s in sysset.systems if s.index == index]
    if not match:
        return f"no system {index}", 404
    s = match[0]
    return Response(
        s.seq_text + "\n", mimetype="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{s.filename}"'})


@app.route("/api/seq_all")
def api_seq_all():
    """Every generated system as one .zip of .seq files."""
    sysset = _LAST.get("set")
    if not sysset or not sysset.systems:
        return "nothing generated yet", 404
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for s in sysset.systems:
            z.writestr(s.filename, s.seq_text + "\n")
        z.writestr("summary.txt", report.summary_table(sysset) + "\n")
    buf.seek(0)
    base = sysset.spec.base_name
    return Response(
        buf.getvalue(), mimetype="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{base}_systems.zip"'})


# ---------------------------------------------------------------------------
# Legacy single-design endpoint (kept for the CLI / older callers)
# ---------------------------------------------------------------------------
@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.get_json(force=True) or {}
    try:
        spec = _spec_from_json(data)
        if "fov_deg" in data and "semi_field_deg" not in data:
            spec.fov_deg = float(data["fov_deg"])
            spec.semi_field_deg = None
    except (TypeError, ValueError) as e:
        return jsonify({"ok": False, "error": f"invalid input: {e}"}), 400

    try:
        res = generate(spec)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    layout = report.layout_png(res.design, spec.epd, spec.half_field_deg)
    p = res.perf
    return jsonify({
        "ok": True,
        "prototype": res.prototype_name,
        "first_order": {
            "efl": round(p["efl"], 4), "fno": round(p["fno"], 4),
            "epd": round(p["epd"], 4), "bfl": round(p["bfl"], 4),
            "image_height": round(p["image_height"], 4),
            "elements": p["n_elements"],
            "distortion_pct": round(p["distortion_pct"], 3),
        },
        "seidel": {k: round(p[k], 4) for k in ["S1", "S2", "S3", "S4", "S5",
                                               "LCH", "TCH"]},
        "layout": _b64png(layout),
        "seq": res.seq_text,
        "seq_valid": res.seq_valid,
        "warnings": res.warnings,
    })


def main():
    port = int(os.environ.get("PORT", "5000"))
    print(f"OptiForge  (on-premise)  ->  http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
