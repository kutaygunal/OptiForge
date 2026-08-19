"""AI Start Expert — local web application (Flask).

Run entirely on the operator's machine (on-premise, no internet needed):
    python -m aistart.webapp            # serves http://127.0.0.1:5000
"""
from __future__ import annotations

import base64
import io
import os
import tempfile

from flask import Flask, jsonify, render_template, request

from . import generate, report
from .specs import Spec

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.get_json(force=True) or {}
    try:
        spec = Spec(
            efl=float(data.get("efl", 50.0)),
            fov_deg=float(data.get("fov_deg", 40.0)),
            aperture=data.get("aperture", "fno"),
            aperture_value=float(data.get("aperture_value", 2.8)),
            lens_type=data.get("lens_type", "auto"),
            distortion_max=float(data.get("distortion_max", 3.0)),
            min_image_clearance=float(data.get("clearance", 0.0)),
            object_distance=float(data.get("object_distance", 0.0)) or float("inf"),
            object_height=float(data.get("object_height", 0.0)),
        )
    except (TypeError, ValueError) as e:
        return jsonify({"ok": False, "error": f"invalid input: {e}"}), 400

    try:
        res = generate(spec)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    # render layout into bytes
    with tempfile.TemporaryDirectory() as td:
        img_path = os.path.join(td, "layout.png")
        report.draw_layout(res.design, spec.epd, spec.half_field_deg, img_path)
        with open(img_path, "rb") as f:
            layout_b64 = base64.b64encode(f.read()).decode()

    p = res.perf
    payload = {
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
        "layout": "data:image/png;base64," + layout_b64,
        "seq": res.seq_text,
        "seq_valid": res.seq_valid,
        "warnings": res.warnings,
    }
    return jsonify(payload)


def main():
    print("AI Start Expert  (on-premise)  ->  http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
