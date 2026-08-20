"""Generate a sample starting-point .seq for every lens form.

Run:  python examples/generate_samples.py
Writes .seq files + layout images + a text report into examples/.
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optiforge import generate, Spec, report

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

CASES = [
    ("double_gauss", 50.0, 40.0, "fno", 2.8),
    ("cooke_triplet", 50.0, 35.0, "fno", 4.0),
    ("telephoto", 120.0, 20.0, "fno", 4.0),
    ("retrofocus", 20.0, 80.0, "fno", 4.0),
    ("petzval", 50.0, 15.0, "fno", 2.0),
    ("collimator", 50.0, 5.0, "fno", 2.0),
]

if __name__ == "__main__":
    for lt, efl, fov, ap, av in CASES:
        s = Spec(efl=efl, fov_deg=fov, aperture=ap, aperture_value=av, lens_type=lt)
        res = generate(s)
        base = os.path.join(OUT, lt)
        with open(base + ".seq", "w", encoding="utf-8") as f:
            f.write(res.seq_text + "\n")
        report.draw_layout(res.design, s.epd, s.half_field_deg, base + "_layout.png")
        with open(base + ".txt", "w", encoding="utf-8") as f:
            f.write(report.text_report(s, res))
        print(f"{lt:14s} EFL={res.perf['efl']:8.3f}  F#={res.perf['fno']:6.3f}  "
              f"BFL={res.perf['bfl']:8.2f}  valid={res.seq_valid}  -> {lt}.seq")
    print("\nWrote samples to", os.path.abspath(OUT))
