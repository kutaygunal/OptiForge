"""Command-line interface for AI Start Expert.

Usage examples:
    python -m aistart --efl 50 --fov 40 --fno 2.8 --type double_gauss --out out.seq
    python -m aistart --efl 120 --fov 20 --fno 4 --type telephoto --draw
    python -m aistart --efl 50 --fov 5 --na 0.5 --type microscope --obj-dist 1.0 --obj-height 0.1
"""
from __future__ import annotations

import argparse
import os
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aistart",
                                description="AI Start Expert - local starting-point "
                                            "lens design for CODE V")
    p.add_argument("--efl", type=float, default=50.0, help="effective focal length [mm]")
    p.add_argument("--fov", type=float, default=40.0, help="full field of view [deg]")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--fno", type=float, default=None, help="target f-number")
    g.add_argument("--epd", type=float, default=None, help="entrance pupil diameter [mm]")
    g.add_argument("--naf", type=float, default=None, help="numerical aperture")
    p.add_argument("--type", dest="lens_type", default="auto",
                   help="lens form: auto|double_gauss|cooke_triplet|telephoto|"
                        "retrofocus|petzval|microscope|collimator")
    p.add_argument("--dist", type=float, default=3.0,
                   help="max allowed distortion (percent, informational)")
    p.add_argument("--clearance", type=float, default=0.0,
                   help="minimum image clearance / back focal [mm]")
    p.add_argument("--obj-dist", type=float, default=0.0,
                   help="finite-conjugate object distance [mm] (microscope)")
    p.add_argument("--obj-height", type=float, default=0.0,
                   help="finite-conjugate field (object height) [mm] (microscope)")
    p.add_argument("--out", default="out.seq", help="output .seq file path")
    p.add_argument("--outdir", default="output", help="directory for generated files")
    p.add_argument("--layout", action="store_true",
                   help="also render a lens layout image")
    p.add_argument("--iters", type=int, default=300, help="optimization iterations")
    p.add_argument("--report", action="store_true", default=True,
                   help="print a text report")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    from .specs import Spec
    from . import generate, report

    aperture = "fno"
    aval = args.fno
    if args.epd is not None:
        aperture, aval = "epd", args.epd
    elif args.naf is not None:
        aperture, aval = "na", args.naf
    if aval is None:
        print("Provide one of --fno, --epd or --naf", file=sys.stderr)
        return 2

    spec = Spec(efl=args.efl, fov_deg=args.fov, aperture=aperture,
                aperture_value=aval, lens_type=args.lens_type,
                distortion_max=args.dist, min_image_clearance=args.clearance,
                object_distance=args.obj_dist if args.obj_dist else float("inf"),
                object_height=args.obj_height)

    try:
        res = generate(spec, optimize_iters=args.iters)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    os.makedirs(args.outdir, exist_ok=True)
    seq_path = os.path.join(args.outdir, args.out)
    with open(seq_path, "w", encoding="utf-8") as f:
        f.write(res.seq_text + "\n")

    if args.layout:
        img = report.draw_layout(res.design, spec.epd, spec.half_field_deg,
                                 os.path.join(args.outdir, "layout.png"))
        print(f"layout -> {img}")

    if args.report:
        print(report.text_report(spec, res))

    print(f"\nCODE V .seq written to: {os.path.abspath(seq_path)}")
    print("Read it into CODE V with:  IN " + os.path.basename(seq_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
