"""Command-line interface for OptiForge.

One starting point:
    python -m optiforge.cli --efl 50 --fov 40 --fno 2.8 --type double_gauss --out out.seq

A set of starting points, ranked against goals (the OptiForge workflow):
    python -m optiforge.cli --efl 35 --semi-field 30 --fno 2 --systems 10 \
        --base-name Camera --wl-short 450 --wl-long 650 \
        --max-package 250 --min-clearance 22 --max-distortion 1 \
        --min-elem 4 --max-elem 8 --layout
"""
from __future__ import annotations

import argparse
import os
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="optiforge",
                                description="OptiForge - local starting-point "
                                            "lens design")
    # --- basic lens properties -------------------------------------------
    b = p.add_argument_group("basic lens properties")
    b.add_argument("--efl", type=float, default=50.0,
                   help="effective focal length [system units]")
    f = b.add_mutually_exclusive_group()
    f.add_argument("--fov", type=float, default=None,
                   help="full field of view [deg]")
    f.add_argument("--semi-field", dest="semi_field", type=float, default=None,
                   help="semi-field angle [deg]")
    g = b.add_mutually_exclusive_group()
    g.add_argument("--fno", type=float, default=None, help="target f-number")
    g.add_argument("--epd", type=float, default=None, help="entrance pupil diameter")
    g.add_argument("--naf", type=float, default=None, help="numerical aperture")
    b.add_argument("--units", default="mm", choices=["mm", "cm", "in"],
                   help="desired system units (default mm)")
    b.add_argument("--wl-short", dest="wl_short", type=float, default=486.1,
                   help="short wavelength [nm]")
    b.add_argument("--wl-long", dest="wl_long", type=float, default=656.3,
                   help="long wavelength [nm]")
    b.add_argument("--type", dest="lens_type", default="auto",
                   help="lens form: auto|double_gauss|cooke_triplet|telephoto|"
                        "retrofocus|petzval|microscope|collimator")

    # --- optional goals ---------------------------------------------------
    o = p.add_argument_group("optional goals (omit to leave a goal switched off)")
    o.add_argument("--max-package", dest="max_package", type=float, default=None,
                   help="maximum package length")
    o.add_argument("--min-clearance", dest="min_clearance", type=float, default=None,
                   help="minimum image clearance (back working distance)")
    o.add_argument("--max-distortion", dest="max_distortion", type=float,
                   default=3.0, help="maximum distortion [%%, +/-] (default 3)")
    o.add_argument("--cra", type=float, default=None,
                   help="target chief ray angle at the image at max field [deg]")
    o.add_argument("--cra-tol", dest="cra_tol", type=float, default=10.0,
                   help="chief ray angle tolerance [deg, +/-] (default 10)")
    o.add_argument("--min-elem", dest="min_elem", type=int, default=None,
                   help="minimum number of elements")
    o.add_argument("--max-elem", dest="max_elem", type=int, default=None,
                   help="maximum number of elements")

    # --- finite conjugates ------------------------------------------------
    c = p.add_argument_group("finite conjugates (microscope)")
    c.add_argument("--obj-dist", type=float, default=0.0,
                   help="object distance / working distance")
    c.add_argument("--obj-height", type=float, default=0.0,
                   help="object height (field)")

    # --- output -----------------------------------------------------------
    w = p.add_argument_group("output")
    w.add_argument("--systems", type=int, default=1,
                   help="number of starting points to generate (default 1)")
    w.add_argument("--base-name", dest="base_name", default="Lens",
                   help="base file name for a multi-system run")
    w.add_argument("--out", default="out.seq",
                   help="output .seq file (single-system runs only)")
    w.add_argument("--outdir", default="output", help="directory for generated files")
    w.add_argument("--layout", action="store_true",
                   help="also render lens layout images")
    w.add_argument("--iters", type=int, default=300,
                   help="optimization iterations (single-system runs)")
    w.add_argument("--report", action="store_true", default=True,
                   help="print a text report")

    # legacy alias
    w.add_argument("--dist", type=float, default=None,
                   help=argparse.SUPPRESS)
    w.add_argument("--clearance", type=float, default=None,
                   help=argparse.SUPPRESS)
    return p


def _spec_from_args(args):
    from .specs import Spec

    aperture, aval = "fno", args.fno
    if args.epd is not None:
        aperture, aval = "epd", args.epd
    elif args.naf is not None:
        aperture, aval = "na", args.naf
    if aval is None:
        raise SystemExit("Provide one of --fno, --epd or --naf")

    # legacy aliases
    max_distortion = args.dist if args.dist is not None else args.max_distortion
    min_clearance = (args.clearance if args.clearance not in (None, 0.0)
                     else args.min_clearance)

    fov = args.fov
    semi = args.semi_field
    if fov is None and semi is None:
        fov = 40.0

    return Spec(
        efl=args.efl,
        fov_deg=fov if fov is not None else 0.0,
        semi_field_deg=semi,
        aperture=aperture, aperture_value=aval,
        lens_type=args.lens_type,
        units=args.units,
        wl_short=args.wl_short, wl_long=args.wl_long,
        use_package_length=args.max_package is not None,
        package_length_max=args.max_package or 0.0,
        use_image_clearance=min_clearance is not None,
        min_image_clearance=min_clearance or 0.0,
        use_distortion=max_distortion is not None,
        distortion_max=max_distortion if max_distortion is not None else 3.0,
        use_cra=args.cra is not None,
        cra_target=args.cra or 0.0,
        cra_tolerance=args.cra_tol,
        use_elem_min=args.min_elem is not None,
        elem_min=args.min_elem or 1,
        use_elem_max=args.max_elem is not None,
        elem_max=args.max_elem or 99,
        object_distance=args.obj_dist if args.obj_dist else float("inf"),
        object_height=args.obj_height,
        n_systems=max(1, args.systems),
        base_name=args.base_name,
    )


def _run_many(args, spec) -> int:
    """Generate a set of starting points and write the summary table."""
    from . import generator, report

    print(f"OptiForge - searching starting points "
          f"({spec.n_systems} requested)...")
    try:
        sysset = generator.generate_systems(spec)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if not sysset.systems:
        print("No viable system met the specification. Try relaxing a goal or "
              "widening the element count range.", file=sys.stderr)
        return 1

    os.makedirs(args.outdir, exist_ok=True)
    for s in sysset.systems:
        path = os.path.join(args.outdir, s.filename)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(s.seq_text + "\n")
        if args.layout:
            report.draw_layout(s.design, s.metrics.epd, spec.half_field_deg,
                               os.path.join(args.outdir, f"{s.name}_layout.png"))

    summary = report.summary_table(sysset)
    print()
    print(summary)
    sum_path = os.path.join(args.outdir, f"{spec.base_name}_summary.txt")
    with open(sum_path, "w", encoding="utf-8") as fh:
        fh.write(summary + "\n")

    print()
    print(f"{len(sysset.systems)} .seq files written to "
          f"{os.path.abspath(args.outdir)}")
    print(f"summary -> {os.path.abspath(sum_path)}")
    print("Load one with:  IN " + sysset.systems[0].name)
    return 0


def _run_one(args, spec) -> int:
    from . import generate, report

    try:
        res = generate(spec, optimize_iters=args.iters)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    os.makedirs(args.outdir, exist_ok=True)
    seq_path = os.path.join(args.outdir, args.out)
    with open(seq_path, "w", encoding="utf-8") as fh:
        fh.write(res.seq_text + "\n")

    if args.layout:
        img = report.draw_layout(res.design, spec.epd, spec.half_field_deg,
                                 os.path.join(args.outdir, "layout.png"))
        print(f"layout -> {img}")

    if args.report:
        print(report.text_report(spec, res))

    print(f"\nLens sequence written to: {os.path.abspath(seq_path)}")
    print("Load it with:  IN " + os.path.basename(seq_path))
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    spec = _spec_from_args(args)
    if args.systems > 1:
        return _run_many(args, spec)
    return _run_one(args, spec)


if __name__ == "__main__":
    sys.exit(main())
