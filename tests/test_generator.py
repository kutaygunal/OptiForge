"""End-to-end tests: every lens type generates a valid, on-spec .seq."""
import math
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from aistart import generate, Spec, seqparse


CASES = [
    ("double_gauss", 50.0, 40.0, "fno", 2.8),
    ("cooke_triplet", 50.0, 35.0, "fno", 4.0),
    ("telephoto", 120.0, 20.0, "fno", 4.0),
    ("retrofocus", 20.0, 80.0, "fno", 4.0),
    ("petzval", 50.0, 15.0, "fno", 2.0),
    ("collimator", 50.0, 5.0, "fno", 2.0),
]


@pytest.mark.parametrize("lt,efl,fov,ap,av", CASES)
def test_generate_each_type(lt, efl, fov, ap, av):
    s = Spec(efl=efl, fov_deg=fov, aperture=ap, aperture_value=av, lens_type=lt)
    res = generate(s, validate_seq=True)
    assert res.seq_valid, f"seq not valid: {res.seq_valid}"
    p = res.perf
    assert abs(p["efl"] - efl) / efl < 0.02, (lt, p["efl"])
    target_fno = efl / p["epd"] if p["epd"] else None
    assert abs(p["fno"] - av) < 0.05, (lt, p["fno"])


def test_seq_roundtrip_matches_geometry():
    s = Spec(efl=50.0, fov_deg=40.0, aperture="fno", aperture_value=2.8,
             lens_type="double_gauss")
    res = generate(s, validate_seq=False)
    r = seqparse.parse_seq(res.seq_text)
    d = r.design
    # same number of surfaces
    assert len(d.radius) == len(res.design.radius)
    # radii match to rounding
    for a, b in zip(d.radius[1:-1], res.design.radius[1:-1]):
        if math.isfinite(a) and math.isfinite(b):
            assert abs(a - b) / max(abs(b), 1e-6) < 1e-3
    v = seqparse.validate(res.seq_text, expected_efl=50.0)
    assert v.ok, v.errors
    assert abs(v.efl - 50.0) < 0.01


def test_microscope_finite():
    s = Spec(efl=1.0, fov_deg=0.0, aperture="na", aperture_value=0.25,
             lens_type="microscope", object_distance=1.0, object_height=0.1)
    res = generate(s, validate_seq=True)
    assert res.seq_valid
    assert s.finite


def test_auto_selects_something():
    for efl, fov, av in [(50, 40, 2.8), (120, 20, 4.0), (20, 90, 4.0)]:
        s = Spec(efl=efl, fov_deg=fov, aperture="fno", aperture_value=av,
                 lens_type="auto")
        res = generate(s, validate_seq=True)
        assert res.seq_valid
        assert res.prototype_name
