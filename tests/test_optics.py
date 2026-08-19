"""Unit tests for the paraxial/Seidel engine using a real CODE V double-Gauss."""
import math
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aistart.design import Design
from aistart import optics


def make_double_gauss():
    R = [float("inf"),
         round(1/0.01779284094091543, 6), round(1/0.006566600536925569, 6),
         round(1.0/0.02653743294670983, 6), 0.0, round(1.0/0.04126892878570084, 6), 0.0,
         round(1.0/-0.03523942654429074, 6), 0.0, round(1.0/-0.02636750828657709, 6),
         round(1.0/0.005636604951748545, 6), round(1.0/-0.0125926458590576, 6),
         float("inf")]
    th = [1e13, 8.75, 0.5, 12.5, 3.8, 16.36944492653564, 13.74795696388906,
          3.8, 11.0, 0.5, 7.0, 61.4875364179821]
    g = ["", "NSSK2_SCHOTT", "", "NSK2_SCHOTT", "F5_SCHOTT", "", "",
         "F5_SCHOTT", "NSK16_SCHOTT", "", "NSK16_SCHOTT", ""]
    return Design(radius=R, thick=th, glass=g, stop=6, efl=100.0)


def test_efl_of_double_gauss():
    d = make_double_gauss()
    rt = optics.trace(d, epd=50.0, field_angle_deg=10.0)
    assert abs(rt.efl - 100.0) < 2.0, rt.efl
    assert abs(rt.fno - 2.0) < 0.2, rt.fno


def test_scale_invariance():
    d = make_double_gauss()
    rt = optics.trace(d, epd=50.0, field_angle_deg=10.0)
    k = 2.0
    d2 = d.copy()
    d2.radius = [x * k if np.isfinite(x) else x for x in d.radius]
    d2.thick = [x * k for x in d.thick]
    rt2 = optics.trace(d2, epd=100.0, field_angle_deg=10.0)
    assert abs(rt2.efl / rt.efl - k) < 1e-6


def test_seidel_nonzero_for_singlet_like():
    d = make_double_gauss()
    rt = optics.trace(d, epd=50.0, field_angle_deg=10.0)
    sd = optics.seidel(rt, d)
    # a real double-Gauss has a non-trivial aberration balance (not all zero)
    assert abs(sd.s1) > 1.0
