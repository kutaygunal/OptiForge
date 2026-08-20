"""Tests for the summary-table metrics and the buildability checks."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optiforge import catalog, metrics, optics, optimize
from optiforge.design import Design
from optiforge.specs import Spec


def scaled_double_gauss(efl=35.0, epd=17.5, semi=20.0):
    """The stock double Gauss, homothetically scaled to a working spec."""
    p = catalog.get("double_gauss")
    d = catalog.build(p)
    rt = optics.trace(d, epd=p.f0 / 2.0, field_angle_deg=semi)
    k = efl / rt.efl
    for i in range(1, len(d.radius) - 1):
        if d.radius[i] and np.isfinite(d.radius[i]):
            d.radius[i] *= k
    d.thick = [t * k for t in d.thick]
    rt = optics.trace(d, epd=epd, field_angle_deg=semi)
    d.thick[-1] = rt.bfl
    return d


def spec_for(semi=20.0, efl=35.0, fno=2.0):
    return Spec(efl=efl, semi_field_deg=semi, aperture="fno",
                aperture_value=fno, wl_short=450.0, wl_long=650.0).compute()


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def test_package_length_is_front_vertex_to_image():
    d = scaled_double_gauss()
    assert abs(metrics.package_length(d) - sum(d.thick[1:])) < 1e-9
    # the (near-infinite) object gap must not leak in
    assert metrics.package_length(d) < 1000.0


def test_image_clearance_is_the_last_gap():
    d = scaled_double_gauss()
    assert abs(metrics.image_clearance(d) - d.thick[-1]) < 1e-9


def test_element_count_groups_cemented_members_separately():
    # air / crown / flint cemented / air  ->  two elements
    d = Design(radius=[float("inf"), 50.0, -50.0, 80.0, float("inf")],
               thick=[1e13, 4.0, 3.0, 40.0],
               glass=["", "NBK7_SCHOTT", "NSF5_SCHOTT", ""], stop=1)
    assert metrics.element_count(d) == 2
    # one air-spaced singlet
    d2 = Design(radius=[float("inf"), 50.0, -50.0, float("inf")],
                thick=[1e13, 4.0, 40.0],
                glass=["", "NBK7_SCHOTT", ""], stop=1)
    assert metrics.element_count(d2) == 1


# ---------------------------------------------------------------------------
# image quality
# ---------------------------------------------------------------------------
def test_distortion_is_a_ratio_not_a_length():
    """Distortion is S5/(2H): scaling the whole lens must not change it."""
    s = spec_for()
    d = scaled_double_gauss()
    m1 = metrics.evaluate(d, s)

    d2 = d.copy()
    k = 3.0
    for i in range(1, len(d2.radius) - 1):
        if d2.radius[i] and np.isfinite(d2.radius[i]):
            d2.radius[i] *= k
    d2.thick = [t * k for t in d2.thick]
    d2.thick[0] = d.thick[0]
    s2 = spec_for(efl=s.efl * k)
    m2 = metrics.evaluate(d2, s2)

    assert abs(m1.distortion_pct - m2.distortion_pct) < 1e-6 * max(
        1.0, m1.distortion_pct)


def test_reference_double_gauss_has_plausible_distortion():
    d = scaled_double_gauss(semi=20.0)
    m = metrics.evaluate(d, spec_for(semi=20.0))
    assert 0.1 < m.distortion_pct < 5.0, m.distortion_pct


def test_chief_ray_angle_matches_the_traced_chief_ray():
    d = scaled_double_gauss()
    s = spec_for()
    m = metrics.evaluate(d, s)
    expect = abs(math.degrees(math.atan(float(m.raytrace.V_c[-1]))))
    assert abs(m.cra_deg - expect) < 1e-9
    assert 0.0 < m.cra_deg < 60.0


def test_relative_illumination_falls_with_field():
    lo = metrics.evaluate(scaled_double_gauss(semi=5.0), spec_for(semi=5.0))
    hi = metrics.evaluate(scaled_double_gauss(semi=30.0), spec_for(semi=30.0))
    assert lo.rel_illum_pct > hi.rel_illum_pct
    assert 0.0 <= hi.rel_illum_pct <= 100.0


def test_spot_grows_with_aperture():
    """A faster lens has more third-order blur, all else being equal."""
    slow = metrics.evaluate(scaled_double_gauss(epd=35.0 / 8.0),
                            spec_for(fno=8.0))
    fast = metrics.evaluate(scaled_double_gauss(epd=35.0 / 2.0),
                            spec_for(fno=2.0))
    assert fast.avg_spot_diam > slow.avg_spot_diam


# ---------------------------------------------------------------------------
# buildability
# ---------------------------------------------------------------------------
def test_min_radius_ratio_flags_an_impossible_surface():
    """A surface steeper than a hemisphere over its beam must be caught."""
    d = scaled_double_gauss()
    s = spec_for()
    assert metrics.evaluate(d, s).min_radius_ratio > 1.0

    bad = d.copy()
    bad.radius[1] = 1.0          # 1 mm radius under a ~9 mm marginal ray
    assert metrics.evaluate(bad, s).min_radius_ratio < 1.0


def test_edge_thickness_of_a_biconvex_element():
    """A biconvex lens is thinner at its edge than at its centre."""
    et = optimize._edge_thickness(50.0, -50.0, 5.0, 10.0)
    assert et < 5.0
    # plano-plano keeps its thickness everywhere
    flat = optimize._edge_thickness(float("inf"), float("inf"), 5.0, 10.0)
    assert abs(flat - 5.0) < 1e-9


def test_clear_aperture_includes_a_vignetted_chief_ray_share():
    d = scaled_double_gauss(semi=25.0)
    rt = optics.trace(d, epd=17.5, field_angle_deg=25.0)
    sa = metrics.clear_semi_aperture(rt, 1)
    assert sa > abs(rt.y[1])                       # more than the axial beam
    assert sa < abs(rt.y[1]) + abs(rt.y_b[1])      # but less than unvignetted
