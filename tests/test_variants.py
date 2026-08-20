"""Tests for the structural moves that build the candidate population."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optiforge import catalog, metrics, optics, variants

FORMS = ["double_gauss", "cooke_triplet", "telephoto", "retrofocus",
         "petzval", "collimator"]


def base_of(pid):
    p = catalog.get(pid)
    return catalog.build(p), p


def is_consistent(d):
    """Surface / gap / medium arrays must stay in step."""
    return (len(d.thick) == len(d.radius) - 1 and
            len(d.glass) == len(d.radius) - 1 and
            1 <= d.stop <= len(d.radius) - 2)


@pytest.mark.parametrize("pid", FORMS)
def test_every_move_leaves_a_consistent_traceable_design(pid):
    d0, p = base_of(pid)
    for mv in variants.base_moves():
        d = mv.fn(d0, p.f0)
        if d is None:
            continue                      # a move that does not apply is fine
        assert is_consistent(d), (pid, mv.name)
        rt = optics.trace(d, epd=p.f0 / 4.0, field_angle_deg=5.0)
        assert rt.efl == rt.efl            # not NaN
        assert all(t > 0 for t in d.thick[1:]), (pid, mv.name)
        # the original must never be mutated
        assert len(d0.radius) == len(catalog.build(p).radius)


@pytest.mark.parametrize("pid", FORMS)
def test_element_adding_moves_add_exactly_one_element(pid):
    d0, p = base_of(pid)
    n0 = metrics.element_count(d0)
    for name in ("field flattener", "front corrector"):
        mv = [m for m in variants.base_moves() if m.name == name][0]
        d = mv.fn(d0, p.f0)
        if d is None:
            continue
        assert metrics.element_count(d) == n0 + 1, (pid, name)


def test_remove_element_never_breaks_a_cemented_doublet():
    """Pulling one member out of a cemented pair would strand its partner."""
    d0, p = base_of("double_gauss")
    d = variants.remove_element(d0, p.f0)
    assert d is not None
    assert is_consistent(d)
    assert metrics.element_count(d) < metrics.element_count(d0)
    # every glass run must still be bounded by air on both sides
    for (k1, k2, _n, _phi) in variants._air_spaced_groups(d):
        before = d.glass[k1 - 1] if k1 - 1 < len(d.glass) else ""
        after = d.glass[k2] if k2 < len(d.glass) else ""
        assert not (before or "").strip()
        assert not (after or "").strip()


def test_remove_element_keeps_the_object_gap():
    """Removing the leading group must not swallow the object distance."""
    d0, p = base_of("petzval")
    d = variants.remove_element(d0, p.f0)
    assert d is not None
    assert d.thick[0] == d0.thick[0]


def test_split_doublet_adds_a_surface_but_no_element():
    d0, p = base_of("petzval")
    d = variants.split_doublet(d0, p.f0)
    assert d is not None
    assert len(d.radius) == len(d0.radius) + 1
    assert metrics.element_count(d) == metrics.element_count(d0)
    assert is_consistent(d)


def test_swap_glasses_changes_glass_but_not_geometry():
    d0, p = base_of("cooke_triplet")
    d = variants.swap_glasses(d0, p.f0, 1)
    assert d is not None
    assert d.radius == d0.radius
    assert d.thick == d0.thick
    assert d.glass != d0.glass


def test_perturb_is_deterministic_per_seed():
    d0, p = base_of("double_gauss")
    a = variants.perturb(d0, p.f0, seed=3)
    b = variants.perturb(d0, p.f0, seed=3)
    c = variants.perturb(d0, p.f0, seed=4)
    assert a.radius == b.radius
    assert a.radius != c.radius
    assert a.radius != d0.radius
