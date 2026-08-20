"""End-to-end tests for the multi-system (OptiForge) workflow."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optiforge import generator, metrics, report, seqparse
from optiforge.specs import Spec


def camera_spec(**kw):
    """The worked example: a 35 mm f/2 camera lens with the usual goals."""
    base = dict(
        efl=35.0, semi_field_deg=30.0, aperture="fno", aperture_value=2.0,
        units="mm", wl_short=450.0, wl_long=650.0,
        use_package_length=True, package_length_max=250.0,
        use_image_clearance=True, min_image_clearance=22.0,
        use_distortion=True, distortion_max=1.0,
        use_elem_min=True, elem_min=4, use_elem_max=True, elem_max=8,
        n_systems=4, base_name="Camera",
    )
    base.update(kw)
    return Spec(**base)


@pytest.fixture(scope="module")
def sysset():
    return generator.generate_systems(camera_spec())


def test_returns_the_requested_number_of_systems(sysset):
    assert len(sysset.systems) == 4


def test_systems_are_named_from_the_base_name(sysset):
    assert [s.name for s in sysset.systems] == [
        "Camera_01", "Camera_02", "Camera_03", "Camera_04"]
    assert [s.filename for s in sysset.systems][0] == "Camera_01.seq"


def test_systems_are_ranked_best_first(sysset):
    scores = [s.score for s in sysset.systems]
    assert scores == sorted(scores)


def test_systems_are_distinct(sysset):
    """The population must be different designs, not one design ten times."""
    sigs = {generator._signature(s.metrics, sysset.spec) for s in sysset.systems}
    assert len(sigs) == len(sysset.systems)


def test_every_system_holds_the_first_order_spec(sysset):
    for s in sysset.systems:
        m = s.metrics
        assert abs(m.efl - 35.0) < 0.35, (s.name, m.efl)
        assert abs(m.epd - 17.5) < 0.05, (s.name, m.epd)
        assert abs(m.fno - 2.0) < 0.02, (s.name, m.fno)


def test_every_system_is_buildable(sysset):
    """No surface steeper than a hemisphere over its own beam."""
    for s in sysset.systems:
        assert s.metrics.min_radius_ratio >= 1.0, (s.name,
                                                   s.metrics.min_radius_ratio)
        assert all(t > 0 for t in s.design.thick[1:]), s.name


def test_element_count_goal_is_respected(sysset):
    for s in sysset.systems:
        assert 4 <= s.metrics.elem_count <= 8, (s.name, s.metrics.elem_count)


def test_package_and_clearance_goals_are_respected(sysset):
    """A system reported as meeting its goals really does meet them.

    The search returns the best population it can find and *flags* whatever
    falls short, so the contract is "meets_goals is honest", not "every
    system always complies".
    """
    compliant = [s for s in sysset.systems if s.meets_goals]
    assert compliant, "no system met the goals at all"
    for s in compliant:
        assert s.metrics.package_length <= 250.0 * 1.001, s.name
        assert s.metrics.image_clearance >= 22.0 * 0.999, s.name
        assert s.metrics.distortion_pct <= 1.0 * 1.001, s.name
        assert 4 <= s.metrics.elem_count <= 8, s.name


def test_every_system_round_trips_through_the_seq_parser(sysset):
    for s in sysset.systems:
        assert s.seq_valid, (s.name, s.warnings)
        r = seqparse.validate(s.seq_text, expected_efl=35.0)
        assert r.ok, (s.name, r.errors)


def test_seq_carries_the_requested_wavelengths_and_units(sysset):
    text = sysset.systems[0].seq_text
    assert "DIM MM" in text
    r = seqparse.parse_seq(text)
    assert 450.0 in r.wavelengths and 650.0 in r.wavelengths
    assert r.title == "Camera_01"


def test_goal_status_is_reported_per_system(sysset):
    for s in sysset.systems:
        for key in ("efl", "package_length", "image_clearance",
                    "distortion_pct", "elem_count"):
            assert key in s.goal_status, (s.name, key)
        assert s.meets_goals == all(g["met"] for g in s.goal_status.values())


def test_summary_table_lists_goals_and_every_system(sysset):
    txt = report.summary_table(sysset)
    assert "Goals" in txt
    for s in sysset.systems:
        assert s.name in txt


# ---------------------------------------------------------------------------
# goals actually steer the search
# ---------------------------------------------------------------------------
def test_chief_ray_angle_goal_pulls_the_designs_in():
    """Switching the CRA goal on must actually steer the search."""
    free = generator.generate_systems(camera_spec(n_systems=3))
    held = generator.generate_systems(camera_spec(
        n_systems=3, use_cra=True, cra_target=10.0, cra_tolerance=4.0))

    # anything reported as compliant must really be inside the tolerance
    for s in held.systems:
        if s.goal_status["cra_deg"]["met"]:
            assert abs(s.metrics.cra_deg - 10.0) <= 4.0 * 1.01, (
                s.name, s.metrics.cra_deg)

    # and the goal must have moved the population, not just labelled it
    assert min(s.metrics.cra_deg for s in held.systems) <         min(s.metrics.cra_deg for s in free.systems)
    assert max(s.metrics.cra_deg for s in free.systems) > 14.0


def test_element_range_goal_can_ask_for_simpler_lenses():
    ss = generator.generate_systems(camera_spec(
        n_systems=3, use_elem_min=True, elem_min=2,
        use_elem_max=True, elem_max=3))
    assert ss.systems
    for s in ss.systems:
        assert 2 <= s.metrics.elem_count <= 3, (s.name, s.metrics.elem_count)


def test_tighter_package_length_produces_shorter_systems():
    """A package-length goal must bound the population it returns."""
    loose = generator.generate_systems(camera_spec(n_systems=3,
                                                   use_package_length=False))
    tight = generator.generate_systems(camera_spec(
        n_systems=3, package_length_max=45.0))
    assert tight.systems

    # anything reported as compliant really is within the limit
    for s in tight.systems:
        if s.goal_status["package_length"]["met"]:
            assert s.metrics.package_length <= 45.0 * 1.001, (
                s.name, s.metrics.package_length)

    # and the constrained population is never longer than the free one
    assert (max(s.metrics.package_length for s in tight.systems) <=
            max(s.metrics.package_length for s in loose.systems) + 1e-9)
