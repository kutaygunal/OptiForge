"""Tests for spec parsing and aperture/field conversion."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aistart.specs import Spec


def test_fno_conversion():
    s = Spec(efl=50.0, fov_deg=40.0, aperture="fno", aperture_value=2.8).compute()
    assert abs(s.epd - 50.0 / 2.8) < 1e-6
    assert abs(s.na - 1.0 / 5.6) < 1e-6
    assert s.half_field_deg == 20.0


def test_epd_conversion():
    s = Spec(efl=100.0, aperture="epd", aperture_value=50.0).compute()
    assert abs(s.fno - 2.0) < 1e-9


def test_na_conversion():
    s = Spec(efl=100.0, aperture="na", aperture_value=0.5).compute()
    assert abs(s.fno - 1.0) < 1e-9
    assert abs(s.epd - 100.0) < 1e-6


def test_validate_catches_bad_input():
    assert Spec(efl=0.0).validate()        # bad efl
    assert not Spec(efl=50.0, aperture="fno", aperture_value=2.8).validate()
