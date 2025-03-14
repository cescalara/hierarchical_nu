import pytest
from astropy import units as u
from hierarchical_nu.utils.lifetime import LifeTime
from hierarchical_nu.detector.icecube import IC86_II


def test_lifetime():
    lifetime = LifeTime()
    lt = lifetime.lifetime_from_mjd(56917, 57113)

    assert lt[IC86_II].to_value(u.year) == pytest.approx(0.49401826)

    lt = lifetime.lifetime_from_dm(IC86_II)

    assert lt[IC86_II].to_value(u.year) == pytest.approx(6.02031467)
