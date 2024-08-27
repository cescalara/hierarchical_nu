import pytest
import numpy as np
from astropy import units as u

from hierarchical_nu.source.cosmology import luminosity_distance
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.source.source import Sources, PointSource
from hierarchical_nu.source.flux_model import (
    integral_power_law,
    LogParabolaSpectrum,
    PowerLawSpectrum,
)

Parameter.clear_registry()


def test_logparabola():
    Parameter.clear_registry()
    index = Parameter(2.0, "src_index")
    alpha = Parameter(index.value, "alpha")
    beta = Parameter(0.0, "beta")
    norm = Parameter(1e-10 / u.GeV / u.s / u.m**2, "norm")
    Emin = 1e2 * u.GeV
    Emax = 1e8 * u.GeV
    Enorm = 1e5 * u.GeV

    pl = PowerLawSpectrum(norm, Enorm, index, Emin, Emax)
    log = LogParabolaSpectrum(norm, Enorm, alpha, beta, Emin, Emax)

    factor_pl = PowerLawSpectrum.flux_conv_(
        index.value, Emin.to_value(u.GeV), Emax.to_value(u.GeV), 0.0, 0.0
    )
    factor_log = log.flux_conv()

    flux_density_pl = pl.total_flux_density.to_value(u.erg / u.m**2 / u.s)
    flux_density_log = log.total_flux_density.to_value(u.erg / u.m**2 / u.s)

    integral_pl = pl.integral(Emin, Emax).to_value(1 / u.m**2 / u.s)
    integral_log = log.integral(Emin, Emax).to_value(1 / u.m**2 / u.s)

    assert integral_pl == pytest.approx(integral_log)

    assert factor_pl == pytest.approx(factor_log)

    assert flux_density_pl == pytest.approx(flux_density_log)
