from icecube_tools.utils.data import RealEvents
from hierarchical_nu.events import Events
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.utils.roi import CircularROI, RectangularROI, ROIList, FullSkyROI
from hierarchical_nu.detector.icecube import Refrigerator

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
import numpy as np

events_file_name = "test_event_read_write.h5"


def test_event_class(output_directory):
    ROIList.clear_registry()
    roi = FullSkyROI()

    Parameter.clear_registry()
    Emin_det = Parameter(1e1 * u.GeV, "Emin_det", fixed=True)

    N_in = 50
    energies_in = np.linspace(1, 10, N_in) * u.TeV

    coords_in = SkyCoord(
        ra=np.linspace(0, 2 * np.pi, N_in) * u.rad,
        dec=np.linspace(-np.pi / 2, np.pi / 2, N_in) * u.rad,
    )

    types_in = np.zeros(N_in)
    ang_errs_in = np.ones(N_in) * u.deg
    mjd_in = Time(np.tile(99, N_in), format="mjd")

    events_in = Events(
        energies=energies_in,
        coords=coords_in,
        types=types_in,
        ang_errs=ang_errs_in,
        mjd=mjd_in,
    )

    events_file = output_directory / events_file_name
    events_in.to_file(events_file)
    events_out = Events.from_file(events_file)

    assert np.all(events_out.energies == energies_in)

    assert np.all(events_out.types == types_in)
    assert np.all(events_out.ang_errs == ang_errs_in)
    assert np.all(events_out.mjd == mjd_in)

    events_out = Events.from_file(events_file)
    N = events_out.N

    events_out.remove(1)

    assert events_out.N == N - 1

    mask = events_out.energies > 5 * u.TeV
    events_out.select(mask)

    assert np.all(events_out.energies > 5 * u.TeV)
    assert events_out.N < N
