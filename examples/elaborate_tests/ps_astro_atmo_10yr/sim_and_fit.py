# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: hi_nu
#     language: python
#     name: python3
# ---

from astropy.coordinates import SkyCoord
import astropy.units as u
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.simulation import Simulation, SimInfo
from hierarchical_nu.fit import StanFit
from hierarchical_nu.priors import Priors
from hierarchical_nu.source.source import Sources, PointSource, DetectorFrame
from hierarchical_nu.events import Events
from hierarchical_nu.fit import StanFit
from hierarchical_nu.utils.lifetime import LifeTime
from hierarchical_nu.priors import Priors, LogNormalPrior, NormalPrior, FluxPrior, LuminosityPrior, IndexPrior
from hierarchical_nu.utils.plotting import SphericalCircle
from hierarchical_nu.utils.roi import CircularROI, ROIList
from hierarchical_nu.detector.icecube import IC86_II, IC40, IC59, IC79, IC86_I
from hierarchical_nu.detector.r2021 import R2021LogNormEnergyResolution, R2021EnergyResolution
from hierarchical_nu.backend import DistributionMode
from hierarchical_nu.detector.input import mceq
from icecube_tools.utils.data import Uptime
import h5py
import numpy as np
import matplotlib.pyplot as plt
import arviz as av
import h5py
import ligo.skymap.plot
from hierarchical_nu.source.source import uv_to_icrs
from hierarchical_nu.utils.config_parser import ConfigParser
from hierarchical_nu.utils.config import HierarchicalNuConfig

config = HierarchicalNuConfig.from_path("hnu_config.yml")

parser = ConfigParser(config)

sources = parser.sources

sources.sources

parser.ROI
ROIList.STACK

dm = parser.detector_model
dm

obs_time = parser.obs_time
obs_time

sim = parser.create_simulation(sources, dm, obs_time)

sim.precomputation()
print(sim._get_expected_Nnu(sim._get_sim_inputs()))
print(sim._expected_Nnu_per_comp)


# fix-what w/o convolution
sim._Nex_et

# Master
sim._Nex_et

# with convolved Eres
sim._Nex_et

sim.generate_stan_code()
sim.compile_stan_code()

sim.run()

sim.show_skymap()
sim.show_spectrum()

sim.events.N

sim.save("test_events.h5", overwrite=True)

events = Events.from_file("test_events.h5")
events.N

events.plot_energy(sources[0])

fit = parser.create_fit(sources, events, dm, obs_time)

fit.precomputation(sim._exposure_integral)
fit.generate_stan_code()
fit.compile_stan_code()

fit.run(
    show_progress=True,
    inits={"L": 1e49, "src_index": 2.2, "E": [1e5] * fit.events.N, "F_atmo": 0.3, "F_diff": 1e-4, "diff_index": 2.5},
    chains=1,
    save_profile=True,
    show_console=True,
    seed=43,
)

print(fit._fit_output.diagnose())

fig, axs = fit.plot_energy_and_roi(assoc_idx=0)
# fig.savefig("roi.pdf", dpi=150)

sim_info = SimInfo.from_file("test_events.h5")
true_assoc = np.ones(events.N)
true_assoc[sim_info.outputs["Lambda"] - 1 == 0] = 0.

fig, axs = fit.plot_energy_and_roi(assoc_idx=0, true_assoc=true_assoc)


fig, axs = fit.plot_trace_and_priors(fit._def_var_names+["Nex_src", "Nex_diff", "Nex_atmo"])
fig.savefig("trace.pdf", dpi=150, bbox_inches="tight")


