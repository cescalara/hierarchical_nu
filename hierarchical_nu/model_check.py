import os
import sys
import numpy as np
import h5py
import arviz as av
import time
from matplotlib import pyplot as plt
import matplotlib.patches as mpl_patches
from joblib import Parallel, delayed
from astropy import units as u
from astropy.coordinates import SkyCoord
from cmdstanpy import CmdStanModel
from scipy.stats import uniform
from omegaconf import OmegaConf
from typing import List, Union
from pathlib import Path
from time import time as thyme

from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.simulation import Simulation
from hierarchical_nu.fit import StanFit
from hierarchical_nu.stan.interface import STAN_GEN_PATH
from hierarchical_nu.stan.sim_interface import StanSimInterface
from hierarchical_nu.stan.fit_interface import StanFitInterface
from hierarchical_nu.utils.config import HierarchicalNuConfig
from hierarchical_nu.utils.config_parser import ConfigParser
from hierarchical_nu.priors import (
    Priors,
)
from hierarchical_nu.utils.git import git_hash
from hierarchical_nu.events import Events

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ModelCheck:
    """
    Check statistical model by repeatedly
    fitting simulated data using different random seeds.
    """

    def __init__(
        self,
        config: HierarchicalNuConfig,
        truths=None,
        priors=None,
    ):
        """
        :param config: HhierarchicalNuConfig instance
        :param truths: true parameter values
        :param priors: priors to overwrite the config's priors
        """

        self.config = config
        self.parser = ConfigParser(self.config)

        if priors:
            self.priors = priors
            logger.info("Found priors")

        else:
            logger.info("Loading priors from config")

            self.priors = self.parser.priors

        if truths:
            logger.info("Found true values")
            self.truths = truths

        # Config
        self.parser.ROI

        # Sources
        self._sources = self.parser.sources
        if not self._sources.background:
            f_arr = self._sources.f_arr().value
            f_arr_astro = self._sources.f_arr_astro().value

        # Detector
        self._detector_model_type = self.parser.detector_model
        self._obs_time = self.parser.obs_time
        if not truths:
            logger.info("Loading true values from config")
            # self._nshards = parameter_config["nshards"]
            self._threads_per_chain = self.config.stan_config["threads_per_chain"]

            sim = self.parser.create_simulation(
                self._sources, self._detector_model_type, self._obs_time
            )

            self.sim = sim
            sim.precomputation()
            self._exposure_integral = sim._exposure_integral
            sim_inputs = sim._get_sim_inputs()
            Nex = sim._get_expected_Nnu(sim_inputs)
            Nex_per_comp = sim._expected_Nnu_per_comp
            self._Nex_et = sim._Nex_et
            # Truths
            self.truths = {}

            flux_unit = 1 / (u.m**2 * u.s)

            diffuse_bg = self._sources.diffuse
            if diffuse_bg:
                # self.truths["F_diff"] = diffuse_bg.flux_model.total_flux_int.to_value(
                #    flux_unit
                # )
                self.truths["diff_index"] = Parameter.get_parameter("diff_index").value
                self.truths["diffuse_norm"] = (
                    diffuse_bg.flux_model(
                        diffuse_bg.flux_model.spectral_shape._normalisation_energy,
                        0 * u.rad,
                        0 * u.rad,
                    )
                    * 4.0
                    * np.pi
                    * u.sr
                ).to_value(1 / u.GeV / u.m**2 / u.s)
            atmo_bg = self._sources.atmospheric
            if atmo_bg:
                self.truths["F_atmo"] = atmo_bg.flux_model.total_flux_int.to_value(
                    flux_unit
                )
            try:
                self.truths["L"] = [
                    Parameter.get_parameter("luminosity").value.to_value(u.GeV / u.s)
                ]
            except ValueError:
                self.truths["L"] = [
                    Parameter.get_parameter(f"ps_{_}_luminosity")
                    .to_value(u.GeV / u.s)
                    .value
                    for _ in range(len(self._sources.point_source))
                ]
            if not self._sources.background:
                self.truths["f_arr"] = f_arr
                self.truths["f_arr_astro"] = f_arr_astro
            try:
                self.truths["src_index"] = [Parameter.get_parameter("src_index").value]
            except ValueError:
                self.truths["src_index"] = [
                    Parameter.get_parameter(f"ps_{_}_src_index").value
                    for _ in range(len(self._sources.point_source))
                ]
            try:
                self.truths["beta_index"] = [
                    Parameter.get_parameter("beta_index").value
                ]
            except ValueError:
                try:
                    self.truths["beta_index"] = [
                        Parameter.get_parameter(f"ps_{_}_beta_index").value
                        for _ in range(len(self._sources.point_source))
                    ]
                except ValueError:
                    pass
            try:
                self.truths["E0_src"] = [
                    Parameter.get_parameter("E0_src").value.to_value(u.GeV)
                ]
            except ValueError:
                try:
                    self.truths["beta_index"] = [
                        Parameter.get_parameter(f"ps_{_}_E0_src").value.to_value(u.GeV)
                        for _ in range(len(self._sources.point_source))
                    ]
                except ValueError:
                    pass
            if not self.parser._hnu_config.parameter_config.data_bg:
                self.truths["Nex"] = Nex
            self.truths["Nex_src"] = np.sum(
                Nex_per_comp[0 : len(self._sources.point_source)]
            )
            upper_idx = len(self._sources.point_source)
            if self._sources.diffuse:
                self.truths["Nex_diff"] = Nex_per_comp[len(self._sources.point_source)]
                upper_idx += 1
            if self._sources.atmospheric:
                self.truths["Nex_atmo"] = Nex_per_comp[-1]
            self.truths["f_det"] = Nex_per_comp[0] / Nex
            self.truths["f_det_astro"] = Nex_per_comp[0] / sum(
                Nex_per_comp[0:upper_idx]
            )

        self._default_var_names = [key for key in self.truths]
        self._default_var_names.append("Fs")
        self._default_var_names.append("L_ind")
        self._default_var_names.append("Nex_bg")
        self._diagnostic_names = ["lp__", "divergent__", "treedepth__", "energy__"]

    @staticmethod
    def initialise_env(output_dir, config: Union[None, HierarchicalNuConfig] = None):
        """
        Script to set up enviroment for parallel
        model checking runs.

        * Runs MCEq for atmo flux if needed
        * Generates and compiles necessary Stan files
        Only need to run once before calling ModelCheck(...).run()
        """

        # Config
        if config is None:
            logger.info("Loading default config")
            config = HierarchicalNuConfig.load_default()

        parser = ConfigParser(config)

        parser.ROI

        # Run MCEq computation
        logger.info("Setting up MCEq run for AtmopshericNumuFlux")

        # Build necessary details to define simulation and fit code
        detector_model_type = parser.detector_model
        obs_time = parser.obs_time
        sources = parser.sources
        # Generate sim Stan file
        sim = parser.create_simulation(sources, detector_model_type, obs_time)
        sim.generate_stan_code()
        logger.info(f"Generated stan sim code")
        sim.compile_stan_code()
        logger.info("Compiled stan sim code")

        # Generate fit Stan file
        fit = parser.create_fit(
            sources,
            # use some dummy event to create a fit
            Events(
                np.array([1e5]) * u.GeV,
                SkyCoord(ra=0 * u.deg, dec=0 * u.deg, frame="icrs"),
                [6],
                np.array([0.2]) * u.deg,
                [99.0],
            ),
            detector_model_type,
            obs_time,
        )
        fit.generate_stan_code()
        logger.info("Generated fit stan file")
        fit.compile_stan_code()
        logger.info("Compiled fit stan file")

        # Comilation of Stan models
        logger.info("Compile Stan models")

        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def parallel_run(self, n_jobs=1, n_subjobs=1, seed: int = 42, **kwargs):
        """
        Run model checks in parallel
        :param n_jobs: Number of parallel jobs
        :param n_subjobs: Number of sequential simulations/fits per parallel job
        :param seed: random seed for simulation and fit
        :param kwargs: kwargs to be passed to hierarchical_nu.fit.StanFit
        """

        job_seeds = [(seed + job) * n_subjobs for job in range(n_jobs)]

        self._results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(self._single_run)(n_subjobs, seed=s, **kwargs) for s in job_seeds
        )

    def save(
        self,
        filename: Path,
        save_events: bool = False,
        overwrite: bool = False,
    ):
        """
        Save model check
        :param filename: output filename
        :param save_events: If True save simulated events as well
        """

        if os.path.exists(filename) and not overwrite:
            logger.warning(f"File {filename} already exists.")
            file = os.path.splitext(filename)[0]
            ext = os.path.splitext(filename)[1]
            file += f"_{int(thyme())}"
            filename = file + ext

        with h5py.File(filename, "w") as f:
            truths_folder = f.create_group("truths")
            for key, value in self.truths.items():
                truths_folder.create_dataset(key, data=value)

            sim_folder = f.create_group("sim")

            for i, res in enumerate(self._results):
                folder = f.create_group("results_%i" % i)

                for key, value in res.items():
                    if key == "events":
                        continue
                    elif key == "association_prob":
                        if not save_events:
                            continue
                        for c, prob in enumerate(value):
                            folder.create_dataset(f"association_prob_{c}", data=prob)
                    elif key == "parameter_names":
                        for c, data in enumerate(res[key]):
                            folder.create_dataset(f"par_names_{c}", data=data)
                    elif key == "N_Eff":
                        for c, data in enumerate(res[key]):
                            folder.create_dataset(f"N_Eff_{c}", data=data)
                    elif key == "ESS_bulk":
                        for c, data in enumerate(res[key]):
                            folder.create_dataset(f"ESS_bulk_{c}", data=data)
                    elif key == "ESS_tail":
                        for c, data in enumerate(res[key]):
                            folder.create_dataset(f"ESS_tail_{c}", data=data)
                    elif key == "R_hat":
                        for c, data in enumerate(res[key]):
                            folder.create_dataset(f"R_hat_{c}", data=data)
                    elif key != "Lambda" and key != "event_Lambda":
                        folder.create_dataset(key, data=value)
                    elif key == "Lambda":
                        sim_folder.create_dataset("sim_%i" % i, data=np.vstack(value))
                    elif key == "event_Lambda":
                        for c, data in enumerate(res["event_Lambda"]):
                            sim_folder.create_dataset(
                                f"event_Lambda_{i}_{c}", data=data
                            )
            f.create_dataset("version", data=git_hash)

            # Create string of used config, to be loaded later to re-create source lists
            config_string = OmegaConf.to_yaml(self.parser._hnu_config)
            f.create_dataset("config", data=config_string)

        if save_events and "events" in res.keys():
            for i, res in enumerate(self._results):
                for c, events in enumerate(res["events"]):
                    events.to_file(
                        filename, append=True, group_name=f"sim/events_{i}_{c}"
                    )

        self.priors.addto(filename, "priors")

    @classmethod
    def load(cls, filename_list):
        """
        Load previously saved model checks
        :param filename_list: list of model check filenames
        """

        if not isinstance(filename_list, list):
            filename_list = [filename_list]
        with h5py.File(filename_list[0], "r") as f:
            job_folder = f["results_0"]
            _result_names = [key for key in job_folder]

        truths = {}
        priors = None

        results = {}
        for key in _result_names:
            results[key] = []

        file_truths = {}

        sim = {}
        sim_N = []

        for filename in filename_list:
            file_priors = Priors.from_group(filename, "priors")

            if not priors:
                priors = file_priors

            with h5py.File(filename, "r") as f:
                truths_folder = f["truths"]
                sim_folder = f["sim"]
                for key, value in truths_folder.items():
                    file_truths[key] = value[()]

                if not truths:
                    truths = file_truths

                if truths != file_truths:
                    raise ValueError(
                        "Files in list have different truth settings and should not be combined"
                    )

                """
                n_jobs = len(
                    [
                        key
                        for key in f
                        if (
                            key != "truths"
                            and key != "priors"
                            and key != "sim"
                            and key != "version"
                            and key != "config"
                        )
                    ]
                )
                """

                i = 0
                while True:
                    try:
                        # for i in range(n_jobs):
                        job_folder = f["results_%i" % i]
                        for res_key in job_folder:
                            results[res_key].extend(job_folder[res_key][()])
                        # sim["sim_%i_Lambda" % i] = sim_folder["sim_%i" % i][()]
                        sim_N.extend(sim_folder["sim_%i" % i][()])
                        i += 1
                    except KeyError:
                        break

        with h5py.File(filename, "r") as f:
            config_string = f["config"][()].decode("ascii")
        config = OmegaConf.create(config_string)

        output = cls(config, truths=truths, priors=priors)
        output.results = results
        output.sim_Lambdas = sim
        output.sim_N = sim_N

        return output

    def compare(
        self,
        var_names: Union[None, List[str]] = None,
        var_labels: Union[None, List[str]] = None,
        show_prior: bool = False,
        nbins: int = 50,
        mask_results: Union[None, np.ndarray] = None,
        alpha=0.1,
        show_N: bool = False,
    ):
        """
        Compare posteriors of parameters with simulation inputs
        :param var_names: Parameter names to compare
        :param var_labels: x-labels to be plotted
        :param show_prior: True if prior distributions should be overplotted
        :param nbins: Number of bins for posterior histograms
        :param mask_results: numpy array mask to mask out single samples
        :param alpha: alpha of histograms
        :param show_N: overplot true number of events per source component
        """
        if not var_names:
            var_names = self._default_var_names

        if not var_labels:
            var_labels = self._default_var_names

        mask = np.tile(False, len(self.results[var_names[0]]))
        if mask_results is not None:
            mask[mask_results] = True
        else:
            mask_results = np.array([])

        N = len(var_names)
        fig, ax = plt.subplots(N, figsize=(5, N * 3))

        for v, var_name in enumerate(var_names):
            # Check if var_name exists
            if var_name not in self.results.keys():
                continue
            # Check if entry is empty
            if not self.results[var_name]:
                continue
            if (
                var_name == "L"
                or var_name == "L_ind"
                # or var_name == "F_diff"
                # or var_name == "F_atmo"
                or var_name == "Fs"
                or var_name == "E0_src"
            ):
                log = True
                bins = np.geomspace(
                    np.min(np.array(self.results[var_name])[~mask]),
                    np.max(np.array(self.results[var_name])[~mask]),
                    nbins,
                )
                prior_supp = np.geomspace(
                    np.min(np.array(self.results[var_name])[~mask]),
                    np.max(np.array(self.results[var_name])[~mask]),
                    1000,
                )
                ax[v].set_xscale("log")

            else:
                log = False
                prior_supp = np.linspace(
                    np.min(np.array(self.results[var_name])[~mask]),
                    np.max(np.array(self.results[var_name])[~mask]),
                    1000,
                )
                bins = np.linspace(
                    np.min(np.array(self.results[var_name])[~mask]),
                    np.max(np.array(self.results[var_name])[~mask]),
                    nbins,
                )
            max_value = 0

            for i in range(len(self.results[var_name])):
                if i not in mask_results and len(self.results[var_name]) != 0:
                    if log:
                        n, _ = np.histogram(
                            np.log10(self.results[var_name][i]),
                            np.log10(bins),
                            density=True,
                        )
                    else:
                        n, _ = np.histogram(
                            self.results[var_name][i], bins, density=True
                        )
                    n = np.hstack((np.array([0]), n, np.array([n[-1], 0])))
                    plot_bins = np.hstack(
                        (np.array([bins[0] - 1e-10]), bins, np.array([bins[-1]]))
                    )

                    low = np.nonzero(n)[0].min()
                    high = np.nonzero(n)[0].max()
                    ax[v].step(
                        plot_bins[low - 1 : high + 2],
                        n[low - 1 : high + 2],
                        color="#017B76",
                        alpha=alpha,
                        lw=1.0,
                        where="post",
                    )

                    if show_N and var_name == "Nex_src":
                        for val in self.sim_N:
                            # Overplot the actual number of events for each component
                            # for line in val:
                            ax[v].axvline(val[0], ls="-", c="red", lw=0.3)

                    elif show_N and var_name == "Nex_diff":
                        for val in self.sim_N:
                            # Overplot the actual number of events for each component
                            # for line in val:
                            ax[v].axvline(val[1], ls="-", c="red", lw=0.3)

                    elif show_N and var_name == "Nex_atmo":
                        for val in self.sim_N:
                            # Overplot the actual number of events for each component
                            # for line in val:
                            ax[v].axvline(val[2], ls="-", c="red", lw=0.3)

                    max_value = n.max() if n.max() > max_value else max_value

            if show_prior:
                N = len(self.results[var_name][0]) * 100

                # this case distinction is overly complicated
                if var_name == "L" or var_name == "F_atmo" or var_name == "Fs":
                    # wtf, where is the elif or else here?
                    pass

                if "f_" in var_name and not "diff" in var_name:  # yikes
                    plot = False

                elif "Nex" in var_name:
                    plot = False

                elif "L_ind" == var_name or "L" == var_name:
                    plot = True
                    prior = self.priors.to_dict()["L"]
                    prior_density = prior.pdf_logspace(prior_supp * prior.UNITS)

                elif "index" in var_name:
                    prior_density = self.priors.to_dict()[var_name].pdf(
                        prior_supp * self.priors.to_dict()[var_name].UNITS
                    )
                    plot = True

                elif not "Fs" in var_name:
                    prior_density = self.priors.to_dict()[var_name].pdf_logspace(
                        prior_supp * self.priors.to_dict()[var_name].UNITS
                    )
                    plot = True
                else:
                    plot = False

                if plot:
                    ax[v].plot(
                        prior_supp,
                        prior_density,
                        color="k",
                        alpha=0.5,
                        lw=2,
                        label="Prior",
                    )

            if var_name in self.truths.keys():
                ax[v].axvline(
                    self.truths[var_name], color="k", linestyle="-", label="Truth"
                )
                counts_hdi = 0
                counts_50_quantile = 0
                for d in self.results[var_name]:
                    hdi = av.hdi(d, 0.5)
                    quantile = np.quantile(d, [0.25, 0.75])
                    true_val = self.truths[var_name]
                    if true_val <= hdi[1] and true_val >= hdi[0]:
                        counts_hdi += 1
                    if true_val <= quantile[1] and true_val >= hdi[0]:
                        counts_50_quantile += 1
                length = len(self.results[var_name]) - mask_results.size
                text = [
                    f"fraction in 50\% HDI: {counts_hdi / length:.2f}\n"
                    + f"fraction in 50\% central interval: {counts_50_quantile / length:.2f}"
                ]
                handles = [
                    mpl_patches.Rectangle(
                        (0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0
                    )
                ]
                ax[v].legend(
                    handles=handles,
                    labels=text,
                    loc="best",
                    handlelength=0,
                    handletextpad=0,
                )

            ax[v].set_xlabel(var_labels[v], labelpad=10)
            ax[v].set_yticks([])

        fig.tight_layout()
        return fig, ax

    def diagnose(self):
        """
        Quickly check output of CmdStanMCMC.diagnose().
        Return index of fits with issues.
        """

        diagnostics_array = np.array(self.results["diagnostics_ok"])

        ind_not_ok = np.where(diagnostics_array == 0)[0]

        if len(ind_not_ok) > 0:
            logger.warning(
                "%.2f percent of fits have issues!"
                % (len(ind_not_ok) / len(diagnostics_array) * 100)
            )

        else:
            logger.info("No issues found")

        return ind_not_ok

    def _single_run(self, n_subjobs, seed, **kwargs):
        """
        Single run to be called using Parallel.
        """

        save_events = kwargs.pop("save_events", False)

        sys.stderr.write("Random seed: %i\n" % seed)

        self.parser.ROI

        sources = self.parser.sources

        subjob_seeds = [(seed + subjob) * 10 for subjob in range(n_subjobs)]

        outputs = {}
        for key in self._default_var_names:
            outputs[key] = []
        for key in self._diagnostic_names:
            outputs[key] = []

        outputs["diagnostics_ok"] = []
        outputs["run_time"] = []
        outputs["Lambda"] = []
        outputs["parameter_names"] = []
        outputs["N_Eff"] = []
        outputs["ESS_bulk"] = []
        outputs["ESS_tail"] = []
        outputs["R_hat"] = []
        if save_events:
            outputs["events"] = []
            outputs["association_prob"] = []
            outputs["event_Lambda"] = []

        fit = None
        for i, s in enumerate(subjob_seeds):
            sys.stderr.write("Run %i\n" % i)
            # Simulation
            # Should reduce time consumption if only on first iteration model is compiled
            if i == 0:
                sim = self.parser.create_simulation(
                    sources, self.parser.detector_model, self.parser.obs_time
                )
                sim.precomputation(self._exposure_integral)
                sim.setup_stan_sim(".stan_files/sim_code")

            sim.run(seed=s, verbose=True)
            # self.sim = sim

            # If asimov option is used and data added as background, just use the background
            # else continue

            data_bg = self.config.parameter_config.data_bg
            if not sim.events and not data_bg:
                continue

            events = sim.events
            if events:
                # Create a mask for the sampled events
                # for point sources; events may be scattered outside of the ROI,
                # we need to catch these and delete from the event lists used for the fit.
                # Skip the temporal selection because currently the sampled time stamps
                # have an arbitrary value (we can only do time-averaged simulations)
                mask = events.apply_ROIS(events.coords, events.mjd, skip_time=True)
                idx = np.logical_or.reduce(mask)

                lambd = sim._sim_output.stan_variable("Lambda").squeeze()[idx]

                new_events = Events(
                    events.energies[idx],
                    events.coords[idx],
                    events.types[idx],
                    events.ang_errs[idx],
                    events.mjd[idx],
                )

            if data_bg:
                # If we use data as background, sample scrambled data and add to point source events
                bg_events = Events.from_ev_file(
                    *self.parser.detector_model,
                    scramble_ra=True,
                    scramble_mjd=True,
                    seed=s,
                )
                N_bg = bg_events.N
                if events:
                    new_events = new_events.merge(bg_events)
                    lambd = np.concatenate((lambd, np.array([4.0] * N_bg)))
                else:
                    new_events = bg_events
                    lambd = np.array([4.0] * N_bg)
            else:
                N_bg = 0

            ps = np.sum(lambd == 1.0)
            diff = np.sum(lambd == 2.0)
            atmo = np.sum(lambd == 3.0)
            lam = np.array([ps, diff, atmo, N_bg])

            # Fit
            # Same as above, save time
            # Also handle in case first sim has no events
            if not fit:
                fit = self.parser.create_fit(
                    sources,
                    new_events,
                    self.parser.detector_model,
                    self.parser.obs_time,
                )
                fit.precomputation(self._exposure_integral)
                fit.setup_stan_fit(".stan_files/model_code")

            else:
                fit.events = new_events

            start_time = time.time()

            share_L = self.config.parameter_config.share_L
            share_src_index = self.config.parameter_config.share_src_index

            if not share_L:
                L_init = [1e49] * len(self._sources.point_source)
                P_init = [0.3] * len(self._sources.point_source)
            else:
                L_init = 1e49
                P_init = 0.3

            if not share_src_index:
                src_init = [2.3] * len(self._sources.point_source)
                beta_init = [0.05] * len(self._sources.point_source)
                E0_init = [1e6] * len(self._sources.point_source)
                eta_init = [40] * len(self._sources.point_source)
            else:
                src_init = 2.3
                beta_init = 0.05
                E0_init = 1e6
                eta_init = 40
            try:
                F_atmo_range = Parameter.get_parameter("F_atmo").par_range
                F_atmo_init = np.sum(F_atmo_range) / 2
            except ValueError:
                F_atmo_init = 0.3 / u.m**2 / u.s

            try:
                diff_norm = Parameter.get_parameter("diff_norm").value
                diff_init = diff_norm.to_value(1 / u.GeV / u.m**2 / u.s)
            except ValueError:
                diff_init = 2e-13

            inits = {
                "diffuse_norm": diff_init,
                "F_atmo": F_atmo_init.to_value(1 / (u.m**2 * u.s)),
                "E": [1e5] * fit.events.N,
                "L": L_init,
                "src_index": src_init,
                "Nex_per_ps": [5] * len(fit.sources.point_source),
                "N_bg": fit.events.N - 5,
                "E0_src": E0_init,
                "beta_index": beta_init,
                "eta": eta_init,
                "pressure_ratio": P_init,
            }
            fit.run(
                seed=s,
                show_progress=True,
                inits=inits,
                **kwargs,
            )
            self.fit = fit

            # Store output
            run_time = time.time() - start_time
            sys.stderr.write("time: %.5f\n" % run_time)
            outputs["run_time"].append(run_time)
            outputs["Lambda"].append(lam)

            for key in self._default_var_names:
                try:
                    outputs[key].append(fit._fit_output.stan_variable(key))
                except ValueError:
                    pass
            for key in self._diagnostic_names:
                outputs[key].append(fit._fit_output.method_variables()[key])

            if save_events:
                outputs["events"].append(new_events)
                outputs["association_prob"].append(
                    np.array(fit._get_event_classifications())
                )
                outputs["event_Lambda"].append(lambd)

            diagnostics_output_str = fit._fit_output.diagnose()

            if "no problems detected" in diagnostics_output_str:
                outputs["diagnostics_ok"].append(1)
            else:
                outputs["diagnostics_ok"].append(0)

            # List of keys for which we are looking in the entirety of stan parameters
            key_stubs = [
                "lp__",
                "L",
                "_luminosity",
                "src_index",
                "_src_index",
                "beta_index",
                "_beta_index",
                "E0_src",
                "_E0_src",
                "E",
                "Esrc",
                "F_atmo",
                "F_diff",
                "diff_index",
                "Nex",
                "f_det",
                "f_arr",
                "logF",
                "Ftot",
                "Fs",
            ]

            keys = []
            summary = fit._fit_output.summary()
            for k, v in summary["R_hat"].items():
                for key in key_stubs:
                    if key in k:
                        keys.append(k)
                        break

            R_hat = np.array([summary["R_hat"][k] for k in keys])
            if "ESS_bulk" in summary.keys():
                ESS_bulk = np.array([summary["ESS_bulk"][k] for k in keys])
                ESS_tail = np.array([summary["ESS_tail"][k] for k in keys])
                outputs["ESS_bulk"].append(ESS_bulk)
                outputs["ESS_tail"].append(ESS_tail)
            if "N_Eff" in summary.keys():
                N_Eff = np.array([summary["N_Eff"][k] for k in keys])
                outputs["N_Eff"].append(N_Eff)
            outputs["parameter_names"].append(np.array(keys, dtype="S"))
            outputs["R_hat"].append(R_hat)

            logging.disable(logging.NOTSET)

        return outputs

    def _get_prior_func(self, var_name):
        """
        Return function of param "var_name" that
        describes its prior.
        """

        try:
            prior = self.priors.to_dict()[var_name]

            prior_func = prior.pdf

        except:
            if var_name == "f_arr" or var_name == "f_arr_astro":

                def prior_func(f):
                    return uniform(0, 1).pdf(f)

            else:
                raise ValueError("var_name not recognised")

        return prior_func
