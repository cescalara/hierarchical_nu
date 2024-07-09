from typing import List
from collections import OrderedDict
from astropy import units as u


from hierarchical_nu.stan.interface import StanInterface

from hierarchical_nu.backend.stan_generator import (
    FunctionsContext,
    Include,
    DataContext,
    TransformedDataContext,
    GeneratedQuantitiesContext,
    ForLoopContext,
    IfBlockContext,
    ElseIfBlockContext,
    ElseBlockContext,
    WhileLoopContext,
    FunctionCall,
)

from hierarchical_nu.backend.variable_definitions import (
    ForwardVariableDef,
    ForwardArrayDef,
    ForwardVectorDef,
    InstantVariableDef,
)

from hierarchical_nu.backend.expression import StringExpression
from hierarchical_nu.backend.parameterizations import DistributionMode

# from hierarchical_nu.events import NT, CAS, IC40, IC59, IC79, IC86_I, IC86_II

from hierarchical_nu.detector.icecube import EventType

from hierarchical_nu.source.source import Sources, DetectorFrame
from hierarchical_nu.source.flux_model import LogParabolaSpectrum

from hierarchical_nu.utils.roi import CircularROI, ROIList


class StanSimInterface(StanInterface):
    """
    An interface for generating the Stan simulation code.
    """

    def __init__(
        self,
        output_file: str,
        sources: Sources,
        event_types: List[EventType],
        atmo_flux_energy_points: int = 100,
        atmo_flux_theta_points: int = 30,
        includes: List[str] = [
            "interpolation.stan",
            "utils.stan",
            "vMF.stan",
            "rejection_sampling.stan",
        ],
        force_N: bool = False,
    ):
        """
        An interface for generating the Stan simulation code.

        :param output_file: Name of the file to write to
        :param sources: Sources object containing sources to be simulated
        :param detector_model_type: Type of the detector model to be used
        :param atmo_flux_theta_points: Number of points to use for the grid of
        atmospheric flux
        :param includes: List of names of stan files to include into the
        functions block of the generated file
        :param N: dict with keys being python string of DetectorModel. Value needs to be a list of length `len(sources)`
        """

        super().__init__(
            output_file=output_file,
            sources=sources,
            event_types=event_types,
            includes=includes,
        )

        self._atmo_flux_energy_points = atmo_flux_energy_points

        self._atmo_flux_theta_points = atmo_flux_theta_points

        if force_N:
            self._force_N = True

        else:
            self._force_N = False

        self._dm = OrderedDict()

        for et in self._event_types:
            # Include the PDF mode of the detector model
            dm = et.model(DistributionMode.RNG)

            if dm.RNG_FILENAME not in self._includes:
                self._includes.append(dm.RNG_FILENAME)
            dm.generate_code(
                DistributionMode.RNG,
                rewrite=False,
            )

            self._dm[et] = dm

    def _functions(self):
        """
        Write the functions section of the Stan file.
        """

        with FunctionsContext():
            # Include all the specified files
            for include_file in self._includes:
                _ = Include(include_file)

            for event_type in self._event_types:
                self._dm[event_type].generate_rng_function_code()

            # If we have point sources, include the shape of their PDF
            # and how to convert from energy to number flux
            if self.sources.point_source:
                if self._logparabola or self._pgamma:
                    self._ps_spectrum.make_stan_utility_func(False, False, False)
                self._src_spectrum_lpdf = (
                    # Use a different function here
                    # because for the twicebroken powerlaw we
                    # do not want to sample the steep flanks
                    self._ps_spectrum.make_stan_sampling_lpdf_func(
                        "src_spectrum_logpdf",
                    )
                )

                self._flux_conv = self._ps_spectrum.make_stan_flux_conv_func(
                    "flux_conv", False, False, False
                )

            # If we have diffuse sources, include the shape of their PDF
            if self.sources.diffuse:
                self._diff_spectrum_lpdf = self._diff_spectrum.make_stan_lpdf_func(
                    "diff_spectrum_logpdf"
                )

            # If we have atmospheric sources, include the atmospheric flux table
            # the density of the grid in theta (ie. declination) is specified here
            if self.sources.atmospheric:
                atmo_flux_model = self.sources.atmospheric.flux_model

                # Increasing theta points too much makes compilation very slow
                # Could switch to passing array as data if problematic
                self._atmo_flux = atmo_flux_model.make_stan_function(
                    energy_points=self._atmo_flux_energy_points,
                    theta_points=self._atmo_flux_theta_points,
                )

    def _data(self):
        """
        Write the data section of the Stan file.
        """

        with DataContext():
            # Useful strings depending on the total number of sources
            # Ns is the number of point sources
            self._Ns = ForwardVariableDef("Ns", "int")
            self._Ns_str = ["[", self._Ns, "]"]
            self._Ns_1p_str = ["[", self._Ns, "+1]"]
            self._Ns_2p_str = ["[", self._Ns, "+2]"]

            if self.sources.atmospheric and self.sources.diffuse:
                Ns_string = "Ns+2"
            elif self.sources.diffuse or self.sources.atmospheric:
                Ns_string = "Ns+1"
            else:
                Ns_string = "Ns"

            if self.sources.point_source:
                self._L = ForwardVariableDef("L", "vector[Ns]")
                self._Emin_src = ForwardVariableDef("Emin_src", "vector[Ns]")
                self._Emax_src = ForwardVariableDef("Emax_src", "vector[Ns]")
                self._src_index = ForwardVariableDef("src_index", "vector[Ns]")
                if self._logparabola:
                    self._beta_index = ForwardVariableDef("beta_index", "vector[Ns]")
                if self._logparabola or self._pgamma:
                    self._E0_src = ForwardVariableDef("E0_src", "vector[Ns]")

            # True directions of point sources as a unit vector
            self._varpi = ForwardArrayDef("varpi", "unit_vector[3]", self._Ns_str)

            # Distance of sources in Mpc
            if self.sources.point_source:
                self._D = ForwardVariableDef("D", "vector[Ns]")

            # Diffuse sources have a redshift (default z=0) a spectral index,
            # and a grid over this index for the interpolation mentioned above
            if self.sources.diffuse:
                self._diff_index = ForwardVariableDef("diff_index", "real")
                # Energy range considered for diffuse astrophysical sources, defined at redshift z of shell
                self._Emin_diff = ForwardVariableDef("Emin_diff", "real")
                self._Emax_diff = ForwardVariableDef("Emax_diff", "real")
                self._F_diff = ForwardVariableDef("F_diff", "real")
                self._z = ForwardVariableDef("z", "vector[Ns+1]")
            else:
                self._z = ForwardVariableDef("z", "vector[Ns]")

            # Energy range that the detector should consider
            # Is influenced by parameterisation of energy resolution
            self._Emin = ForwardVariableDef("Emin", "real")
            self._Emax = ForwardVariableDef("Emax", "real")

            # For tracks, we specify Emin_det, and several parameters for the
            # rejection sampling, denoted by rs_...
            # Separate interpolation grids are also provided for all event types

            self._Emin_det = ForwardArrayDef("Emin_det", "real", ["[", self._Net, "]"])
            self._rs_bbpl_Eth = ForwardArrayDef(
                "rs_bbpl_Eth", "real", ["[", self._Net, "]"]
            )
            self._rs_bbpl_gamma1 = ForwardArrayDef(
                "rs_bbpl_gamma1", "real", ["[", self._Net, "]"]
            )
            self._rs_bbpl_gamma2_scale = ForwardArrayDef(
                "rs_bbpl_gamma2_scale", "real", ["[", self._Net, "]"]
            )
            # entry for each component
            self._rs_cvals = ForwardArrayDef(
                "rs_cvals", "real", ["[", self._Net, ",", Ns_string, "]"]
            )

            if self._force_N:
                self._forced_N = ForwardArrayDef(
                    "forced_N", "int", ["[", self._Net, ",", Ns_string, "]"]
                )
            # Get number of expected events per detector model and source component from python
            # rids us of all the interpolation needed here (done twice, first in python...)
            self._Nex_et = ForwardArrayDef(
                "Nex_et", "real", ["[", self._Net, ",", Ns_string, "]"]
            )

            if self.sources.atmospheric:
                self._F_atmo = ForwardVariableDef("F_atmo", "real")

                self._atmo_integ_val = ForwardArrayDef(
                    "atmo_integ_val", "real", ["[", self._Net, "]"]
                )

            # v_lim sets the edge of the uniform sampling on a sphere, for example,
            # for Northern skies only. See sphere_lim_rng.
            self._v_low = ForwardVariableDef("v_low", "real")
            self._v_high = ForwardVariableDef("v_high", "real")
            self._u_low = ForwardVariableDef("u_low", "real")
            self._u_high = ForwardVariableDef("u_high", "real")

            # If we have a circular ROI, set center point and radius
            try:
                ROIList.STACK[0]
            except IndexError:
                raise ValueError("An ROI is needed at this point.")
            if isinstance(ROIList.STACK[0], CircularROI):
                self._roi_center = ForwardArrayDef(
                    "roi_center", "unit_vector[3]", ["[", len(ROIList.STACK), "]"]
                )
                self._roi_radius = ForwardArrayDef(
                    "roi_radius", "real", ["[", len(ROIList.STACK), "]"]
                )

            # The observation time
            self._T = ForwardArrayDef("T", "real", ["[", self._Net, "]"])

    def _transformed_data(self):
        """
        Write the transformed data section of the Stan file.
        """

        with TransformedDataContext():
            if isinstance(ROIList.STACK[0], CircularROI):
                self._n_roi = ForwardVariableDef("n_roi", "int")
                self._n_roi << StringExpression(["num_elements(roi_radius)"])

            # Decide how many flux entries, F, we have
            if self.sources.diffuse and self.sources.atmospheric:
                self._F = ForwardVariableDef("F", "vector[Ns+2]")

                N_tot = "[Ns+2]"

            elif self.sources.diffuse or self.sources.atmospheric:
                self._F = ForwardVariableDef("F", "vector[Ns+1]")

                N_tot = "[Ns+1]"

            else:
                self._F = ForwardVariableDef("F", "vector[Ns]")

                N_tot = "[Ns]"

            self._et_stan = ForwardArrayDef("event_types", "int", ["[", self._Net, "]"])
            self._Net_stan = ForwardVariableDef("Net", "int")
            self._Net_stan << StringExpression(["size(event_types)"])

            idx = 1
            for et in self._event_types:
                self._et_stan[idx] << et.S
                idx += 1
            if not self._force_N:
                # Relative exposure weights of sources
                self._w_exposure = ForwardArrayDef(
                    "w_exposure", "simplex" + N_tot, ["[", self._Net, "]"]
                )

            # Expected number of events
            self._Nex = ForwardArrayDef("Nex", "real", ["[", self._Net, "]"])

            # Sampled number of events
            self._N_comp = ForwardArrayDef("N_comp", "int", ["[", self._Net, "]"])

            # Total flux
            self._Ftot = ForwardVariableDef("Ftot", "real")

            # Total flux from point sources
            self._F_src = ForwardVariableDef("Fs", "real")

            # Different assumptions for fractional flux from point sources
            # vs. other components
            self._f_arr_ = ForwardVariableDef("f_arr_", "real")
            self._f_arr_astro_ = ForwardVariableDef("f_arr_astro_", "real")
            self._f_det_ = ForwardVariableDef("f_det_", "real")
            self._f_det_astro_ = ForwardVariableDef("f_det_astro_", "real")

            # Expected number of events from different source components
            if self.sources.point_source:
                self._Nex_src_comp = ForwardArrayDef(
                    "Nex_src_comp", "real", ["[", self._Net, "]"]
                )
            self._Nex_src = InstantVariableDef("Nex_src", "real", [0])
            if self.sources.diffuse:
                self._Nex_diff_comp = ForwardArrayDef(
                    "Nex_diff_comp", "real", ["[", self._Net, "]"]
                )
                self._Nex_diff = ForwardVariableDef("Nex_diff", "real")
            if self.sources.atmospheric:
                self._Nex_atmo_comp = ForwardArrayDef(
                    "Nex_atmo_comp", "real", ["[", self._Net, "]"]
                )
                self._Nex_atmo = ForwardVariableDef("Nex_atmo", "real")

            # Sampled total number of events
            self._N = ForwardVariableDef("N", "int")

            self._F_src << 0.0

            if self.sources.point_source:
                with ForLoopContext(1, self._Ns, "k") as k:
                    self._F[k] << StringExpression(
                        [
                            self._L[k],
                            "/ (4 * pi() * pow(",
                            self._D[k],
                            " * ",
                            3.086e22,
                            ", 2))",
                        ]
                    )

                    if self._logparabola:
                        x_r = StringExpression(
                            [
                                "{",
                                self._src_index[k],
                                ",",
                                self._beta_index[k],
                                ",",
                                self._E0_src[k],
                                ",",
                                self._Emin_src[k],
                                ",",
                                self._Emax_src[k],
                                "}",
                            ]
                        )
                    elif self._pgamma:
                        x_r = StringExpression(
                            [
                                "{",
                                self._E0_src[k],
                                ",",
                                self._Emin_src[k],
                                ",",
                                self._Emax_src[k],
                                "}",
                            ]
                        )
                    if self._logparabola or self._pgamma:
                        theta = StringExpression(["{1.}"])
                        x_i = StringExpression(
                            [
                                "{",
                                0,
                                "}",
                            ]
                        )
                        StringExpression(
                            [
                                self._F[k],
                                "*=",
                                self._flux_conv(
                                    theta,
                                    x_r,
                                    x_i,
                                ),
                            ]
                        )
                    else:
                        StringExpression(
                            [
                                self._F[k],
                                "*=",
                                self._flux_conv(
                                    self._src_index[k],
                                    self._Emin_src[k],
                                    self._Emax_src[k],
                                ),
                            ]
                        )

                    # Sum point source flux
                    StringExpression([self._F_src, " += ", self._F[k]])

            if self.sources.diffuse:
                StringExpression("F[Ns+1]") << self._F_diff

            if self.sources.atmospheric and not self.sources.diffuse:
                StringExpression("F[Ns+1]") << self._F_atmo

            if self.sources.atmospheric and self.sources.diffuse:
                StringExpression("F[Ns+2]") << self._F_atmo

            # Calculate the exposure for diffuse/atmospheric sources
            # For cascades, we assume no atmo component

            with ForLoopContext(1, self._Net_stan, "i") as i:
                if not self._force_N:
                    if self.sources.point_source:
                        self._Nex_src_comp[i] << FunctionCall(
                            [self._Nex_et[i, 1 : self._Ns]], "sum"
                        )
                    if self.sources.diffuse:
                        self._Nex_diff_comp[i] << self._Nex_et[i, "Ns+1"]
                    if self.sources.atmospheric and not self.sources.diffuse:
                        self._Nex_atmo_comp[i] << self._Nex_et[i, "Ns+1"]
                    elif self.sources.atmospheric:
                        self._Nex_atmo_comp[i] << self._Nex_et[i, "Ns+2"]

                    # Get the relative exposure weights of all sources
                    # This will be used to sample the labels
                    # Also sample the number of events

                    self._Nex[i] << FunctionCall([self._Nex_et[i]], "sum")
                    self._w_exposure[i] << FunctionCall(
                        [self._Nex_et[i]], "get_exposure_weights_from_Nex_et"
                    )

                # If we passed the `force_N` keyword we ignore the exposure weighting
                # and sample a fixed amount of events
                if self._force_N:
                    self._N_comp[i] << StringExpression(
                        ["sum(", self._forced_N[i], ")"]
                    )

                # Else sample Poisson random variable with the expected number of events as parameter
                else:
                    self._N_comp[i] << StringExpression(
                        ["poisson_rng(", self._Nex[i], ")"]
                    )

            if self.sources.point_source:
                self._Nex_src << StringExpression(["sum(", self._Nex_src_comp, ")"])
            else:
                self._Nex_src << StringExpression(["0"])
            if self.sources.diffuse:
                self._Nex_diff << StringExpression(["sum(", self._Nex_diff_comp, ")"])
            if self.sources.atmospheric:
                self._Nex_atmo << StringExpression(["sum(", self._Nex_atmo_comp, ")"])
            self._N << StringExpression(["sum(", self._N_comp, ")"])

            # Calculate the fractional association for different assumptions
            if self.sources.diffuse and self.sources.atmospheric:
                self._Ftot << self._F_src + self._F_diff + self._F_atmo
                self._f_arr_astro_ << StringExpression(
                    [self._F_src, "/", self._F_src + self._F_diff]
                )
                self._f_det_ << self._Nex_src / (
                    self._Nex_src + self._Nex_diff + self._Nex_atmo
                )
                self._f_det_astro_ << self._Nex_src / (self._Nex_src + self._Nex_diff)

            elif self.sources.diffuse:
                self._Ftot << self._F_src + self._F_diff
                self._f_arr_astro_ << StringExpression(
                    [self._F_src, "/", self._F_src + self._F_diff]
                )
                self._f_det_ << self._Nex_src / (self._Nex_src + self._Nex_diff)
                self._f_det_astro_ << self._f_det_

            elif self.sources.atmospheric:
                self._Ftot << self._F_src + self._F_atmo
                self._f_arr_astro_ << 1.0
                self._f_det_ << self._Nex_src / (self._Nex_src + self._Nex_atmo)
                self._f_det_astro_ << 1.0

            else:
                self._Ftot << self._F_src
                self._f_arr_astro_ << 1.0
                self._f_det_ << 1.0
                self._f_det_astro_ << 1.0

            self._f_arr_ << StringExpression([self._F_src, "/", self._Ftot])

    def _generated_quantities(self):
        """
        Write the generated quantities section of the Stan file.
        """

        with GeneratedQuantitiesContext():
            self._loop_start = ForwardVariableDef("loop_start", "int")
            self._loop_end = ForwardVariableDef("loop_end", "int")
            # We redefine a bunch of variables from transformed data here, as we would
            # like to access them as outputs from the Stan simulation.
            self._f_arr = ForwardVariableDef("f_arr", "real")
            self._f_arr_astro = ForwardVariableDef("f_arr_astro", "real")
            self._f_det = ForwardVariableDef("f_det", "real")
            self._f_det_astro = ForwardVariableDef("f_det_astro", "real")
            self._f_arr << self._f_arr_
            self._f_arr_astro << self._f_arr_astro_
            self._f_det << self._f_det_
            self._f_det_astro << self._f_det_astro_

            self._N_str = ["[", self._N, "]"]

            # Lambda is the label of each event, matching to its true source
            self._lam = ForwardArrayDef("Lambda", "int", self._N_str)

            # omega is the true direction
            self._omega = ForwardVariableDef("omega", "unit_vector[3]")

            # Energies at the source, Earth and reconstructed in the detector
            self._Esrc = ForwardVariableDef("Esrc", "vector[N]")
            self._E = ForwardVariableDef("E", "vector[N]")
            self._Edet = ForwardVariableDef("Edet", "vector[N]")
            self._Emin_src_arr = ForwardVariableDef("Emin_src_arr", "real")
            self._Emax_src_arr = ForwardVariableDef("Emax_src_arr", "real")
            self._rs_bbpl_Eth_tmp = ForwardVariableDef("rs_bbpl_Eth_tmp", "real")

            # The cos(zenith) corresponding to each omega and assuming South Pole detector
            self._cosz = ForwardArrayDef("cosz", "real", self._N_str)

            # Detected directions as unit vectors for each event
            self._event = ForwardArrayDef("event", "unit_vector[3]", self._N_str)
            self._pre_event = ForwardVectorDef("pre_event", [5])

            # Variables for rejection sampling
            # Rejection sampling is based on a broken power law envelope,
            # and then the true source spectrum x effective area is sampled.
            # The tail of the envelope must not be steeper than that of the true
            # distribution. See e.g. https://bookdown.org/rdpeng/advstatcomp/rejection-sampling.html
            self._accept = ForwardVariableDef("accept", "int")
            self._detected = ForwardVariableDef("detected", "int")
            self._ntrials = ForwardVariableDef("ntrials", "int")
            self._u_samp = ForwardVariableDef("u_samp", "real")
            self._aeff_factor = ForwardVariableDef("aeff_factor", "real")
            self._src_factor = ForwardVariableDef("src_factor", "real")
            self._f_value = ForwardVariableDef("f_value", "real")
            self._g_value = ForwardVariableDef("g_value", "real")
            self._c_value = ForwardVariableDef("c_value", "real")
            self._idx_cosz = ForwardVariableDef("idx_cosz", "int")
            self._gamma2 = ForwardVariableDef("gamma2", "real")

            # Label for the currently sampled source component, starts obviously with 1
            # Is reset to 1 when using multiple detector models and sampling moves on to the next
            if self._force_N:
                self._currently_sampling = ForwardVariableDef(
                    "currently_sampling",
                    "int",
                )

            self._event_type = ForwardVariableDef("event_type", "vector[N]")

            # Kappa is the shape param of the vMF used in sampling the detected direction
            self._kappa = ForwardVariableDef("kappa", "vector[N]")

            # Here we start the sampling, first of tracks and then cascades, all other detector mdoels...
            with ForLoopContext(1, self._Net_stan, "j") as j:
                if self._force_N:
                    # Counts the events sampled of each source components
                    # Counter is reset after the required number is reached
                    self._forced_N_sampled = InstantVariableDef("sampled_N", "int", [0])
                with IfBlockContext([j, " == ", 1]):
                    self._loop_start << 1
                with ElseBlockContext():
                    self._loop_start << StringExpression(
                        ["sum(", self._N_comp[1:"j - 1"], ") + 1"]
                    )
                self._loop_end << StringExpression(["sum(", self._N_comp[1:j], ")"])

                if self._force_N:
                    # currently sampling first source component when starting with a detector model
                    self._currently_sampling << 1

                # For each event, we rejection sample the true energy and direction
                # and then directly sample the detected properties
                with ForLoopContext(self._loop_start, self._loop_end, "i") as i:
                    # Sample source label
                    # If we force N, proceed to sample the given number of events for each source
                    if self._force_N:
                        # If we have not sampled enough events,
                        # get the source label and increase the sampled events by 1
                        with WhileLoopContext([1]):
                            with IfBlockContext(
                                [
                                    self._forced_N_sampled,
                                    " < ",
                                    self._forced_N[j, self._currently_sampling],
                                ]
                            ):
                                self._lam[i] << self._currently_sampling
                                self._event_type[i] << self._et_stan[j]
                                StringExpression([self._forced_N_sampled, " += 1"])
                                StringExpression(["break"])
                            with ElseBlockContext():
                                StringExpression([self._currently_sampling, " += 1"])
                                self._forced_N_sampled << 0

                    # Otherwise, use the exposure weights to determine the event label
                    else:
                        self._lam[i] << FunctionCall(
                            [self._w_exposure[j]], "categorical_rng"
                        )
                        self._event_type[i] << self._et_stan[j]

                    # Reset rejection
                    self._accept << 0
                    self._detected << 0
                    self._ntrials << 0

                    # While not accepted
                    with WhileLoopContext([StringExpression([self._accept != 1])]):
                        # Used for rejection sampling
                        self._u_samp << FunctionCall([0.0, 1.0], "uniform_rng")

                        # For point sources, the true direction is specified
                        with IfBlockContext(
                            [StringExpression([self._lam[i], " <= ", self._Ns])]
                        ):
                            self._omega << self._varpi[self._lam[i]]

                        # Otherwise, sample uniformly over sphere, considering v_lim
                        if self.sources.atmospheric and not self.sources.diffuse:
                            with ElseIfBlockContext(
                                [StringExpression([self._lam[i], " == ", self._Ns + 1])]
                            ):
                                self._omega << FunctionCall(
                                    [
                                        1,
                                        self._v_low,
                                        self._v_high,
                                        self._u_low,
                                        self._u_high,
                                    ],
                                    "sphere_lim_rng",
                                )

                        elif self.sources.diffuse:
                            with ElseIfBlockContext(
                                [StringExpression([self._lam[i], " == ", self._Ns + 1])]
                            ):
                                self._omega << FunctionCall(
                                    [
                                        1,
                                        self._v_low,
                                        self._v_high,
                                        self._u_low,
                                        self._u_high,
                                    ],
                                    "sphere_lim_rng",
                                )

                        if self.sources.atmospheric and self._sources.diffuse:
                            with ElseIfBlockContext(
                                [StringExpression([self._lam[i], " == ", self._Ns + 2])]
                            ):
                                self._omega << FunctionCall(
                                    [
                                        1,
                                        self._v_low,
                                        self._v_high,
                                        self._u_low,
                                        self._u_high,
                                    ],
                                    "sphere_lim_rng",
                                )

                        self._cosz[i] << FunctionCall(
                            [FunctionCall([self._omega], "omega_to_zenith")], "cos"
                        )

                        # Calculate the envelope for rejection sampling and the shape of
                        # the source spectrum for the various source components
                        if self.sources.point_source:
                            # TODO continue here
                            with IfBlockContext(
                                [StringExpression([self._lam[i], " <= ", self._Ns])]
                            ):

                                # The shape of the envelope to use depends on the
                                # source spectrum. This is to make things more efficient.
                                (
                                    self._gamma2
                                    << self._rs_bbpl_gamma2_scale[j]
                                    - self._src_index[self._lam[i]]
                                )

                                # Handle energy thresholds
                                # 3 cases:
                                # Emin < Eth and Emax > Eth - use broken pl
                                # Emin < Eth and Emax <= Eth - use pl
                                # Emin >= Eth and Emax > Eth - use pl
                                self._Emin_src_arr << (self._Emin_src[self._lam[i]])
                                self._Emax_src_arr << (self._Emax_src[self._lam[i]])

                                with IfBlockContext(
                                    [
                                        StringExpression(
                                            [
                                                "(",
                                                self._Emin_src_arr,
                                                "<",
                                                self._rs_bbpl_Eth[j],
                                                ") && (",
                                                self._Emax_src_arr,
                                                ">",
                                                self._rs_bbpl_Eth[j],
                                                ")",
                                            ]
                                        )
                                    ]
                                ):
                                    # Sample a true energy from the envelope function
                                    self._E[i] << FunctionCall(
                                        [
                                            self._Emin_src_arr,
                                            self._rs_bbpl_Eth[j],
                                            self._Emax_src_arr,
                                            self._rs_bbpl_gamma1[j],
                                            self._gamma2,
                                        ],
                                        "bbpl_rng",
                                    )

                                    # Also store the value of the envelope function at this true energy
                                    self._g_value << FunctionCall(
                                        [
                                            self._E[i],
                                            self._Emin_src_arr,
                                            self._rs_bbpl_Eth[j],
                                            self._Emax_src_arr,
                                            self._rs_bbpl_gamma1[j],
                                            self._gamma2,
                                        ],
                                        "bbpl_pdf",
                                    )

                                with ElseIfBlockContext(
                                    [
                                        StringExpression(
                                            [
                                                "(",
                                                self._Emin_src_arr,
                                                "<",
                                                self._rs_bbpl_Eth[j],
                                                ") && (",
                                                self._Emax_src_arr,
                                                "<=",
                                                self._rs_bbpl_Eth[j],
                                                ")",
                                            ]
                                        )
                                    ]
                                ):
                                    # Sample a true energy from the envelope function
                                    # Modify to power law
                                    (
                                        self._rs_bbpl_Eth_tmp
                                        << self._Emin_src_arr
                                        + (self._Emax_src_arr - self._Emin_src_arr) / 2
                                    )
                                    self._E[i] << FunctionCall(
                                        [
                                            self._Emin_src_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Emax_src_arr,
                                            self._rs_bbpl_gamma1[j],
                                            self._rs_bbpl_gamma1[j],
                                        ],
                                        "bbpl_rng",
                                    )

                                    # Also store the value of the envelope function at this true energy
                                    self._g_value << FunctionCall(
                                        [
                                            self._E[i],
                                            self._Emin_src_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Emax_src_arr,
                                            self._rs_bbpl_gamma1[j],
                                            self._rs_bbpl_gamma1[j],
                                        ],
                                        "bbpl_pdf",
                                    )

                                with ElseIfBlockContext(
                                    [
                                        StringExpression(
                                            [
                                                "(",
                                                self._Emin_src_arr,
                                                ">=",
                                                self._rs_bbpl_Eth[j],
                                                ") && (",
                                                self._Emax_src_arr,
                                                ">",
                                                self._rs_bbpl_Eth[j],
                                                ")",
                                            ]
                                        )
                                    ]
                                ):
                                    # Sample a true energy from the envelope function
                                    # Modify to power law
                                    (
                                        self._rs_bbpl_Eth_tmp
                                        << self._Emin_src_arr
                                        + (self._Emax_src_arr - self._Emin_src_arr) / 2
                                    )
                                    self._E[i] << FunctionCall(
                                        [
                                            self._Emin_src_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Emax_src_arr,
                                            self._gamma2,
                                            self._gamma2,
                                        ],
                                        "bbpl_rng",
                                    )

                                    # Also store the value of the envelope function at this true energy
                                    self._g_value << FunctionCall(
                                        [
                                            self._E[i],
                                            self._Emin_src_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Emax_src_arr,
                                            self._gamma2,
                                            self._gamma2,
                                        ],
                                        "bbpl_pdf",
                                    )

                                # Store the value of the source PDF at this energy
                                if self._logparabola:
                                    x_r = StringExpression(
                                        [
                                            "{",
                                            self._src_index[self._lam[i]],
                                            ",",
                                            self._beta_index[self._lam[i]],
                                            ",",
                                            self._E0_src[self._lam[i]],
                                            ",",
                                            self._Emin_src[self._lam[i]],
                                            ",",
                                            self._Emax_src[self._lam[i]],
                                            "}",
                                        ]
                                    )
                                elif self._pgamma:
                                    x_r = StringExpression(
                                        [
                                            "{",
                                            self._E0_src[self._lam[i]],
                                            ",",
                                            self._Emin_src[self._lam[i]],
                                            ",",
                                            self._Emax_src[self._lam[i]],
                                            "}",
                                        ]
                                    )
                                if self._logparabola or self._pgamma:
                                    theta = StringExpression(
                                        [
                                            "{1.}",
                                        ]
                                    )
                                    x_i = StringExpression(
                                        [
                                            "{",
                                            0,
                                            "}",
                                        ]
                                    )
                                    self._src_factor << self._src_spectrum_lpdf(
                                        self._E[i],
                                        theta,
                                        x_r,
                                        x_i,
                                    )
                                else:
                                    self._src_factor << self._src_spectrum_lpdf(
                                        self._E[i],
                                        self._src_index[self._lam[i]],
                                        self._Emin_src[self._lam[i]],
                                        self._Emax_src[self._lam[i]],
                                    )

                                # It is log, to take the exp()
                                self._src_factor << FunctionCall(
                                    [self._src_factor], "exp"
                                )

                                # Account for energy redshift losses
                                self._Esrc[i] << DetectorFrame.stan_to_src(
                                    self._E[i], self._z, self._lam, i
                                )

                        # Treat the atmospheric and diffuse components similarly
                        if self.sources.atmospheric and not self.sources.diffuse:
                            with IfBlockContext(
                                [StringExpression([self._lam[i], " == ", self._Ns + 1])]
                            ):
                                # Assume fixed index of ~3.6 for atmo to get reasonable
                                # envelope function
                                self._gamma2 << self._rs_bbpl_gamma2_scale[j] - 3.6

                                # Handle energy thresholds
                                # 3 cases:
                                # Emin < Eth and Emax > Eth - use broken pl
                                # Emin < Eth and Emax <= Eth - use pl
                                # Emin >= Eth and Emax > Eth - use pl
                                self._Emin_src_arr << self._Emin
                                self._Emax_src_arr << self._Emax
                                with IfBlockContext(
                                    [
                                        StringExpression(
                                            [
                                                "(",
                                                self._Emin_src_arr,
                                                "<",
                                                self._rs_bbpl_Eth[j],
                                                ") && (",
                                                self._Emax_src_arr,
                                                ">",
                                                self._rs_bbpl_Eth[j],
                                                ")",
                                            ]
                                        )
                                    ]
                                ):
                                    # Sample a true energy from the envelope function
                                    self._E[i] << FunctionCall(
                                        [
                                            self._Emin_src_arr,
                                            self._rs_bbpl_Eth[j],
                                            self._Emax_src_arr,
                                            self._rs_bbpl_gamma1[j],
                                            self._gamma2,
                                        ],
                                        "bbpl_rng",
                                    )

                                    # Also store the value of the envelope function at this true energy
                                    self._g_value << FunctionCall(
                                        [
                                            self._E[i],
                                            self._Emin_src_arr,
                                            self._rs_bbpl_Eth[j],
                                            self._Emax_src_arr,
                                            self._rs_bbpl_gamma1[j],
                                            self._gamma2,
                                        ],
                                        "bbpl_pdf",
                                    )

                                with ElseIfBlockContext(
                                    [
                                        StringExpression(
                                            [
                                                "(",
                                                self._Emin_src_arr,
                                                "<",
                                                self._rs_bbpl_Eth[j],
                                                ") && (",
                                                self._Emax_src_arr,
                                                "<=",
                                                self._rs_bbpl_Eth[j],
                                                ")",
                                            ]
                                        )
                                    ]
                                ):
                                    # Sample a true energy from the envelope function
                                    # Modify to power law
                                    (
                                        self._rs_bbpl_Eth_tmp
                                        << self._Emin_src_arr
                                        + (self._Emax_src_arr - self._Emin_src_arr) / 2
                                    )
                                    self._E[i] << FunctionCall(
                                        [
                                            self._Emin_src_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Emax_src_arr,
                                            self._rs_bbpl_gamma1[j],
                                            self._rs_bbpl_gamma1[j],
                                        ],
                                        "bbpl_rng",
                                    )

                                    # Also store the value of the envelope function at this true energy
                                    self._g_value << FunctionCall(
                                        [
                                            self._E[i],
                                            self._Emin_src_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Emax_src_arr,
                                            self._rs_bbpl_gamma1[j],
                                            self._rs_bbpl_gamma1[j],
                                        ],
                                        "bbpl_pdf",
                                    )

                                with ElseIfBlockContext(
                                    [
                                        StringExpression(
                                            [
                                                "(",
                                                self._Emin_src_arr,
                                                ">=",
                                                self._rs_bbpl_Eth[j],
                                                ") && (",
                                                self._Emax_src_arr,
                                                ">",
                                                self._rs_bbpl_Eth[j],
                                                ")",
                                            ]
                                        )
                                    ]
                                ):
                                    # Sample a true energy from the envelope function
                                    # Modify to power law
                                    (
                                        self._rs_bbpl_Eth_tmp
                                        << self._Emin_src_arr
                                        + (self._Emax_src_arr - self._Emin_src_arr) / 2
                                    )
                                    self._E[i] << FunctionCall(
                                        [
                                            self._Emin_src_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Emax_src_arr,
                                            self._gamma2,
                                            self._gamma2,
                                        ],
                                        "bbpl_rng",
                                    )

                                    # Also store the value of the envelope function at this true energy
                                    self._g_value << FunctionCall(
                                        [
                                            self._E[i],
                                            self._Emin_src_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Emax_src_arr,
                                            self._gamma2,
                                            self._gamma2,
                                        ],
                                        "bbpl_pdf",
                                    )

                                (
                                    self._src_factor
                                    << self._atmo_flux(self._E[i], self._omega)
                                    / self._F_atmo
                                )  # Normalise
                                self._Esrc[i] << self._E[i]

                        elif self.sources.diffuse:
                            with IfBlockContext(
                                [StringExpression([self._lam[i], " == ", self._Ns + 1])]
                            ):
                                (
                                    self._gamma2
                                    << self._rs_bbpl_gamma2_scale[j] - self._diff_index
                                )

                                # Handle energy thresholds
                                # 3 cases:
                                # Emin < Eth and Emax > Eth - use broken pl
                                # Emin < Eth and Emax <= Eth - use pl
                                # Emin >= Eth and Emax > Eth - use pl

                                self._Emin_src_arr << self._Emin_diff
                                self._Emax_src_arr << self._Emax_diff

                                with IfBlockContext(
                                    [
                                        StringExpression(
                                            [
                                                "(",
                                                self._Emin_src_arr,
                                                "<",
                                                self._rs_bbpl_Eth[j],
                                                ") && (",
                                                self._Emax_src_arr,
                                                ">",
                                                self._rs_bbpl_Eth[j],
                                                ")",
                                            ]
                                        )
                                    ]
                                ):
                                    # Sample a true energy from the envelope function
                                    self._E[i] << FunctionCall(
                                        [
                                            self._Emin_src_arr,
                                            self._rs_bbpl_Eth[j],
                                            self._Emax_src_arr,
                                            self._rs_bbpl_gamma1[j],
                                            self._gamma2,
                                        ],
                                        "bbpl_rng",
                                    )

                                    # Also store the value of the envelope function at this true energy
                                    self._g_value << FunctionCall(
                                        [
                                            self._E[i],
                                            self._Emin_src_arr,
                                            self._rs_bbpl_Eth[j],
                                            self._Emax_src_arr,
                                            self._rs_bbpl_gamma1[j],
                                            self._gamma2,
                                        ],
                                        "bbpl_pdf",
                                    )

                                with ElseIfBlockContext(
                                    [
                                        StringExpression(
                                            [
                                                "(",
                                                self._Emin_src_arr,
                                                "<",
                                                self._rs_bbpl_Eth[j],
                                                ") && (",
                                                self._Emax_src_arr,
                                                "<=",
                                                self._rs_bbpl_Eth[j],
                                                ")",
                                            ]
                                        )
                                    ]
                                ):
                                    # Sample a true energy from the envelope function
                                    # Modify to power law
                                    (
                                        self._rs_bbpl_Eth_tmp
                                        << self._Emin_src_arr
                                        + (self._Emax_src_arr - self._Emin_src_arr) / 2
                                    )
                                    self._E[i] << FunctionCall(
                                        [
                                            self._Emin_src_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Emax_src_arr,
                                            self._rs_bbpl_gamma1[j],
                                            self._rs_bbpl_gamma1[j],
                                        ],
                                        "bbpl_rng",
                                    )

                                    # Also store the value of the envelope function at this true energy
                                    self._g_value << FunctionCall(
                                        [
                                            self._E[i],
                                            self._Emin_src_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Emax_src_arr,
                                            self._rs_bbpl_gamma1[j],
                                            self._rs_bbpl_gamma1[j],
                                        ],
                                        "bbpl_pdf",
                                    )

                                with ElseIfBlockContext(
                                    [
                                        StringExpression(
                                            [
                                                "(",
                                                self._Emin_src_arr,
                                                ">=",
                                                self._rs_bbpl_Eth[j],
                                                ") && (",
                                                self._Emax_src_arr,
                                                ">",
                                                self._rs_bbpl_Eth[j],
                                                ")",
                                            ]
                                        )
                                    ]
                                ):
                                    # Sample a true energy from the envelope function
                                    # Modify to power law
                                    (
                                        self._rs_bbpl_Eth_tmp
                                        << self._Emin_src_arr
                                        + (self._Emax_src_arr - self._Emin_src_arr) / 2
                                    )
                                    self._E[i] << FunctionCall(
                                        [
                                            self._Emin_src_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Emax_src_arr,
                                            self._gamma2,
                                            self._gamma2,
                                        ],
                                        "bbpl_rng",
                                    )

                                    # Also store the value of the envelope function at this true energy
                                    self._g_value << FunctionCall(
                                        [
                                            self._E[i],
                                            self._Emin_src_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Emax_src_arr,
                                            self._gamma2,
                                            self._gamma2,
                                        ],
                                        "bbpl_pdf",
                                    )

                                self._src_factor << self._diff_spectrum_lpdf(
                                    self._E[i],
                                    self._diff_index,
                                    self._diff_frame.stan_to_det(
                                        self._Emin_diff, self._z, self._lam, i
                                    ),
                                    self._diff_frame.stan_to_det(
                                        self._Emax_diff, self._z, self._lam, i
                                    ),
                                )
                                self._src_factor << FunctionCall(
                                    [self._src_factor], "exp"
                                )

                                self._Esrc[i] << DetectorFrame.stan_to_src(
                                    self._E[i], self._z, self._lam[i]
                                )

                        if self.sources.diffuse and self.sources.atmospheric:
                            with IfBlockContext(
                                [StringExpression([self._lam[i], " == ", self._Ns + 2])]
                            ):
                                self._gamma2 << self._rs_bbpl_gamma2_scale[j] - 3.6

                                # Handle energy thresholds
                                # 3 cases:
                                # Emin < Eth and Emax > Eth - use broken pl
                                # Emin < Eth and Emax <= Eth - use pl
                                # Emin >= Eth and Emax > Eth - use pl
                                self._Emin_src_arr << self._Emin
                                self._Emax_src_arr << self._Emax

                                with IfBlockContext(
                                    [
                                        StringExpression(
                                            [
                                                "(",
                                                self._Emin_src_arr,
                                                "<",
                                                self._rs_bbpl_Eth[j],
                                                ") && (",
                                                self._Emax_src_arr,
                                                ">",
                                                self._rs_bbpl_Eth[j],
                                                ")",
                                            ]
                                        )
                                    ]
                                ):
                                    # Sample a true energy from the envelope function
                                    self._E[i] << FunctionCall(
                                        [
                                            self._Emin_src_arr,
                                            self._rs_bbpl_Eth[j],
                                            self._Emax_src_arr,
                                            self._rs_bbpl_gamma1[j],
                                            self._gamma2,
                                        ],
                                        "bbpl_rng",
                                    )

                                    # Also store the value of the envelope function at this true energy
                                    self._g_value << FunctionCall(
                                        [
                                            self._E[i],
                                            self._Emin_src_arr,
                                            self._rs_bbpl_Eth[j],
                                            self._Emax_src_arr,
                                            self._rs_bbpl_gamma1[j],
                                            self._gamma2,
                                        ],
                                        "bbpl_pdf",
                                    )

                                with ElseIfBlockContext(
                                    [
                                        StringExpression(
                                            [
                                                "(",
                                                self._Emin_src_arr,
                                                "<",
                                                self._rs_bbpl_Eth[j],
                                                ") && (",
                                                self._Emax_src_arr,
                                                "<=",
                                                self._rs_bbpl_Eth[j],
                                                ")",
                                            ]
                                        )
                                    ]
                                ):
                                    # Sample a true energy from the envelope function
                                    # Modify to power law
                                    (
                                        self._rs_bbpl_Eth_tmp
                                        << self._Emin_src_arr
                                        + (self._Emax_src_arr - self._Emin_src_arr) / 2
                                    )
                                    self._E[i] << FunctionCall(
                                        [
                                            self._Emin_src_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Emax_src_arr,
                                            self._rs_bbpl_gamma1[j],
                                            self._rs_bbpl_gamma1[j],
                                        ],
                                        "bbpl_rng",
                                    )

                                    # Also store the value of the envelope function at this true energy
                                    self._g_value << FunctionCall(
                                        [
                                            self._E[i],
                                            self._Emin_src_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Emax_src_arr,
                                            self._rs_bbpl_gamma1[j],
                                            self._rs_bbpl_gamma1[j],
                                        ],
                                        "bbpl_pdf",
                                    )

                                with ElseIfBlockContext(
                                    [
                                        StringExpression(
                                            [
                                                "(",
                                                self._Emin_src_arr,
                                                ">=",
                                                self._rs_bbpl_Eth[j],
                                                ") && (",
                                                self._Emax_src_arr,
                                                ">",
                                                self._rs_bbpl_Eth[j],
                                                ")",
                                            ]
                                        )
                                    ]
                                ):
                                    # Sample a true energy from the envelope function
                                    # Modify to power law
                                    (
                                        self._rs_bbpl_Eth_tmp
                                        << self._Emin_src_arr
                                        + (self._Emax_src_arr - self._Emin_src_arr) / 2
                                    )
                                    self._E[i] << FunctionCall(
                                        [
                                            self._Emin_src_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Emax_src_arr,
                                            self._gamma2,
                                            self._gamma2,
                                        ],
                                        "bbpl_rng",
                                    )

                                    # Also store the value of the envelope function at this true energy
                                    self._g_value << FunctionCall(
                                        [
                                            self._E[i],
                                            self._Emin_src_arr,
                                            self._rs_bbpl_Eth_tmp,
                                            self._Emax_src_arr,
                                            self._gamma2,
                                            self._gamma2,
                                        ],
                                        "bbpl_pdf",
                                    )

                                (
                                    self._src_factor
                                    << self._atmo_flux(self._E[i], self._omega)
                                    / self._F_atmo
                                )  # Normalise
                                self._Esrc[i] << self._E[i]

                        for c, event_type in enumerate(self._event_types):
                            with IfBlockContext(
                                [
                                    self._event_type[i],
                                    " == ",
                                    event_type.S,
                                ]
                            ):
                                self._aeff_factor << self._dm[
                                    event_type
                                ].effective_area(self._E[i], self._omega)

                        # Calculate quantities for rejection sampling
                        # Value of the distribution that we want to sample from
                        self._f_value << self._src_factor * self._aeff_factor

                        # Find the precomputed c_value for this source and cosz
                        # self._idx_cosz << FunctionCall(
                        #    [self._cosz[i], self._rs_cosz_bin_edges_t], "binary_search"
                        # )
                        self._c_value << self._rs_cvals[j][self._lam[i]]
                        # Debugging when sampling gets stuck
                        StringExpression([self._ntrials, " += ", 1])

                        with IfBlockContext(
                            [StringExpression([self._ntrials, "< 1000000"])]
                        ):
                            # Here is the rejection sampling criterion
                            with IfBlockContext(
                                [
                                    StringExpression(
                                        [
                                            self._u_samp,
                                            " < ",
                                            self._f_value
                                            / (self._c_value * self._g_value),
                                        ]
                                    )
                                ]
                            ):
                                for c, event_type in enumerate(self._event_types):
                                    with IfBlockContext(
                                        [
                                            self._event_type[i],
                                            " == ",
                                            event_type.S,
                                        ]
                                    ):
                                        self._pre_event << self._dm[event_type](
                                            self._E[i],
                                            self._omega,
                                        )

                                self._Edet[i] << self._pre_event[1]
                                if self.sources.point_source:
                                    with IfBlockContext(
                                        [
                                            StringExpression(
                                                [self._lam[i], " < ", self._Ns + 1]
                                            )
                                        ]
                                    ):
                                        self._event[i] << self._pre_event[2:4]
                                    with ElseBlockContext():
                                        self._event[i] << self._omega
                                else:
                                    self._event[i] << self._omega
                                self._kappa[i] << self._pre_event[5]
                                self._detected << 0
                                if isinstance(ROIList.STACK[0], CircularROI):
                                    with ForLoopContext(1, self._n_roi, "n") as n:
                                        with IfBlockContext(
                                            [
                                                "ang_sep(event[i], roi_center[",
                                                n,
                                                "]) <= roi_radius[n]",
                                            ]
                                        ):
                                            self._detected << 1
                                            StringExpression(["break"])
                                else:
                                    self._detected << 1

                                # Insert condition for the sample to be inside the ROI,
                                # should also apply to the rectangular ROI as events
                                # can be recontructed outside at large ang err

                            with ElseBlockContext():
                                self._detected << 0

                            # Also apply the threshold on possible detected energies
                            with IfBlockContext(
                                [
                                    StringExpression(
                                        [
                                            "(",
                                            self._detected == 1,
                                            ") && (",
                                            self._Edet[i],
                                            " >= ",
                                            self._Emin_det[j],
                                            ")",
                                        ]
                                    )
                                ]
                            ):
                                # Accept this sample!
                                self._accept << 1
                            with ElseBlockContext():
                                self._accept << 0

                        # Debugging
                        with ElseBlockContext():
                            # If sampler gets stuck, print a warning message and move on.
                            self._accept << 1

                            StringExpression(
                                ['print("problem component: ", ', self._lam[i], ");\n"]
                            )
