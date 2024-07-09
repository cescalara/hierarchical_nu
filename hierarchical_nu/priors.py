from abc import ABCMeta, abstractmethod
from typing import Union, Iterable
import astropy.units as u
from astropy.units.core import UnitConversionError
import numpy as np
from scipy import stats
import h5py


class PriorDistribution(metaclass=ABCMeta):
    """
    Abstract base class for prior
    distributions.
    """

    def __init__(self, name="prior"):
        self._name = name

    @property
    def name(self):
        return self._name

    @abstractmethod
    def pdf(self, x):
        pass

    def pdf_logspace(self, x):
        return self.pdf(x) * x * np.log(10)

    @abstractmethod
    def sample(self, N):
        pass

    @abstractmethod
    def to_dict(self):
        pass


class NormalPrior(PriorDistribution):
    def __init__(self, name="normal", mu=0.0, sigma=1.0):
        super().__init__(name=name)

        self._mu = mu
        self._sigma = sigma

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, val: float):
        self._mu = val

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, val: float):
        self._sigma = val

    def pdf(self, x):
        return stats.norm(loc=self._mu, scale=self._sigma).pdf(x)

    def sample(self, N):
        return stats.norm(loc=self._mu, scale=self._sigma).rvs(N)

    def to_dict(self, units):
        prior_dict = {}

        prior_dict["name"] = self._name

        prior_dict["mu"] = self._mu

        prior_dict["sigma"] = self._sigma

        prior_dict["units"] = units

        return prior_dict


class LogNormalPrior(PriorDistribution):
    def __init__(self, name="lognormal", mu=1.0, sigma=1.0):
        super().__init__(name=name)

        self._mu = mu
        self._sigma = sigma

    def pdf(self, x):
        return stats.lognorm(scale=np.exp(self._mu), s=self._sigma).pdf(x)

    def sample(self, N):
        return stats.lognorm(scale=np.exp(self._mu), s=self._sigma).rvs(N)

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, val: float):
        self._mu = val

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, val: float):
        self._sigma = val

    def to_dict(self, units):
        prior_dict = {}

        prior_dict["name"] = self._name

        prior_dict["mu"] = self._mu

        prior_dict["sigma"] = self._sigma

        prior_dict["units"] = units

        return prior_dict


class ParetoPrior(PriorDistribution):
    def __init__(self, name="pareto", xmin=1.0, alpha=1.0):
        super().__init__(name=name)

        self._xmin = xmin
        self._alpha = alpha

    @property
    def xmin(self):
        return self._xmin

    @xmin.setter
    def xmin(self, val: float):
        self._xmin = val

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, val: float):
        self._alpha = val

    def pdf(self, x):
        return stats.pareto(b=self._alpha).pdf(x)

    def sample(self, N):
        return stats.pareto(b=self._alpha).rvs(N) * self._xmin

    def to_dict(self, units):
        prior_dict = {}

        prior_dict["name"] = self._name

        prior_dict["xmin"] = self._xmin

        prior_dict["alpha"] = self._alpha

        prior_dict["units"] = units

        return prior_dict


class PriorDictHandler:
    """
    Class to translate priors from and to dictionaries.
    """

    @classmethod
    def from_dict(cls, prior_dict):
        # Translate "key" into Lumi, Flux or Index
        translate = {
            "L": LuminosityPrior,
            "F_diff": FluxPrior,
            "F_atmo": FluxPrior,
            "src_index": IndexPrior,
            "beta_index": IndexPrior,
            "diff_index": IndexPrior,
            "E0_src": EnergyPrior,
        }
        prior_name = prior_dict["name"]
        prior = translate[prior_dict["quantity"]]
        units = u.Unit(prior_dict["units"])
        if prior_name == "pareto":
            xmin = prior_dict["xmin"]
            alpha = prior_dict["alpha"]
            return prior(ParetoPrior, xmin=xmin * units, alpha=alpha)
        mu = prior_dict["mu"]
        sigma = prior_dict["sigma"]
        if prior_name == "normal":
            sigma *= units
            mu *= units
            return prior(NormalPrior, mu=mu, sigma=sigma)
        elif prior_name == "lognormal":
            return prior(LogNormalPrior, mu=np.exp(mu) * units, sigma=sigma)


class UnitPrior:
    """
    Class to handle priors on parameters carrying a unit.
    """

    def __init__(self, name, **kwargs):
        if name == ParetoPrior:
            xmin = kwargs.get("xmin")
            alpha = kwargs.get("alpha")
            units = kwargs.get("units")
            self._units = units
            self._prior = name(xmin=xmin.to_value(units), alpha=alpha)

        else:
            mu = kwargs.get("mu")
            sigma = kwargs.get("sigma")
            units = kwargs.get("units")
            self._units = units

            if name == LogNormalPrior:
                # Check that sigma is a real number
                try:
                    sigma.to_value(1)
                except AttributeError:
                    # Is raised if sigma has no astropy unit attached
                    pass
                # else UnitConversionError is raised
                self._prior = name(mu=np.log(mu.to_value(units)), sigma=sigma)
            elif name == NormalPrior:
                self._prior = name(mu=mu.to_value(units), sigma=sigma.to_value(units))
            elif name == ParetoPrior:
                pass

    def pdf(self, x):
        return self._prior.pdf(x.to_value(self._units))

    def pdf_logspace(self, x):
        return self._prior.pdf_logspace(x.to_value(self._units))

    @property
    def mu(self):
        if isinstance(self._prior, NormalPrior):
            return self._prior.mu * self._units
        else:
            return self._prior.mu

    @mu.setter
    def mu(self, val):
        if isinstance(self._prior, NormalPrior):
            self._prior.mu = val.to_value(self._units)
        else:
            self._prior.mu = val

    @property
    def sigma(self):
        if isinstance(self._prior, NormalPrior):
            return self._prior.sigma * self._units
        else:
            return self._prior.sigma

    @sigma.setter
    def sigma(self, val):
        if isinstance(self._prior, NormalPrior):
            self._prior.sigma = val.to_value(self._units)
        else:
            self._prior.sigma = val

    @property
    def xmin(self):
        return self._prior.xmin * self._units

    @xmin.setter
    def xmin(self, val):
        if isinstance(self._prior, ParetoPrior):
            self._prior.xmin = val.to_value(self._units)

    @property
    def alpha(self):
        return self._prior.alpha

    @alpha.setter
    def alpha(self, val):
        self._prior.alpha = val

    @property
    def name(self):
        return self._prior.name

    def to_dict(self, units):
        return self._prior.to_dict(units)


class UnitlessPrior:
    """
    Class to handle priors on unitless parameters.
    """

    def __init__(self, name, **kwargs):
        if name == ParetoPrior:
            alpha = kwargs.get("alpha")
            xmin = kwargs.get("xmin")
            self._prior = name(xmin=xmin, alpha=alpha)

        else:
            mu = kwargs.get("mu")
            sigma = kwargs.get("sigma")
            if name == LogNormalPrior:
                self._prior = name(mu=np.log(mu), sigma=sigma)
            elif name == NormalPrior:
                self._prior = name(mu=mu, sigma=sigma)

    @property
    def mu(self):
        return self._prior.mu

    @mu.setter
    def mu(self, val: float):
        self._prior.mu = val

    @property
    def sigma(self):
        return self._prior.sigma

    @sigma.setter
    def sigma(self, val: float):
        self._prior.sigma = val

    @property
    def xmin(self):
        return self._prior.xmin

    @xmin.setter
    def xmin(self, val: float):
        self._prior.xmin = val

    @property
    def alpha(self):
        return self._prior.alpha

    @alpha.setter
    def alpha(self, val: float):
        self._prior.alpha = val

    @property
    def name(self):
        return self._prior.name

    def pdf(self, x):
        return self._prior.pdf(x)

    def pdf_logspace(self, x):
        return self._prior.pdf_logspace(x)

    # Poor man's conditional inheritance
    # copied from https://stackoverflow.com/a/65754897
    # only in case someone tries to do self._mu or similar, then the private attribute should be accessed
    # def __getattr__(self, name):
    #     return self._prior.__getattribute__(name)

    def to_dict(self, units):
        return self._prior.to_dict(units)


class LuminosityPrior(UnitPrior):
    UNITS = u.GeV / u.s
    UNITS_STRING = UNITS.to_string()

    @u.quantity_input
    def __init__(
        self,
        name=LogNormalPrior,
        mu: Union[u.Quantity[u.GeV / u.s], None] = 1e49 * u.GeV / u.s,
        sigma: Union[u.Quantity[u.GeV / u.s], u.Quantity[1], None] = 3.0,
        xmin: Union[u.Quantity[u.GeV / u.s], None] = None,
        alpha: Union[float, None] = None,
    ):
        """
        Converts automatically to log of values, be aware of misuse of notation.
        """
        # This sigma thing is weird due to the log
        super().__init__(
            name, mu=mu, sigma=sigma, xmin=xmin, alpha=alpha, units=self.UNITS
        )


class EnergyPrior(UnitPrior):
    UNITS = u.GeV
    UNITS_STRING = UNITS.to_string()

    @u.quantity_input
    def __init__(
        self,
        name=LogNormalPrior,
        mu: Union[u.Quantity[u.GeV], None] = 1e6 * u.GeV,
        sigma: Union[u.Quantity[u.GeV], u.Quantity[1], None] = 3.0,
    ):
        """
        Converts automatically to log of values, be aware of misuse of notation.
        """
        # This sigma thing is weird due to the log
        super().__init__(name, mu=mu, sigma=sigma, units=self.UNITS)


class FluxPrior(UnitPrior):
    UNITS = 1 / u.m**2 / u.s
    UNITS_STRING = UNITS.unit.to_string()

    @u.quantity_input
    def __init__(
        self,
        name=NormalPrior,
        mu: u.Quantity[1 / u.m**2 / u.s] = 0.314 / u.m**2 / u.s,
        sigma: Union[u.Quantity[1 / u.m**2 / u.s], u.Quantity[1]] = 0.08 / u.m**2 / u.s,
    ):
        super().__init__(name, mu=mu, sigma=sigma, units=self.UNITS)


class IndexPrior(UnitlessPrior):
    UNITS = 1
    UNITS_STRING = "1"

    @u.quantity_input
    def __init__(
        self,
        name=NormalPrior,
        mu: float = 2.5,
        sigma: float = 0.5,
    ):
        super().__init__(name, mu=mu, sigma=sigma, units=self.UNITS)


class MultiSourcePrior:
    """
    Container base class to handle priors with different parameters
    in fits of multiple point sources. If all sources should
    have the same prior, use the normal classes above.
    """

    def __init__(self, priors):
        assert all(isinstance(_._prior, type(priors[0]._prior)) for _ in priors)
        self._priors = priors

    def __getitem__(self, index):
        return self._priors[index]

    def __len__(self):
        return len(self._priors)

    @property
    def name(self):
        return self._priors[0]._prior.name


class MultiSourceLuminosityPrior(MultiSourcePrior, LuminosityPrior):
    def __init__(self, priors: Iterable[LuminosityPrior]):
        assert all(isinstance(_, LuminosityPrior) for _ in priors)
        super().__init__(priors)

    @property
    def mu(self):
        try:
            return np.array([_.mu for _ in self._priors])
        except TypeError:
            return (
                np.array([_.mu.to_value(self.UNITS) for _ in self._priors]) * self.UNITS
            )

    @property
    def sigma(self):
        try:
            return np.array([_.sigma for _ in self._priors])
        except TypeError:
            return (
                np.array([_.sigma.to_value(self.UNITS) for _ in self._priors])
                * self.UNITS
            )

    @property
    def xmin(self):
        try:
            return np.array([_.xmin for _ in self._priors])
        except TypeError:
            return (
                np.array([_.xmin.to_value(self.UNITS) for _ in self._priors])
                * self.UNITS
            )

    @property
    def alpha(self):
        return np.array([_.alpha for _ in self._priors])


class MultiSourceIndexPrior(MultiSourcePrior, IndexPrior):
    def __init__(self, priors: Iterable[IndexPrior]):
        assert all(isinstance(_, IndexPrior) for _ in priors)
        super().__init__(priors)

    @property
    def mu(self):
        return np.array([_.mu for _ in self._priors])

    @property
    def sigma(self):
        return np.array([_.sigma for _ in self._priors])


class MultiSourceEnergyPrior(MultiSourcePrior, EnergyPrior):
    def __init__(self, priors: Iterable[EnergyPrior]):
        assert all(isinstance(_, EnergyPrior) for _ in priors)
        super().__init__(priors)

    @property
    def mu(self):
        return np.array([_.mu for _ in self._priors])

    @property
    def sigma(self):
        return np.array([_.sigma for _ in self._priors])


class Priors(object):
    """
    Container for model priors.

    Defaults to sensible uninformative priors
    on all hyperparameters.
    """

    def __init__(self):
        self._luminosity = LuminosityPrior()

        self.diffuse_flux = FluxPrior(
            LogNormalPrior, mu=1e-4 * FluxPrior.UNITS, sigma=1.0
        )

        self.src_index = IndexPrior()

        self.beta_index = IndexPrior(mu=0.0, sigma=0.1)

        self.diff_index = IndexPrior()

        self.atmospheric_flux = FluxPrior()

        self.E0_src = EnergyPrior()

    @property
    def luminosity(self):
        return self._luminosity

    @luminosity.setter
    def luminosity(self, prior: LuminosityPrior):
        if not isinstance(prior, LuminosityPrior):
            raise ValueError("Wrong prior type")
        self._luminosity = prior

    @property
    def diffuse_flux(self):
        return self._diffuse_flux

    @diffuse_flux.setter
    def diffuse_flux(self, prior: FluxPrior):
        if not isinstance(prior, FluxPrior):
            raise ValueError("Wrong prior type")
        self._diffuse_flux = prior

    @property
    def src_index(self):
        return self._src_index

    @src_index.setter
    def src_index(self, prior: IndexPrior):
        if not isinstance(prior, IndexPrior):
            raise ValueError("Wrong prior type")
        self._src_index = prior

    @property
    def beta_index(self):
        return self._beta_index

    @beta_index.setter
    def beta_index(self, prior: IndexPrior):
        if not isinstance(prior, IndexPrior):
            raise ValueError("Wrong prior type")
        self._beta_index = prior

    @property
    def E0_src(self):
        return self._E0_src

    @E0_src.setter
    def E0_src(self, prior: EnergyPrior):
        if not isinstance(prior, EnergyPrior):
            raise ValueError("Wrong prior type")
        self._E0_src = prior

    @property
    def diff_index(self):
        return self._diff_index

    @diff_index.setter
    def diff_index(self, prior: IndexPrior):
        if not isinstance(prior, IndexPrior):
            raise ValueError("Wrong prior type")
        self._diff_index = prior

    @property
    def atmospheric_flux(self):
        return self._atmospheric_flux

    @atmospheric_flux.setter
    def atmospheric_flux(self, prior: FluxPrior):
        if not isinstance(prior, FluxPrior):
            raise ValueError("Wrong prior type")
        self._atmospheric_flux = prior

    def to_dict(self):
        priors_dict = {}

        priors_dict["L"] = self._luminosity

        priors_dict["F_diff"] = self._diffuse_flux

        priors_dict["F_atmo"] = self._atmospheric_flux

        priors_dict["src_index"] = self._src_index

        priors_dict["beta_index"] = self._beta_index

        priors_dict["diff_index"] = self._diff_index

        priors_dict["beta_index"] = self._beta_index

        priors_dict["E0_src"] = self._E0_src

        return priors_dict

    def save(self, file_name: str):
        with h5py.File(file_name, "w") as f:
            self._writeto(f)

    def _writeto(self, f):
        priors_dict = self.to_dict()

        def create_dataset(g, prior):
            for key, value in prior.to_dict(prior.UNITS_STRING).items():
                g.create_dataset(key, data=value)

        for key, value in priors_dict.items():
            g = f.create_group(key)

            if isinstance(value, MultiSourcePrior):
                for c, prior in enumerate(value):
                    sub_g = g.create_group(f"ps_{c}")
                    create_dataset(sub_g, prior)

            else:
                create_dataset(g, value)

    def addto(self, file_name: str, group_name: str):
        with h5py.File(file_name, "r+") as f:
            g = f.create_group(group_name)

            self._writeto(g)

    @classmethod
    def from_file(cls, file_name: str):
        with h5py.File(file_name, "r") as f:
            return cls._load_from(f)

    @classmethod
    def from_group(cls, file_name: str, group_name: str):
        with h5py.File(file_name, "r") as f:
            g = f[group_name]

            return cls._load_from(g)

    @classmethod
    def _load_from(cls, f):
        priors_dict = {}

        def make_dict_entry(d, arg):
            for k, v in arg.items():
                if k == "name":
                    # name should be replaces by LuminosityPrior etc.
                    d[k] = v[()].decode("ascii")

                else:
                    d[k] = v[()]
            return d

        for key, value in f.items():
            prior_dict = {}
            prior_dict["quantity"] = key

            # for k, v in value.items():
            try:
                prior_dict = make_dict_entry(prior_dict, value)
                priors_dict[key] = PriorDictHandler.from_dict(prior_dict)
            except TypeError:
                # Found the multi source prior

                container = []
                for _, b in value.items():
                    prior_dict = {"quantity": key}
                    prior_dict = make_dict_entry(prior_dict, b)
                    container.append(PriorDictHandler.from_dict(prior_dict))
                if key == "src_index":
                    priors_dict[key] = MultiSourceIndexPrior(container)
                elif key == "beta_index":
                    priors_dict[key] = MultiSourceIndexPrior(container)
                elif key == "E0_src":
                    raise NotImplementedError
                    # priors_dict[key] = MultiSourceEnergyPrior(container)
                elif key == "L":
                    priors_dict[key] = MultiSourceLuminosityPrior(container)

        return cls.from_dict(priors_dict)

    @classmethod
    def from_dict(cls, priors_dict):
        priors = cls()

        priors.luminosity = priors_dict["L"]

        priors.diffuse_flux = priors_dict["F_diff"]

        priors.atmospheric_flux = priors_dict["F_atmo"]

        priors.src_index = priors_dict["src_index"]

        try:
            # Backwards compatiblity
            priors.beta_index = priors_dict["beta_index"]
            priors.E0_src = priors_dict["E0_src"]
        except:
            pass

        priors.diff_index = priors_dict["diff_index"]

        return priors
