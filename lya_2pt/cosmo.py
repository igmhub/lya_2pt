"""This file defines the cosmology used to change from angles and redshifts
to distances
"""
import numpy as np

import astropy.units as units
from astropy.cosmology import FlatLambdaCDM, LambdaCDM, FlatwCDM, wCDM

from lya_2pt.errors import CosmologyError
from lya_2pt.utils import parse_config

accepted_options = [
    "hubble", "omega_de", "omega_m", "m_nu", "neff", "use h units", "tcmb", "w0"
]

defaults = {
    "hubble": 67.36,
    "omega_m": 0.315,
    "m_nu": "0.06;0.0;0.0",
    "neff": 3.046,
    "tcmb": 2.72548,
    "omega_de": None,
    "w0": None,
    "use h units": True,
}

class Cosmology:
    """Class for cosmological computations based on astropy cosmology.

    Methods
    -------
    __init__
    __parse_config
    comoving_distance
    comoving_transverse_distance

    Attributes
    ----------
    config: configparser.SectionProxy
    Parsed options to build cosmology

    cosmo: astropy.cosmology.Cosmo
    Astropy cosmology

    use_hunits: bool
    If True, do the computation in h^-1Mpc. Otherwise, do it in Mpc
    """
    def __init__(self, config):
        """Initialize class instance

        Arguments
        ---------
        config: configparser.SectionProxy
        Configuration options
        """
        self.config = self.__parse_config(config)

        self.use_hunits = self.config.getboolean("use h units")

        # Initialize the right cosmology object
        if self.Omega_de is None and self.w0 is None:
            self._cosmo = FlatLambdaCDM(H0=self.config.get("hubble"),
                                        Om0=self.config.get("omega_m"),
                                        Tcmb0=self.config.get("tcmb"),
                                        Neff=self.config.get("neff"),
                                        m_nu=self.m_nu * units.electronvolt)
        elif self.w0 is None:
            self._cosmo = LambdaCDM(H0=self.config.get("hubble"),
                                    Om0=self.config.get("omega_m"),
                                    Ode0=self.Omega_de,
                                    Tcmb0=self.config.get("tcmb"),
                                    Neff=self.config.get("neff"),
                                    m_nu=self.m_nu * units.electronvolt)
        elif self.Omega_de is None:
            self._cosmo = FlatwCDM(H0=self.config.get("hubble"),
                                   Om0=self.config.get("omega_m"),
                                   w0=self.config.get("w0"),
                                   Tcmb0=self.config.get("tmb"),
                                   Neff=self.config.get("neff"),
                                   m_nu=self.m_nu * units.electronvolt)
        else:
            self._cosmo = wCDM(H0=self.config.get("hubble"),
                               Om0=self.config.get("omega_m"),
                               Ode0=self.Omega_de,
                               w0=self.w0,
                               Tcmb0=self.config.get("tcmb"),
                               Neff=self.config.get("neff"),
                               m_nu=self.m_nu * units.electronvolt)

    def __parse_config(self, config):
        """Parse the given configuration

        Check that all required variables are present
        Load default values for missing optional variables

        Arguments
        ---------
        config: configparser.SectionProxy
        Configuration options

        Return
        ------
        config: configparser.SectionProxy
        Parsed options to initialize class
        """
        config = parse_config(config, defaults, accepted_options)

        # special check: m_nu format
        try:
            self.m_nu = np.array(np.fromstring(config.get("m_nu"), sep=";"))
            if self.m_nu.size != int(np.floor(config.getfloat("Neff"))):
                raise CosmologyError(
                    f"Incorrect format for option 'm_nu'. "
                    f"Expected {np.floor(config.getfloat('neff'))} masses. "
                    f"Found {self.m_nu.size}. Read array: {self.m_nu}")
        except TypeError as error:
            raise CosmologyError(
                f"Incorrect format for option 'm_nu'. Expected a string with "
                "coma separated numbers") from error

        if config.get('omega_de') == 'None':
            self.Omega_de = None
        else:
            self.Omega_de = config.getfloat('omega_de')

        if config.get('w0') == 'None':
            self.w0 = None
        else:
            self.w0 = config.getfloat('w0')

        return config

    def comoving_distance(self, z):
        """Compute comoving distance D_C(z)
        Units are either Mpc or Mpc/h depending on the use_hunits flag

        Arguments
        ---------
        z: array of float
        Redshifts at which to compute the comoving distance

        Return
        ------
        distance: array of float
        Comoving distance D_C(z)
        """
        distance = self._cosmo.comoving_distance(z).value
        if self.use_hunits:
            distance *= self._cosmo.H0.value / 100
        return distance

    def comoving_transverse_distance(self, z):
        """Compute comoving angular diameter distance D_M(z)
        Units are either Mpc or Mpc/h depending on the use_hunits flag

        Arguments
        ---------
        z: array of float
        Redshifts at which to compute the angular diameter distance

        Return
        ------
        distance: array of float
        Comoving angular diameter distance D_M(z)
        """
        distance = self._cosmo.comoving_transverse_distance(z).value
        if self.use_hunits:
            distance *= self._cosmo.H0.value / 100
        return distance
