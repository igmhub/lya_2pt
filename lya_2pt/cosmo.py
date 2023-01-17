"""This file defines the cosmology used to change from angles and redshifts
to distances
"""
import numpy as np

import astropy.units as units
from astropy.cosmology import FlatLambdaCDM, LambdaCDM, FlatwCDM, wCDM

accepted_options = [
    "H0", "Omega_de", "Omega_m", "m_nu", "Neff", "use h units", "T_cmb",
]

defaults = {
    "H0": 67.36,
    "Omega_m": 0.315,
    "m_nu": "0.06; 0.0; 0.0",
    "Neff": 3.046,
    "Tcmb": 2.72548,

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
        if self.config.get("Omega_de") is None and self.config.get("w0") is None:
            self._cosmo = FlatLambdaCDM(
                H0=self.config.get("H0"),
                Om0=self.config.get("Omega_m"),
                Tcmb0=self.config.get("Tcmb"),
                Neff=self.config.get("Neff"),
                m_nu=self.config.get("m_nu") * units.electronvolt,
            )
        elif self.config.get("w0") is None:
            self._cosmo = LambdaCDM(
                H0=self.config.get("H0"),
                Om0=self.config.get("Omega_m"),
                Ode0=self.config.get("Omega_de"),
                Tcmb0=self.config.get("Tcmb"),
                Neff=self.config.get("Neff"),
                m_nu=self.config.get("m_nu") * units.electronvolt,
            )
        elif self.config.get("Omega_de") is None:
            self._cosmo = FlatwCDM(
                H0=self.config.get("H0"),
                Om0=self.config.get("Omega_m"),
                w0=self.config.get("w0"),
                Tcmb0=self.config.get("Tcmb"),
                Neff=self.config.get("Neff"),
                m_nu=self.config.get("m_nu") * units.electronvolt,
            )
        else:
            self._cosmo = wCDM(
                H0=self.config.get("H0"),
                Om0=self.config.get("Omega_m"),
                Ode0=self.config.get("Omega_de"),
                w0=self.config.get("w0"),
                Tcmb0=self.config.get("Tcmb"),
                Neff=self.config.get("Neff"),
                m_nu=self.config.get("m_nu") * units.electronvolt,
            )

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
        # update the section adding the default choices when necessary
        for key, value in defaults.items():
            if key not in config:
                config[key] = str(value)

        # make sure all the required variables are present
        for key in accepted_options:
            if key not in config:
                raise CosmologyError(f"Missing option {key}"")

        # check that all arguments are valid
        for key in correction_args:
            if key not in accepted_options:
                raise CosmologyError(
                    f"Unrecognised option. Found: '{key}'. Accepted options are "
                    f"{accepted_options}")

        # special check: m_nu format
        try:
            m_nu = np.fromstring(config.get("m_nu"), sep=";")
            if m_nu.size != np.floor(config.get("Neff")):
                raise CosmologyError(
                    f"Incorrect format for option 'm_nu'. "
                    f"Expected {np.floor(config.get('Neff'))} masses. "
                    f"Found {m_nu.size}. Read array: {m_nu}")
        except TypeError as error:
            raise CosmologyError(
                f"Incorrect format for option 'm_nu'. Expected a string with "
                "coma separated numbers") from error

    def comoving_distance(self, z):
        """Compute comoving distance D_C(z)
        Units are either Mpc or Mpc/h depending on the use_hunits flag

        Arguments
        ---------
        z : array of float
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

    def comoving_transverse_distance(self, z: ArrayLike):
        """Compute comoving angular diameter distance D_M(z)
        Units are either Mpc or Mpc/h depending on the use_hunits flag

        Arguments
        ---------
        z : array of float
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
