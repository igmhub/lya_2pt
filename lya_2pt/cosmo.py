"""This file defines the cosmology used to change from angles and redshifts
to distances
"""
import numpy as np
from numba import jit, float64
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.constants import speed_of_light

from lya_2pt.errors import CosmologyError
from lya_2pt.utils import parse_config

accepted_options = [
    "hubble", "omega_m", "omega_k", "omega_r", "use h units", "w0"
]

defaults = {
    "hubble": 67.36,
    "omega_m": 0.315,
    "omega_k": 0.0,
    "omega_r": 7.963219132297603e-05,
    "w0": -1,
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
        config = parse_config(config, defaults, accepted_options)

        self.use_hunits = config.getboolean("use h units")
        self._hubble = config.getfloat('hubble')
        self._hubble_distance = (speed_of_light / 1000 / self._hubble)
        self._Omega_k = config.getfloat('omega_k')

        # Omega_m, Omega_r, Omega_k, w
        self._inv_efunc_args = (config.getfloat('omega_m'), config.getfloat('omega_r'),
                                self._Omega_k, config.getfloat('w0'))


        z = np.linspace(0, 10, 1000)
        comoving_distance = self._comoving_distance(z)
        comoving_transverse_distance = self._comoving_transverse_distance(z)
        if self.use_hunits:
            comoving_distance *= self._hubble / 100
            comoving_transverse_distance *= self._hubble / 100

        self.comoving_distance = interp1d(z, comoving_distance)
        self.comoving_transverse_distance = interp1d(z, comoving_distance)

    def _comoving_distance_scalar(self, z):
        """Compute integral of inverse efunc for a scalar input redshift

        Parameters
        ----------
        z : float
            Target redshift

        Returns
        -------
        float
            Integral of inverse efunc between redshift 0 and input redshift
        """
        return quad(inv_efunc, 0, z, args=self._inv_efunc_args)[0]

    def _comoving_distance(self, z):
        """Compute comoving distance to target redshifts

        Parameters
        ----------
        z : float or array
            Target redshifts

        Returns
        -------
        float or array
            Comoving distances between redshift 0 and input redshifts
        """
        if isinstance(z, (list, tuple, np.ndarray)):
            return self._hubble_distance * np.array([self._comoving_distance_scalar(z_scalar)
                                                     for z_scalar in z])
        else:
            return self._hubble_distance * self._comoving_distance_scalar(z)

    def _comoving_transverse_distance(self, z):
        """Compute comoving transverse distance to target redshifts

        Parameters
        ----------
        z : float or array
            Target redshifts

        Returns
        -------
        float or array
            Comoving transverse distances between redshift 0 and input redshifts
        """
        dc = self._comoving_distance(z)
        if self._Omega_k == 0:
            return dc

        sqrt_Ok0 = np.sqrt(abs(self._Omega_k))
        dh = self._hubble_distance
        if self._Omega_k > 0:
            return dh / sqrt_Ok0 * np.sinh(sqrt_Ok0 * dc / dh)
        else:
            return dh / sqrt_Ok0 * np.sin(sqrt_Ok0 * dc / dh)


@jit(float64(float64, float64, float64, float64, float64))
def inv_efunc(z, Omega_m, Omega_r, Omega_k, w):
    """Hubble parameter in wCDM + curvature

    Parameters
    ----------
    z : float
        Redshift
    Omega_m : float
        Matter fraction at z = 0
    Omega_de : float
        Dark Energy fraction at z = 0
    Returns
    -------
    float
        Hubble parameter
    """
    Omega_de = 1 - Omega_m - Omega_k - Omega_r
    de_pow = 3 * (1 + w)
    zp = 1 + z
    return (Omega_m * zp**3 + Omega_de * zp**de_pow + Omega_k * zp**2 + Omega_r * zp**4)**(-0.5)
