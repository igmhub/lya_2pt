"""This file defines the cosmology used to change from angles and redshifts
to distances
"""
import numpy as np
from numba import njit, float64
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.constants import speed_of_light

from lya_2pt.utils import parse_config

accepted_options = [
    "use-picca-cosmo", "omega_m", "omega_r", "hubble-constant", "use-h-units",
    "omega_k", "w0"
]

defaults = {
    "use-picca-cosmo": False,
    "omega_m": 0.315,
    "omega_r": 7.963219132297603e-05,
    "hubble-constant": 67.36,
    "use-h-units": True,
    "omega_k": 0.0,
    "w0": -1,
}


class Cosmology:
    """Class for cosmological computations based on astropy cosmology.

    Attributes
    ----------
    config: configparser.SectionProxy
    Parsed options to build cosmology

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

        if config.getboolean("use-picca-cosmo"):
            picca_cosmo = PiccaCosmo(config)

            # D_C, D_M
            self.get_dist_c = picca_cosmo.get_r_comov
            self.get_dist_m = picca_cosmo.get_dist_m
        else:
            self.use_h_units = config.getboolean("use-h-units")
            self._hubble_constant = config.getfloat('hubble-constant')
            self._hubble_distance = (speed_of_light / 1000 / self._hubble_constant)
            self._Omega_k = config.getfloat('omega_k')

            # Omega_m, Omega_r, Omega_k, w
            self._inv_efunc_args = (config.getfloat('omega_m'), config.getfloat('omega_r'),
                                    self._Omega_k, config.getfloat('w0'))

            z = np.linspace(0, 10, 10000)
            comoving_distance = self._comoving_distance(z)
            comoving_transverse_distance = self._comoving_transverse_distance(z)
            if self.use_h_units:
                comoving_distance *= self._hubble_constant / 100
                comoving_transverse_distance *= self._hubble_constant / 100

            # D_C, D_M
            self.get_dist_c = interp1d(z, comoving_distance)
            self.get_dist_m = interp1d(z, comoving_transverse_distance)

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


@njit(float64(float64, float64, float64, float64, float64))
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


class PiccaCosmo():
    def __init__(self, config):
        """Initializes the methods for this instance

        Args:
            Om: float - default: 0.3
                Matter density
            Ok: float - default: 0.0
                Curvature density
            Or: float - default: 0.0
                Radiation density
            wl: float - default: -1.0
                Dark energy equation of state
            H0: float - default: 100.0
                Hubble constant at redshift 0 (in km/s/Mpc)
        """
        Om = config.getfloat('omega_m')
        Ok = config.getfloat('omega_k')
        Or = config.getfloat('omega_r')
        wl = config.getfloat('w0')

        # WARNING: This is introduced due to historical issues in how this class
        # is coded. Using H0=100 implies that we are returning the distances
        # in Mpc/h instead of Mpc. This class should be fixed at some point to
        # make what we are doing more clear.
        H0 = 100.0
        SPEED_LIGHT = speed_of_light / 1000

        # print(f"Om={Om}, Or={Or}, wl={wl}")

        # Ignore evolution of neutrinos from matter to radiation
        Ol = 1. - Ok - Om - Or

        num_bins = 10000
        z_max = 10.
        delta_z = z_max/num_bins
        z = np.arange(num_bins, dtype=float)*delta_z
        hubble = H0*np.sqrt(Ol*(1. + z)**(3.*(1. + wl)) +
                            Ok*(1. + z)**2 +
                            Om*(1. + z)**3 +
                            Or*(1. + z)**4)

        r_comov = np.zeros(num_bins)
        for index in range(1, num_bins):
            r_comov[index] = (SPEED_LIGHT*(1./hubble[index - 1] +
                                           1./hubble[index])/2.*delta_z +
                              r_comov[index - 1])

        self.get_r_comov = interp1d(z, r_comov)

        # dist_m here is the comoving angular diameter distance
        if Ok == 0.:
            dist_m = r_comov
        elif Ok < 0.:
            dist_m = (np.sin(H0*np.sqrt(-Ok)/SPEED_LIGHT*r_comov) /
                      (H0*np.sqrt(-Ok)/SPEED_LIGHT))
        elif Ok > 0.:
            dist_m = (np.sinh(H0*np.sqrt(Ok)/SPEED_LIGHT*r_comov) /
                      (H0*np.sqrt(Ok)/SPEED_LIGHT))
        else:
            # Should never get here
            raise ValueError(f'Picca cosmology failure. Something went wrong with Omega_k={Ok}')

        self.get_hubble = interp1d(z, hubble)
        self.distance_to_redshift = interp1d(r_comov, z)

        # D_H
        self.get_dist_hubble = interp1d(z, SPEED_LIGHT/hubble)
        # D_M
        self.get_dist_m = interp1d(z, dist_m)
        # D_V
        dist_v = np.power(z*self.get_dist_m(z)**2*self.get_dist_hubble(z), 1./3.)
        self.get_dist_v = interp1d(z, dist_v)
