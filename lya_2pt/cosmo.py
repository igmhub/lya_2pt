import numpy as np
from typing import Optional
from numpy.typing import ArrayLike
import astropy.units as units
from astropy.cosmology import FlatLambdaCDM, LambdaCDM, FlatwCDM, wCDM


class Cosmology:
    """Class for cosmological computations based on astropy cosmology.
    """
    def __init__(self, Omega_m: float, use_hunits: bool = True, H0: float = 67.36,
                 Omega_de: Optional[float] = None, w0: Optional[float] = None) -> None:
        # Setup some stuff we need
        m_nu = np.array([0.06, 0.0, 0.0]) * units.electronvolt
        Neff = 3.046
        Tcmb = 2.72548
        self.use_hunits = use_hunits

        # Initialize the right cosmology object
        if Omega_de is None and w0 is None:
            self._cosmo = FlatLambdaCDM(H0=H0, Om0=Omega_m, Tcmb0=Tcmb, Neff=Neff, m_nu=m_nu)
        elif w0 is None:
            self._cosmo = LambdaCDM(H0=H0, Om0=Omega_m, Ode0=Omega_de, Tcmb0=Tcmb, Neff=Neff,
                                    m_nu=m_nu)
        elif Omega_de is None:
            self._cosmo = FlatwCDM(H0=H0, Om0=Omega_m, w0=w0, Tcmb0=Tcmb, Neff=Neff, m_nu=m_nu)
        else:
            self._cosmo = wCDM(H0=H0, Om0=Omega_m, Ode0=Omega_de, w0=w0, Tcmb0=Tcmb, Neff=Neff,
                               m_nu=m_nu)

    def comoving_distance(self, z: ArrayLike):
        """Compute comoving distance D_C(z)
        Units are either Mpc or Mpc/h depending on the use_hunits flag

        Parameters
        ----------
        z : ArrayLike
            Redshift

        Returns
        -------
        ArrayLike
            Comoving distance D_C(z)
        """
        if self.use_hunits:
            return self._cosmo.comoving_distance(z).value * (self._cosmo.H0.value / 100)
        else:
            return self._cosmo.comoving_distance(z).value

    def comoving_transverse_distance(self, z: ArrayLike):
        """Compute comoving angular diameter distance D_M(z)
        Units are either Mpc or Mpc/h depending on the use_hunits flag

        Parameters
        ----------
        z : ArrayLike
            Redshift

        Returns
        -------
        ArrayLike
            Comoving angular diameter distance D_M(z)
        """
        if self.use_hunits:
            return self._cosmo.comoving_transverse_distance(z).value * (self._cosmo.H0.value / 100)
        else:
            return self._cosmo.comoving_transverse_distance(z).