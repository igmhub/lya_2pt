"""This file defines the class Tracer used to compute the correlation functions"""
import numpy as np

from lya_2pt.constants import ABSORBER_IGM
from lya_2pt.tracer_utils import rebin, project_deltas, get_angle
from lya_2pt.compute_utils import compute_rp, compute_rt


class Tracer:
    """Class contanining the tracer data

    Reading functions should create instances of this class, and these instances
    will be used to compute the correlation functions, distortion matrices, ...

    Attributes
    ----------
    healpix_id: int
        Healpix identifier
    los_id: int
        Line of sight identifier
    ra: float
        Line of sight's right ascension
    dec: float
        Line of sight's declination
    order: int
        Order of polynomial used for the continuum fitting
    x_cart: float
        Cartesian x coordinate of the tracer
    y_cart: float
        Cartesian y coordinate of the tracer
    z_cart: float
        Cartesian z coordinate of the tracer
    deltas: array of float
        The array of deltas for continuous tracers. For discrete tracers, an array
        with ones.
    weights: array of float
        The weights associated with the delta field
    log_lambda: array of float
        The logarithm of the wavelength associated with the deltas field.
        Needed mostly to compute the distortion matrix
    z: array of float
        The redshift associated with the deltas field.
    need_distortion : bool
        Whether we need to compute the distortion matrix
    sum_weights: array of float, or None
        Sum of all weights in the tracer instance, needed for distortion computation
    logwave_term: array of float, or None
        Log lambda minus weighted mean of log lambda, needed for distortion computation
    term3_norm: array of float, or None
        Normalization for the third term in the projection matrix, needed for distortion computation
    dist_c: array of float, or None
        Comoving distance, needs initialization through compute_comoving_distances()
    dist_m: array of float, or None
        Comoving transverse distance, needs initialization through compute_comoving_distances()
    distances: array of float, or None
        Comoving distances saved as one array, needed for distortion computation
    neighbours: array of bool, or None
        Array of length equal to the total number of tracers loaded.
        It should be filled with True for the neighbouring lines of sight and
        False otherwise. Needs initialization through add_neighbours()
    num_neighbours: array of bool, or None
        Size of neighbours array, needs initialization through add_neighbours()
    is_projected: bool
        Whether the projection matrix has been aplied or not

    Methods
    -------
    add_neighbours(neighbours):
        Add neighbours mask
    compute_comoving_distances(cosmo):
        Compute the comoving distance and the transverse comoving distance
    is_neighbour(other, auto_flag, z_min, z_max, rp_max, rt_max):
        Check if other tracer is a neighbour
    rebin(rebin_factor, dwave, absorption_line):
        Rebin the forest into coarser pixels
    project():
        Apply projection matrix to deltas
    """
    def __init__(self, healpix_id, los_id, ra, dec, order,
                 deltas, weights, log_lambda, z, need_distortion=False):
        """Initializes class instance

        Parameters
        ----------
        healpix_id: int
            Healpix identifier
        los_id: int
            Line of sight identifier
        ra: float
            Line of sight's right ascension
        dec: float
            Line of sight's declination
        order: int
            Order of polynomial used for the continuum fitting
        deltas: array of float
            The array of deltas for continuous tracers. For discrete tracers, an array
            with ones.
        weights: array of float
            The weights associated with the delta field
        log_lambda: array of float
            The logarithm of the wavelength associated with the deltas field.
            Needed mostly to compute the distortion matrix
        z: array of float
            The redshift associated with the deltas field.
        need_distortion : bool, optional
            Whether we need to compute the distortion matrix, by default False
        """
        self.healpix_id = healpix_id
        self.los_id = los_id
        self.ra = ra
        self.dec = dec
        self.order = order

        self.x_cart = np.cos(ra) * np.cos(dec)
        self.y_cart = np.sin(ra) * np.cos(dec)
        self.z_cart = np.sin(dec)

        self.deltas = deltas
        self.weights = weights
        self.log_lambda = log_lambda
        self.z = z

        # Needed for distortion matrix computation
        self.need_distortion = need_distortion
        self.sum_weight = None
        self.logwave_term = None
        self.term3_norm = None
        if self.need_distortion:
            self.sum_weights = np.sum(weights)
            self.logwave_term = log_lambda - np.sum(log_lambda * weights) / self.sum_weights
            self.term3_norm = (weights * self.logwave_term**2).sum()

        self.dist_c = None
        self.dist_m = None
        self.distances = None

        self.neighbours = None
        self.num_neighbours = None
        self.is_projected = False

    def add_neighbours(self, neighbours):
        """Add neighbours mask

        Arguments
        ---------
        neighbours: array of bool
            Array of length equal to the total number of tracers loaded.
            It should be filled with True for the neighbouring lines of sight and
            False otherwise.
        """
        self.neighbours = neighbours
        self.num_neighbours = np.sum(neighbours)

    def compute_comoving_distances(self, cosmo):
        """Compute the comoving distance and the transverse comoving distance

        Arguments
        ---------
        cosmo: lya_2pt.cosmo.Cosmology
            Cosmology used to convert angles and redshifts to distances
        """
        assert self.z.shape == self.deltas.shape
        self.dist_c = cosmo.get_dist_c(self.z)
        self.dist_m = cosmo.get_dist_m(self.z)

        if self.need_distortion:
            self.distances = np.c_[self.dist_c, self.dist_m]

    def is_neighbour(self, other, auto_flag, z_min, z_max, rp_max, rt_max):
        """Check if other tracer is a neighbour

        Parameters
        ----------
        other : Tracer
            The neighbour candidate tracer object
        auto_flag : bool
            A flag specifying whether we want to compute an auto-correlation.
        z_min : float
            minimum redshift
        z_max : float
            maximum redshift
        rp_max : float
            maximum comoving separation along the line-of-sight
        rt_max : float
            maximum comoving transverse separation

        Returns
        -------
        bool
            True if the tracers are neighbours. False otherwise
        """
        # This is to avoid double counting in the autocorrelation
        # The equality check is purely for compatibility with picca
        # Somehow there are a few forests with identical RA
        if auto_flag and (self.ra <= other.ra):
            return False

        # We don't correlate things in the same line of sight
        # due to continuum fitting errors
        if other.los_id == self.los_id:
            return False

        # Check if they are in the same redshift bin
        z_check = (other.z[-1] + self.z[-1]) / 2.
        if z_check < z_min or z_check >= z_max:
            return False

        # Compute angle between forests
        angle = get_angle(self.x_cart, self.y_cart, self.z_cart, self.ra, self.dec,
                          other.x_cart, other.y_cart, other.z_cart, other.ra, other.dec)

        # Check if transverse separation is small enough
        smallest_rt = compute_rt(self.dist_m, other.dist_m, i=0, j=0, sin_angle=np.sin(angle/2))
        if smallest_rt > rt_max:
            return False

        # Check if line-of-sight separation is small enough
        cos_angle = np.cos(angle/2)
        if self.dist_c[-1] < other.dist_c[0]:
            rp_test = compute_rp(self.dist_c, other.dist_c, i=-1, j=0, cos_angle=cos_angle,
                                 auto_flag=auto_flag)

            if np.abs(rp_test) > rp_max:
                return False

        if self.dist_c[0] > other.dist_c[-1]:
            rp_test = compute_rp(self.dist_c, other.dist_c, i=0, j=-1, cos_angle=cos_angle,
                                 auto_flag=auto_flag)

            if np.abs(rp_test) > rp_max:
                return False

        return True

    def rebin(self, rebin_factor, dwave, absorption_line):
        """Rebin the forest into coarser pixels

        Parameters
        ----------
        rebin_factor : int
            Factor multiplying the input pixel size to obtain the output pixel size
        dwave : float
            Wavelength bin size
        absorption_line : string
            Name of main absorption line
        """
        log_lambda, deltas, weights = rebin(self.log_lambda, self.deltas, self.weights,
                                            rebin_factor, dwave)

        self.log_lambda = log_lambda
        self.deltas = deltas
        self.weights = weights
        self.z = 10**log_lambda/ABSORBER_IGM.get(absorption_line) - 1.0

        if self.need_distortion:
            self.sum_weights = np.sum(weights)
            self.logwave_term = log_lambda - np.sum(log_lambda * weights) / self.sum_weights
            self.term3_norm = (weights * self.logwave_term**2).sum()

    def project(self):
        """Apply projection matrix to deltas"""
        assert not self.is_projected, "Tracer already projected"
        self.deltas = project_deltas(self.log_lambda, self.deltas, self.weights, self.order)
        self.is_projected = True
