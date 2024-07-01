"""This file defines the class Tracer used to compute the correlation functions"""
import numpy as np

from lya_2pt.constants import ABSORBER_IGM
from lya_2pt.tracer_utils import rebin, project_deltas, get_angle_list, gram_schmidt


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
    r_cart: array of float
        Cartesian (x, y, z) coordinate of the tracer
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
    def __init__(self, healpix_id, los_id, ra, dec, order, deltas,
                 weights, ivar, log_lambda, z, need_distortion=False):
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

        self.r_cart = np.array([np.cos(ra) * np.cos(dec), np.sin(ra) * np.cos(dec), np.sin(dec)])

        self.deltas = deltas.copy()
        self.weights = weights.copy()
        self.ivar = ivar.copy()
        self.log_lambda = log_lambda.copy()
        self.z = z.copy()

        # Needed for distortion matrix computation
        self.need_distortion = need_distortion

        if self.need_distortion:
            self.sum_weights = np.sum(weights)
            self.logwave_term = log_lambda - np.sum(log_lambda * weights) / self.sum_weights
            self.term3_norm = (weights * self.logwave_term**2).sum()
        else:
            self.sum_weights = None
            self.logwave_term = None
            self.term3_norm = None

        self.dist_c = None
        self.dist_m = None
        self.distances = None

        # self.neighbours = None
        # self.num_neighbours = None
        self.is_projected = False
        self.invcov = None
        self.is_weighted = False

    @property
    def size(self):
        return self.z.size

    # def add_neighbours(self, neighbours):
    #     """Add neighbours mask

    #     Arguments
    #     ---------
    #     neighbours: array of bool
    #         Array of length equal to the total number of tracers loaded.
    #         It should be filled with True for the neighbouring lines of sight and
    #         False otherwise.
    #     """
    #     self.neighbours = neighbours.copy()
    #     self.num_neighbours = np.sum(neighbours)

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

    def get_neighbours(self, others, auto_flag, z_min, z_max, rp_max, rt_max):
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
        # # Check if they are in the same redshift bin
        neighbours = [tracer for tracer in others
                      if ((tracer.z[-1] + self.z[-1]) / 2. > z_min
                          and (tracer.z[-1] + self.z[-1]) / 2. < z_max)]

        # For auto correlation we make a selection based on RA to make sure we don't repeat pairs
        if auto_flag:
            neighbours = [tracer for tracer in neighbours if self.ra > tracer.ra]

        # We don't correlate things in the same line of sight
        # due to continuum fitting errors
        neighbours = [tracer for tracer in neighbours if tracer.los_id != self.los_id]

        if not neighbours:
            return np.array([]), np.array([])

        # Compute angle between forests
        angles = get_angle_list(self, neighbours)

        # Check if transverse separation is small enough
        dist_m0 = np.array([tracer.dist_m[0] for tracer in neighbours])
        smallest_rts = (self.dist_m[0] + dist_m0) * np.sin(angles / 2)

        w = smallest_rts < rt_max
        neighbours = np.array(neighbours)[w]
        angles = angles[w]

        # Check if line-of-sight separation is small enough
        cos_angles = np.cos(angles / 2)

        dist_c_start = np.array([tracer.dist_c[0] for tracer in neighbours])
        w1 = self.dist_c[-1] < dist_c_start

        rp_test1 = (self.dist_c[-1] - dist_c_start[w1]) * cos_angles[w1]
        if auto_flag:
            rp_test1 = np.abs(rp_test1)

        w_test1 = np.abs(rp_test1) < rp_max
        not_w1 = ~w1
        not_w1[w1] = w_test1
        neighbours = neighbours[not_w1]
        cos_angles = cos_angles[not_w1]
        angles = angles[not_w1]

        dist_c_end = np.array([tracer.dist_c[-1] for tracer in neighbours])
        w2 = self.dist_c[0] > dist_c_end

        rp_test2 = (self.dist_c[0] - dist_c_end[w2]) * cos_angles[w2]
        if auto_flag:
            rp_test2 = np.abs(rp_test2)

        w_test2 = np.abs(rp_test2) < rp_max
        not_w2 = ~w2
        not_w2[w2] = w_test2
        neighbours = neighbours[not_w2]
        angles = angles[not_w2]

        return neighbours, angles

    def rebin(self, rebin_factor, dwave, absorption_line, use_ivar=False):
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
        log_lambda, deltas, weights, ivar = rebin(self.log_lambda, self.deltas, self.weights,
                                                  self.ivar, rebin_factor, dwave, use_ivar=use_ivar)

        self.log_lambda = log_lambda
        self.deltas = deltas
        self.weights = weights
        self.ivar = ivar
        self.z = 10**log_lambda/ABSORBER_IGM.get(absorption_line) - 1.0

        if self.need_distortion:
            self.sum_weights = np.sum(weights)
            self.logwave_term = log_lambda - np.sum(log_lambda * weights) / self.sum_weights
            self.term3_norm = (weights * self.logwave_term**2).sum()

    def project(self, old_projection=True):
        """Apply projection matrix to deltas"""
        assert not self.is_projected, "Tracer already projected"
        if old_projection:
            self.deltas = project_deltas(self.log_lambda, self.deltas, self.weights, self.order)
        else:
            basis = gram_schmidt(self.log_lambda, self.weights, self.order)

            for b in basis:
                self.deltas -= b * np.dot(b * self.weights, self.deltas)

            self.proj_vec_mat = (basis * self.weights).T

        self.is_projected = True

    def apply_z_evol_to_weights(self, redshift_evol, reference_z):
        self.weights *= ((1 + self.z) / (1 + reference_z))**(redshift_evol - 1)

        if self.need_distortion:
            self.sum_weights = np.sum(self.weights)
            self.logwave_term = self.log_lambda - (np.sum(self.log_lambda * self.weights)
                                                   / self.sum_weights)
            self.term3_norm = (self.weights * self.logwave_term**2).sum()

    def set_inverse_covariance(self, xi1d_interp, cont_order=1):
        if self.invcov is not None:
            return

        z_ij = np.sqrt((1 + self.z[:, None]) * (1 + self.z[None, :])) - 1
        wavelength = 10**self.log_lambda

        delta_lambdas = wavelength[:, None] - wavelength[None, :]
        covariance = xi1d_interp((z_ij, delta_lambdas))
        covariance[np.diag_indices(self.size)] += 1 / self.ivar

        self.invcov = np.linalg.inv(covariance)

        if cont_order < 0:
            return

        template_matrix = np.vander(self.log_lambda, cont_order + 1)
        U, s, _ = np.linalg.svd(template_matrix, full_matrices=False)

        # Remove small singular valued vectors
        w = s > 1e-6
        U = U[:, w]  # shape = (self.size, cont_order + 1)
        Y = self.invcov @ U
        # Woodbury formula. Note that U and Y are not square matrices.
        self.invcov -= Y @ np.linalg.inv(U.T @ Y) @ Y.T

    def release_inverse_covariance(self):
        self.invcov = None

    def apply_invcov_to_deltas(self):
        if self.is_weighted:
            return

        self.deltas = self.invcov.dot(self.deltas)
        self.is_weighted = True
