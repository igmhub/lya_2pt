"""This file defines the class Tracer used to compute the correlation functions"""
import logging
import numpy as np
from numba.experimental import jitclass
from numba import int64, float64, bool_

spec = [
    ('los_id', int64),
    ('ra', float64),
    ('dec', float64),
    ('x_cart', float64),
    ('y_cart', float64),
    ('z_cart', float64),
    ('deltas', float64[:]),
    ('weights', float64[:]),
    ('log_lambda', float64[:]),
    ('z', float64[:]),
    ('comoving_distance', float64[:]),
    ('comoving_transverse_distance', float64[:]),
    ('neighbours', int64[:]),
    ('angles', float64[:])
]


@jitclass(spec)
class Tracer:
    """Class contanining the information about the tracers.

    Reading functions should create instances of this class, and these instances
    will be used to compute the correlation functions, distortion matrixes, ...

    Methods
    -------
    __init__
    add_neighbours
    compute_comoving_distances

    Attributes
    ----------
    comoving_distance: array of float
    Comoving distance to the object, for each of the deltas

    comoving_transverse_distance: array of float
    comoving angular diameter distance D_M(z), for each of the redshifts

    dec: float
    Line of sight's declination

    deltas: array of float
    The array of deltas for continuous tracers. For discrete tracers, an array
    with ones.

    log_lambda: array of float
    The logarithm of the wavelength associated with the deltas field.
    Needed mostly to compute the distortion matrix

    los_id: int
    Line of sight identifier

    neighbours: array of bool or None
    Once neightbours are looked up, this will contain an array of length equal
    to the total number of tracers loaded. It will be filled with True for the
    neighbouring lines of sight and False otherwise.

    ra: float
    Line of sight's right ascension

    weights: array of float
    The weights associated with the delta field

    z: array of float
    The redshift associated with the deltas field.
    """
    def __init__(self, los_id, ra, dec, deltas, weights, log_lambda, z):
        """Initializes class instance

        neighbours class attribute is initialized to None. Method add_neighbours
        should be called to fully initialize it

        Arguments
        ---------
        los_id: int
        Line of sight identifier

        ra: float
        Line of sight's right ascension

        dec: float
        Line of sight's declination

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
        """
        self.los_id = los_id
        self.ra = ra
        self.dec = dec

        self.x_cart = np.cos(ra) * np.cos(dec)
        self.y_cart = np.sin(ra) * np.cos(dec)
        self.z_cart = np.sin(dec)

        self.deltas = deltas
        self.weights = weights
        self.log_lambda = log_lambda
        self.z = z

        self.comoving_distance = np.zeros(deltas.shape)
        self.comoving_transverse_distance = np.zeros(deltas.shape)

        self.neighbours = np.zeros(1, dtype=np.int64)
        self.angles = np.zeros(1)

    def add_neighbours(self, neighbours, angles):
        """Update the neighbours

        Arguments
        ---------
        neighbours: array of bool
        Array of length equal to the total number of tracers loaded.
        It should be filled with True for the neighbouring lines of sight and
        False otherwise.
        """
        self.neighbours = neighbours
        self.angles = angles

    def set_comoving_distances(self, comoving_distance, comoving_transverse_distance):
        """Compute the comoving distance and the transverse comoving distance

        Arguments
        ---------
        cosmo: lya_2pt.cosmo.Cosmology
        Cosmology used to convert angles and redshifts to distances
        """
        # assert self.z.shape == self.deltas.shape
        self.comoving_distance = comoving_distance
        self.comoving_transverse_distance = comoving_transverse_distance

    # def compute_comoving_distances(self, cosmo):
    #     """Compute the comoving distance and the transverse comoving distance

    #     Arguments
    #     ---------
    #     cosmo: lya_2pt.cosmo.Cosmology
    #     Cosmology used to convert angles and redshifts to distances
    #     """
    #     # assert self.z.shape == self.deltas.shape
    #     self.comoving_distance = cosmo.comoving_distance(self.z)
    #     self.comoving_transverse_distance = cosmo.comoving_transverse_distance(self.z)

    # def check_if_neighbour(self, other, auto_flag, z_min, z_max):
    #     """Check if other tracer is a neighbour

    #     Arguments
    #     ---------
    #     other: Tracer
    #     The neighbour candidate

    #     auto_flag: bool
    #     A flag specifying whether we want to compute the auto-correlation.

    #     Return
    #     ------
    #     is_neighbour: bool
    #     True if the tracers are neighbours. False otherwise
    #     """
    #     # this is to avoid double counting in the autocorrelation
    #     if auto_flag and (self.ra < other.ra):
    #         return False
    #     # we don't correlate things in the same line of sight, due to continuum
    #     # fitting errors
    #     if other.los_id == self.los_id:
    #         return False

    #     # redshift checks
    #     z_check = (other.z[-1] + self.z[-1]) / 2.
    #     if z_check < z_min or z_check >= z_max:
    #         return False

    #     # Add more conditions

    #     return True
