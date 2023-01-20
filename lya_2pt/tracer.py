"""This file defines the class Tracer used to compute the correlation functions"""
import logging

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

        self.deltas = deltas
        self.weights = weights
        self.log_lambda = log_lambda
        self.z = z

        self.comoving_distance = None
        self.comoving_transverse_distance = None

        self.neighbours = None

    def add_neighbours(self, neighbours):
        """Update the neighbours

        Arguments
        ---------
        neighbours: array of bool
        Array of length equal to the total number of tracers loaded.
        It should be filled with True for the neighbouring lines of sight and
        False otherwise.
        """
        self.neighbours = neighbours

    def compute_comoving_distances(self, cosmo):
        """Compute the comoving distance and the transverse comoving distance

        Arguments
        ---------
        cosmo: lya_2pt.cosmo.Cosmology
        Cosmology used to convert angles and redshifts to distances
        """
        self.comoving_distance = cosmo.comoving_distance(z)
        self.comoving_transverse_distance = cosmo.comoving_transverse_distance(z)
