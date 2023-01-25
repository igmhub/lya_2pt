"""This file defines the class ForestReader used to read the data.
It also contains some auxliary functions for this class
"""
import fitsio
import numpy as np
from numba import njit
from healpy import query_disc
from multiprocessing import Pool


# TODO: check this
# why not load ABSORBER_IGM from delta_extraction?
from lya_2pt.constants import ABSORBER_IGM, ACCEPTED_BLINDING_STRATEGIES
from lya_2pt.errors import ReaderException
from lya_2pt.utils import parse_config, find_bins
from lya_2pt.tracer import Tracer

accepted_options = [
    "input directory", "type", "absorption line", "num processors",
    "project deltas", "order", "rebin", 
]

defaults = {
    "type": 'continuous',
    "absorption line": "LYA",
    "num processors": 1,
    "project deltas" : True,
    "order": 1,
    "rebin": 1
}


class ForestHealpixReader:
    """Class to read the data of a forest-like tracer

    This class will automatically deduce the data format and call the
    relevant methods.
    Two data formats are accepted (from picca.delta_extraction):
    - an HDU per forest
    - image table
    Read data will be formatted as a list of tracers

    Methods
    -------
    __init__
    find_healpix_neighbours
    find_neighbours

    Attributes
    ----------
    auto_flag: bool
    True if we are working with an auto-correlation, False for cross-correlation

    blinding: str
    A string specifying the chosen blinding strategies. Must be one of the
    accepted values in ACCEPTED_BLINDING_STRATEGIES

    tracers: array of Tracer
    The set of tracers for this healpix
    """
    def __init__(self, config, file, cosmo):
        """Initialize class instance

        Arguments
        ---------
        config: configparser.SectionProxy
        Configuration options

        file: str
        Name of the file to read

        cosmo: Cosmology
        Fiducial cosmology used to go from angles and redshift to distances

        Raise
        -----
        ReaderException if the tracer type is not continuous
        ReaderException if the blinding strategy is not valid
        """
        # parse configuration
        reader_config = parse_config(config, defaults, accepted_options)

        # extract parameters from config
        absorption_line = reader_config.get("absorption line")
        tracer1_type = config.get('type')
        if tracer1_type != 'continuous':
            raise ReaderException(
                f"Tracer type must be 'continuous'. Found: '{tracer1_type}'")

        # initialize auto_flag to False
        self.auto_flag = False

        # read data
        self.tracers = None
        hdul = fitsio.FITS(file)
        # image format
        if "METADATA" in hdul:
            self.tracers, self.wave_solution = read_from_image(hdul, absorption_line)
            self.blinding = hdul["METADATA"].read_header()["BLINDING"]
        # HDU per forest
        else:
            self.tracers, self.wave_solution = read_from_hdu(hdul, absorption_line)
            self.blinding = hdul[1].read_header()["BLINDING"]
        if self.blinding not in ACCEPTED_BLINDING_STRATEGIES:
            raise ReaderException(
                "Expected blinding strategy fo be one of: " +
                " ".join(ACCEPTED_BLINDING_STRATEGIES) +
                f" Found: {self.blinding}"
            )

        # rebin
        if config.getint("rebin") > 1:
            if reader_config.getint("num processors") > 1:
                arguments = [(tracer.log_lambda, tracer.deltas, tracer.weights,
                              config.getint("rebin"), self.wave_solution)
                             for tracer in self.tracers]
                with Pool(processes=reader_config.getint("num processors")) as pool:
                    results = pool.starmap(rebin, arguments)
            else:
                results = [rebin(tracer.log_lambda, tracer.deltas, tracer.weights,
                                 config.getint("rebin"), self.wave_solution)
                           for tracer in self.tracers]

            for tracer, (log_lambda, deltas, weights) in zip(self.tracers, results):
                tracer.log_lambda = log_lambda
                tracer.deltas = deltas
                tracer.weights = weights

        # project
        if config.getboolean("project deltas"):
            if reader_config.getint("num processors") > 1:
                arguments = [(tracer.log_lambda, tracer.deltas, tracer.weights,
                              reader_config.getint("order"))
                             for tracer in self.tracers]
                with Pool(processes=reader_config.getint("num processors")) as pool:
                    results = pool.starmap(project_deltas, arguments)
            else:
                results = [project_deltas(tracer.log_lambda, tracer.deltas, tracer.weights,
                                          reader_config.getint("order"),)
                           for tracer in self.tracers]

            for tracer, deltas in zip(self.tracers, results):
                tracer.deltas = deltas

        # add distances
        for tracer in self.tracers:
            tracer.compute_comoving_distances(cosmo)

    def find_healpix_neighbours(self, nside, ang_max):
        """Find the healpix neighbours

        Arguments
        ---------
        nside: int
        Nside parameter to construct the healpix pixels

        ang_max: float
        Maximum angle for two lines-of-sight to have neightbours

        Return
        ------
        healpix_ids: array of int
        The healpix id of the neighbouring healpixes

        Raise
        -----
        ReaderException if the self.tracers is None
        """
        if self.tracers is None:
            raise ReaderException(
                "In ForestHealpixReader, self.tracer should not be None")

        # TODO We need to initialize nside (read from config) and ang_max (compute using cosmo)
        neighbour_ids = set()
        for tracer in self.tracers:
            tracer_neighbour_ids = query_disc(
                nside,
                [tracer.x_cart, tracer.y_cart, tracer.z_cart],
                ang_max,
                inclusive=True)
            neighbour_ids = neighbour_ids.union(set(tracer_neighbour_ids))

        return np.array(list(neighbour_ids))

    def find_neighbours(self, other, z_min, z_max):
        """For each tracer, find neighbouring tracers. Keep the results in
        tracer.neighbours

        Arguments
        ---------
        other: Tracer2Reader
        Other tracers

        z_min: float
        Minimum redshift of the tracers

        z_max: float
        Maximum redshfit of the tracers

        Raise
        -----
        ReaderException if the self.tracers is None
        ReaderException if the other.tracers is None
        """
        if self.tracers is None:
            raise ReaderException(
                "In ForestHealpixReader, self.tracer should not be None")
        if other.tracers is None:
            raise ReaderException(
                "In ForestHealpixReader, other.tracer should not be None")

        for tracer1 in self.tracers:
            neighbour_mask = np.full(other.tracers.shape, False)

            # TODO: This could be vectorized
            for index2, tracer2 in enumerate(other.tracers):
                if tracer1.check_if_neighbour(tracer2, self.auto_flag, z_min, z_max):
                    neighbour_mask[index2] = True

            tracer1.add_neighbours(neighbour_mask)

def read_from_image(hdul, absorption_line):
    """Read data with image format

    Arguments
    ---------
    files: list of str
    List of all the files to read

    cosmo: Cosmology
    Fiducial cosmology used to compute distances

    absorption_line: str
    Name of the absoprtion line responsible for the absorption. Used to translate
    wavelength to redshift. Must be one of the keys of ABSORBER_IGM

    Return
    ------
    tracers: array of Tracer
    The loaded tracers

    Raise
    -----
    ReaderException if both LOGLAM and LAMBDA extensions are not
    in the HDU list
    """
    los_id_array = hdul["METADATA"]["LOS_ID"][:]
    ra_array = hdul["METADATA"]["RA"][:]
    dec_array = hdul["METADATA"]["DEC"][:]

    deltas_array = hdul["DELTA"].read().astype(float)
    weights_array = hdul["WEIGHT"].read().astype(float)
    wave_solution = None
    if "LOGLAM" in hdul:
        log_lambda = hdul["LOGLAM"][:].astype(float)
        z = 10**log_lambda/ABSORBER_IGM.get(absorption_line) - 1.0
        wave_solution = 'log'
    elif "LAMBDA" in hdul:
        lambda_ = hdul["LAMBDA"][:].astype(float)
        log_lambda = np.log10(lambda_)
        z = lambda_/ABSORBER_IGM.get(absorption_line) - 1.0
        wave_solution = 'lin'
    else:
        raise ReaderException(
            "Did not find LOGLAM or LAMBDA in delta file")

    tracers = np.array([Tracer(los_id, ra, dec, deltas_array[index],
                               weights_array[index], log_lambda, z)
                        for index, (los_id, ra, dec)
                        in enumerate(zip(los_id_array, ra_array, dec_array))])

    return tracers, wave_solution

def read_from_hdu(hdul, absorption_line):
    """Read data with an HDU per forest

    Arguments
    ---------
    files: list of str
    List of all the files to read

    cosmo: Cosmology
    Fiducial cosmology used to compute distances

    absorption_line: str
    Name of the absoprtion line responsible for the absorption. Used to translate
    wavelength to redshift. Must be one of the keys of ABSORBER_IGM

    Return
    ------
    tracers: array of Tracer
    The loaded tracers

    Raise
    -----
    ReaderException if both LOGLAM and LAMBDA extensions are not
    in the HDU list
    """
    tracers = []
    for hdu in hdul[1:]:
        header = hdu.read_header()

        los_id = header["LOS_ID"]
        ra = header['RA']
        dec = header['DEC']

        delta = hdu["DELTA"][:].astype(float)
        weights = hdu["WEIGHT"][:].astype(float)
        wave_solution = None
        if 'LOGLAM' in hdu.get_colnames():
            log_lambda = hdu['LOGLAM'][:].astype(float)
            z = 10**log_lambda/ABSORBER_IGM.get(absorption_line) - 1.0
            wave_solution = 'log'
        elif 'LAMBDA' in hdu.get_colnames():
            lambda_ = hdu['LAMBDA'][:].astype(float)
            log_lambda = np.log10(lambda_)
            z = lambda_/ABSORBER_IGM.get(absorption_line) - 1.0
            wave_solution = 'lin'
        else:
            raise ReaderException(
                "Did not find LOGLAM or LAMBDA in delta file")

        tracers.append(Tracer(los_id, ra, dec, delta, weights, log_lambda, z))

    return np.array(tracers), wave_solution


@njit()
def rebin(log_lambda, deltas, weights, rebin_factor, wave_solution):
    """Rebin a Tracer by combining N pixels together

    Arguments
    ---------
    log_lambda: array of float
    An array with the logarithm of the wavelength

    deltas: array of float
    An array with the delta field

    weights: array of float
    An array with the weights associated with the delta field

    rebin_factor: int
    Number of pixels to merge together

    wave_solution: "lin" or "log"
    Specifies whether the underlying wavelength grid is evenly
    spaced on wavelength (lin) or on the logarithm of the wavelength (log)

    Return
    ------
    rebin_log_lambda: array of float
    The rebinned array for the logarithm of the wavelength

    rebin_deltas: array of float
    The rebinned array for the deltas

    rebin_weight: array of float
    The rebinned array for the weights
    """
    # find new grid
    if wave_solution == "lin":
        lambda_ = 10**np.array(log_lambda)
        rebin_lambda = np.average(lambda_.reshape(-1, rebin_factor), axis=1)
        rebin_log_lambda = np.log10(rebin_lambda)
    else:
        rebin_log_lambda = np.average(log_lambda.reshape(-1, rebin_factor), axis=1)

    # do the rebinning
    bins = find_bins(log_lambda, rebin_log_lambda, wave_solution)
    binned_arr_size = bins.max() + 1

    rebin_deltas = np.bincount(bins, weights=weights * deltas, minlength=binned_arr_size)
    rebin_weights = np.bincount(bins, weights=weights, minlength=binned_arr_size)

    return rebin_log_lambda, rebin_deltas, rebin_weights


@njit()
def project_deltas(log_lambda, deltas, weights, order):
    """Project the delta field

    The projection gets rid of the distortion caused by the continuum
    fitiing. See equations 5 and 6 of du Mas des Bourboux et al. 2020

    Arguments
    ---------
    log_lambda: array of float
    An array with the logarithm of the wavelength

    deltas: array of float
    An array with the delta field

    weights: array of float
    An array with the weights associated with the delta field

    order: int
    Order of

    Return
    ------
    projected_deltas: array of float
    The projected deltas. If the sum of weights is zero, then the function
    does nothing and returns the original deltas
    """
    # 2nd term in equation 6
    sum_weights = np.sum(weights)
    if sum_weights > 0.0:
        mean_delta = np.average(deltas, weights=weights)
    else:
        # TODO: write a warning
        return deltas

    projected_deltas = deltas - mean_delta

    # 3rd term in equation 6
    if order == 1:
        mean_log_lambda = np.average(log_lambda, weights=weights)
        meanless_log_lambda = log_lambda - mean_log_lambda
        mean_delta_log_lambda = (
            np.sum(weights * deltas * meanless_log_lambda) /
            np.sum(weights * meanless_log_lambda**2))
        projected_deltas -= mean_delta_log_lambda * meanless_log_lambda

    return projected_deltas
