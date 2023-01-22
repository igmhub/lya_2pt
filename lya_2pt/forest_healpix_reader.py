"""This file defines the class ForestReader used to read the data"""

import fitsio
import numpy as np
from numba import njit
from healpy import query_disc
from multiprocessing import Pool

from lya_2pt.utils import parse_config, compute_ang_max
from lya_2pt.tracer import Tracer
from lya_2pt.absorbers import ABSORBER_IGM

accepted_options = ["absorption_line", "project_deltas", "num_processors", "order",
                    "rebin", "z_min", "z_max", "nside"]

defaults = {
    "absorption_line": "LYA",
    "num_processors": 1,
    "project_deltas" : True,
    "order": 1,
    "rebin": 1,
    "z_min": 0,
    "z_max": 10,
    "nside": 16
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
    blinding: str
    A string specifying the chosen blinding strategies. Must be one of the
    accepted values in ACCEPTED_BLINDING_STRATEGIES

    tracers: array of Tracer

    """
    def __init__(self, config, file, cosmo):
        """Initialize class instance

        Arguments
        ---------
        config: configparser.ConfigParser
        Configuration options

        file: str
        Name of the file to read
        """
        # locate files
        reader_config = parse_config(config, defaults, accepted_options)

        # figure out format and blinding
        self.tracers = None
        hdul = fitsio.FITS(file)

        absorption_line = reader_config.get("absorption_line")
        # image format
        if "METADATA" in hdul:
            self.tracers, self.wave_solution = read_from_image(hdul, absorption_line)
            self.blinding = hdul["METADATA"].read_header()["BLINDING"]
        # HDU per forest
        else:
            self.tracers, self.wave_solution = read_from_hdu(hdul, absorption_line)
            self.blinding = hdul[1].read_header()["BLINDING"]

        # rebin
        if config.getint("rebin") > 1:
            if reader_config.getint("num_processors") > 1:
                arguments = [(tracer.log_lambda, tracer.deltas, tracer.weights,
                              config.getint("rebin"), self.wave_solution)
                             for tracer in self.tracers]
                with Pool(processes=reader_config.getint("num_processors")) as pool:
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
        if config.getboolean("project_deltas"):
            if reader_config.getint("num_processors") > 1:
                arguments = [(tracer.log_lambda, tracer.deltas, tracer.weights,
                              reader_config.getint("order"))
                             for tracer in self.tracers]
                with Pool(processes=reader_config.getint("num_processors")) as pool:
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

        # We need to figure out somewhere if this is an auto-correlation or not
        # and initialize this flag (i.e. if data_set_1 = data_set_2)
        self.auto_flag = True

    def find_healpix_neighbours(self, nside, ang_max):
        """Find the healpix neighbours

        Return
        ------
        healpix_ids: array of int
        The healpix id of the neighbouring healpixes
        """
        assert self.tracers is not None, "Forest reader failure"

        # TODO We need to initialize nside (read from config) and ang_max (compute using cosmo)
        neighbour_ids = set()
        for tracer in self.tracers:
            tracer_neighbour_ids = query_disc(nside, [tracer.x_cart, tracer.y_cart, tracer.z_cart],
                                              ang_max, inclusive=True)
            neighbour_ids = neighbour_ids.union(set(tracer_neighbour_ids))

        return np.array(list(neighbour_ids))

    def find_neighbours(self, other_tracers, z_min, z_max):
        """

        Arguments
        ---------
        other_tracers: array of Tracer
        Other tracers
        """
        assert self.tracers is not None, "Forest reader failure"

        for tracer1 in self.tracers:
            neighbour_mask = np.full(other_tracers.shape, False)

            for i, tracer2 in enumerate(other_tracers):
                if tracer1.check_if_neighbour(tracer2, self.auto_flag, z_min, z_max):
                    neighbour_mask[i] = True

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
    """
    # hdul = fitsio.FITS(input_file)

    # header = hdul["METADATA"].read_header()
    # num_forests = hdul["METADATA"].get_nrows()

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
        raise KeyError("Did not find LOGLAM or LAMBDA in delta file")

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
    """
    # hdul = fitsio.FITS(input_file)

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
            raise KeyError("Did not find LOGLAM or LAMBDA in delta file")

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
