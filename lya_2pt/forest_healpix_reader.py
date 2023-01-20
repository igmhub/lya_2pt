"""This file defines the class ForestReader used to read the data"""

import numpy as np
from healpy import query_disc

from lya_2pt.utils import parse_config
from lya_2pt.tracer import Tracer

from picca.delta_extraction.utils import ABSORBER_IGM

accepted_options = ["absorption line", "num processors",  "order"]

defaults = {
    "absorption line": "LYA",
    "num processors": 1,
    "order": 1,
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
    def __init__(self, config, file):
        """Initialize class instance

        Arguments
        ---------
        config: configparser.ConfigParser
        Configuration options

        file: str
        Name of the file to read
        """
        # locate files
        reader_config = parse_config(config["reader"], defaults, accepted_options)

        # intialize cosmology
        cosmo = Cosmology(config["cosmology"])

        # figure out format and blinding
        self.tracers = None
        hdu = fitsio.FITS(input_file)
        # image format
        if "METADATA" in hdu:
            self.tracers = read_from_image(
                input_file,
                reader_config.get("absorption line"))
            self.blinding = hdu["METADATA"].read_header()["BLINDING"]
        # HDU per forest
        else:
            self.tracers = read_from_image(input_file)
            self.blinding = hdu[1].read_header()["BLINDING"]

        # rebin
        if config.getint("rebin") > 1:
            if reader_config.getint("num processors") > 1:
                arguments = [
                    (tracer.log_lambda, tracer.flux, rebin_factor)
                    for tracer in self.tracers
                ]
                pool = Pool(processes=reader_config.getint("num processors"))
                results = pool.starmap(rebin, arguments)
                pool.close()
            else:
                results = [
                    rebin(tracer.log_lambda, tracer.flux, rebin_factor)
                    for tracer in self.tracers
                ]
            for tracer, (log_lambda, deltas, weights) in zip(tracers, results):
                tracer.log_lambda = log_lambda
                tracer.deltas = deltas
                tracer.weights = weights


        # project
        if config.getint("project deltas"):
            if reader_config.getint("num processors") > 1:
                arguments = [
                    (tracer.log_lambda, tracer.deltas, tracer.weights,
                     reader_config.getint("order"))
                    for tracer in self.tracers
                ]
                pool = Pool(processes=reader_config.getint("num processors"))
                results = pool.starmap(project_deltas, arguments)
                pool.close()
            else:
                results = [
                    project_deltas(
                        tracer.log_lambda,
                        tracer.deltas,
                        tracer.weights,
                        reader_config.getint("order"),
                    )
                    for tracer in tracers
                ]

            for tracer, (log_lambda, deltas, weights) in zip(tracers, results):
                tracer.deltas = deltas

        # add distances
        [tracer.compute_comoving_distances(cosmo) for tracer in tracers]

        self.z_min = config["cuts"].getfloat('z_min', 0.)
        self.z_max = config["cuts"].getfloat('z_max', 10.)

        # We need to figure out somewhere if this is an auto-correlation or not
        # and initialize this flag (i.e. if data_set_1 = data_set_2)
        self.auto_flag = True

    def find_healpix_neighbours(self):
        """Find the healpix neighbours

        Return
        ------
        healpix_ids: array of int
        The healpix id of the neighbouring healpixes
        """
        # TODO We need to initialize nside (read from config) and ang_max (compute using cosmo)
        neighbour_ids = set()
        for tracer in self.tracers:
            tracer_neighbour_ids = query_disc(self.nside, [tracer.x_cart, tracer.y_cart, tracer.z_cart],
                                              self.ang_max, inclusive=True)
            neighbour_ids = neighbour_ids.union(set(tracer_neighbour_ids))

        return np.array(neighbour_ids)
            

    def find_neighbours(self, other_tracers):
        """

        Arguments
        ---------
        other_tracers: array of Tracer
        Other tracers
        """
        for tracer1 in self.tracers:
            neighbour_mask = np.full(other_tracers.shape, False)

            for i, tracer2 in enumerate(other_tracers):
                if tracer1.check_if_neighbour(tracer2, self.auto_flag,
                                              self.z_min, self.z_max):
                    neighbour_mask[i] = True

            tracer1.add_neighbours(neighbour_mask)

def read_from_image(input_file, cosmo, absorption_line):
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
    hdul = fitsiio.FITS(input_file)

    header = hdul["METADATA"].read_header()
    num_forests = hdul["METADATA"].get_nrows()

    los_id_array = hdul["METADATA"]["LOS_ID"][:]
    ra_array = hdul["METADATA"]["RA"][:]
    dec_array = hdul["METADATA"]["DEC"][:]

    deltas_array = hdul["DELTA_BLIND"].read().astype(float)
    weights_array = hdul["WEIGHT"].read().astype(float)
    if "LOGLAM" in hdul:
        log_lambda = hdul["LOGLAM"][:].astype(float)
        z = 10**log_lambda/ABSORBER_IGM.get(absorption_line) - 1.0
    elif "LAMBDA" in hdul:
        lambda_ = hdul["LAMBDA"][:].astype(float)
        log_lambda = np.log10(lambda_)
        z = lambda_/ABSORBER_IGM.get(absorption_line) - 1.0
    else:
        raise KeyError("Did not find LOGLAM or LAMBDA in delta file")

    tracers = np.array([
        Tracer(los_id, ra, dec, deltas_array[index], weights_array[index],
               log_lambda, z, cosmo)
        for index, (los_id, ra, dec) in enumerate(zip(los_id_array, ra_array, dec_array))
    ])

    return tracers

def read_from_hdu(input_file, cosmo, absorption_line):
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
    hdul = fitsio.FITS(input_file)

    tracers = []
    for hdu in hdul[1:]:
        header = hdu.read_header()

        los_id = header["LOS_ID"][:]
        ra = header['RA']
        dec = header['DEC']

        delta = hdu["DELTA_BLIND"][:].astype(float)
        if 'LOGLAM' in hdu.get_colnames():
            log_lambda = hdu['LOGLAM'][:].astype(float)
            z = 10**log_lambda/ABSORBER_IGM.get(absorption_line) - 1.0
        elif 'LAMBDA' in hdu.get_colnames():
            lambda_ = hdu['LAMBDA'][:].astype(float)
            log_lambda = np.log10(lambda_)
            z = lambda_/ABSORBER_IGM.get(absorption_line) - 1.0
        else:
            raise KeyError("Did not find LOGLAM or LAMBDA in delta file")

        tracers.append(Tracer(los_id, ra, dec, deltas, weights, log_lambda, z, cosmo))

    return np.arrays(tracers)

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
    # find neew grid
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
