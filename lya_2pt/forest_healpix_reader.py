"""This file defines the class ForestReader used to read the data"""

from picca.delta_extraction.utils import ABSORBER_IGM

defaults = {
    "absorption line": "LYA",
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
        reader_config = config["reader"]

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
            arguments = [(tracer, rebin_factor) for tracer in self.tracers]
            pool = Pool(processes=reader_config.getint("num processors"))
            results = pool.starmap(rebin, arguments)
            pool.close()

        # project
        if config.getint("project deltas"):
            pool = Pool(processes=reader_config.getint("num processors"))
            results = pool.starmap(project_deltas, tracers)
            pool.close()

    def find_healpix_neighbours(self):
        """Find the healpix neighbours

        Return
        ------
        healpix_ids: array of int
        The healpix id of the neighbouring healpixes
        """

    def find_neighbours(self, other_tracers):
        """

        Arguments
        ---------
        other_tracers: array of Tracer
        Other tracers
        """
        # for tracer in tracers
            # intialize mask matrix

            # fill True for the neighbours

            # add neightbours to forest

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

def rebin(*args):
    print("Not implemented")
    continue

def project_deltas(*args):
    print("Not implemented")
    continue
