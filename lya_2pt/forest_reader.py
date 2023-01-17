"""This file defines the class ForestReader used to read the data"""


class ForestReader:
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

    Attributes
    ----------
    blinding: str
    A string specifying the chosen blinding strategies. Must be one of the
    accepted values in ACCEPTED_BLINDING_STRATEGIES

    """
    def __init__(self, config):
        """Initialize class instance

        Arguments
        ---------
        config: configparser.SectionProxy
        Configuration options
        """
        # locate files
        in_dir = config.get("input directory")
        input_file = config.get("input file")

        # figure out format and blinding
        self.tracers = None
        hdu = fitsio.FITS(input_file)
        # image format
        if "LAMBDA" in hdu:
            self.tracers = read_from_image(input_file)
            self.blinding = hdu["METADATA"].read_header()["BLINDING"]
        # HDU per forest
        else:
            self.tracers = read_from_image(input_file)
            self.blinding = hdu[1].read_header()["BLINDING"]

        # rebin
        if config.getint("rebin") > 1:
            arguments = [(tracer, rebin_factor) for tracer in self.tracers]
            pool = Pool(processes=config.getint("num processors"))
            results = pool.starmap(rebin, arguments)
            pool.close()

        # project
        if config.getint("project deltas"):
            pool = Pool(processes=config.getint("num processors"))
            results = pool.starmap(project_deltas, tracers)
            pool.close()

def read_from_image(input_file):
    """Read data with image format

    Arguments
    ---------
    files: list of str
    List of all the files to read

    Return
    ------
    tracers: array of Tracer
    The loaded tracers
    """
    hdul = fitsiio.FITS(input_file)

    header = hdul["METADATA"].read_header()
    num_forests = hdul["METADATA"].get_nrows()
    nones = np.full(N_forests, None)

    delta = hdul["DELTA_BLIND"].read().astype(float)
    if "LOGLAM" in hdul:
        log_lambda = hdul["LOGLAM"][:].astype(float)
    elif "LAMBDA" in hdul:
        log_lambda = np.log10(hdul["LAMBDA"][:].astype(float))
    else:
        raise KeyError("Did not find LOGLAM or LAMBDA in delta file")

    ivar = nones
    exposures_diff = nones
    mean_snr = nones
    mean_reso = nones
    mean_z = nones
    resolution_matrix = nones
    mean_resolution_matrix = nones
    mean_reso_pix = nones
    weights = hdul["WEIGHT"].read().astype(float)
    w = weights > 0
    cont = hdul["CONT"].read().astype(float)

    if "THING_ID" in hdul["METADATA"].get_colnames():
        los_id = hdul["METADATA"]["THING_ID"][:]
        plate = hdul["METADATA"]["PLATE"][:]
        mjd = hdul["METADATA"]["MJD"][:]
        fiberid=hdul["METADATA"]["FIBERID"][:]
    elif "LOS_ID" in hdul["METADATA"].get_colnames():
        los_id = hdul["METADATA"]["LOS_ID"][:]
        plate=los_id
        mjd=los_id
        fiberid=los_id
    else:
        raise Exception("Could not find THING_ID or LOS_ID")

    ra = hdul["METADATA"]["RA"][:]
    dec = hdul["METADATA"]["DEC"][:]
    z_qso = hdul["METADATA"]["Z"][:]
    try:
        order = hdul["METADATA"]["ORDER"][:]
    except (KeyError, ValueError):
        order = np.full(N_forests, 1)

    deltas = []
    for (los_id_i, ra_i, dec_i, z_qso_i, plate_i, mjd_i, fiberid_i, log_lambda,
        weights_i, cont_i, delta_i, order_i, ivar_i, exposures_diff_i, mean_snr_i,
        mean_reso_i, mean_z_i, resolution_matrix_i,
        mean_resolution_matrix_i, mean_reso_pix_i, w_i
    ) in zip(los_id, ra, dec, z_qso, plate, mjd, fiberid, repeat(log_lambda),
               weights, cont, delta, order, ivar, exposures_diff, mean_snr,
               mean_reso, mean_z, resolution_matrix,
               mean_resolution_matrix, mean_reso_pix, w):
        deltas.append(cls(
            los_id_i, ra_i, dec_i, z_qso_i, plate_i, mjd_i, fiberid_i, log_lambda[w_i],
            weights_i[w_i] if weights_i is not None else None,
            cont_i[w_i],
            delta_i[w_i],
            order_i,
            ivar_i[w_i] if ivar_i is not None else None,
            exposures_diff_i[w_i] if exposures_diff_i is not None else None,
            mean_snr_i, mean_reso_i, mean_z_i,
            resolution_matrix_i if resolution_matrix_i is not None else None,
            mean_resolution_matrix_i if mean_resolution_matrix_i is not None else None,
            mean_reso_pix_i,
        ))

    return deltas

def read_from_hdu(files):
    """Read data with an HDU per forest

    Arguments
    ---------
    files: list of str
    List of all the files to read

    Return
    ------
    tracers: array of Tracer
    The loaded tracers
    """
    pass

def rebin(*args):
    print("Not implemented")
    continue

def project_deltas(*args):
    print("Not implemented")
    continue
