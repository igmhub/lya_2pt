import numpy as np

from lya_2pt.constants import ABSORBER_IGM
from lya_2pt.errors import ReaderException
from lya_2pt.tracer import Tracer


def read_from_image(hdul, absorption_line, healpix_id, need_distortion=False, projection_order=1):
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
    dwave = hdul["LAMBDA"].read_header()['DELTA_LAMBDA']

    accepted_deltas = set(["DELTA", "DELTA_BLIND"])
    key = accepted_deltas.intersection([hdul[3].get_extname()])
    if not key:
        raise Exception("Delta table not found")
    key = key.pop()
    deltas_array = hdul[key].read().astype(float)

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

    ivar_array = np.zeros_like(deltas_array)
    if "IVAR" in hdul:
        ivar_array = hdul['IVAR'].read().astype(float)

    tracers = np.empty(los_id_array.shape, dtype=Tracer)
    for i, (los_id, ra, dec) in enumerate(zip(los_id_array, ra_array, dec_array)):
        mask = ~np.isnan(deltas_array[i])
        tracers[i] = Tracer(
            healpix_id, los_id, ra, dec, projection_order, deltas_array[i][mask],
            weights_array[i][mask], ivar_array[i][mask], log_lambda[mask], z[mask], need_distortion
            )

    return tracers, wave_solution, dwave


def read_from_hdu(hdul, absorption_line, healpix_id, need_distortion=False, projection_order=1):
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
    header = hdul[1].read_header()
    if 'DELTA_LAMBDA' in header:
        dwave = ['DELTA_LAMBDA']
    else:
        dwave = None

    tracers = []
    wave_solution = None
    for hdu in hdul[1:]:
        header = hdu.read_header()

        los_id = header["LOS_ID"]
        ra = header['RA']
        dec = header['DEC']

        accepted_deltas = set(["DELTA", "DELTA_BLIND"])
        key = accepted_deltas.intersection(hdu.get_colnames())
        if not key:
            raise Exception("Delta column not found")
        key = key.pop()
        delta = hdu[key][:].astype(float)

        weights = hdu["WEIGHT"][:].astype(float)
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

        mask = weights > 0
        ivar = np.zeros_like(delta)
        if 'IVAR' in hdu.get_colnames():
            ivar = hdu['IVAR'][:].astype(float)
            mask &= ivar > 0

        tracers.append(Tracer(
            healpix_id, los_id, ra, dec, projection_order, delta[mask], weights[mask], ivar[mask],
            log_lambda[mask], z[mask], need_distortion
            ))

    return np.array(tracers), wave_solution, dwave
