import numpy as np

from lya_2pt.constants import ABSORBER_IGM
from lya_2pt.errors import ReaderException
from lya_2pt.tracer import Tracer
import lya_2pt.global_data as globals
import sys


import pdb
class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

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
    z_qso_array = hdul["METADATA"]["Z"][:]
    dwave = hdul["LAMBDA"].read_header()['DELTA_LAMBDA']

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
    
    #to put true z for qsos
    if globals.true_z_qso is not None:
        los_ids_catalogue = globals.true_z_qso['los_ids']
        true_z_list = globals.true_z_qso['ztrue']

    tracers = np.empty(los_id_array.shape, dtype=Tracer)
    for i, (los_id, ra, dec, z_qso) in enumerate(
        zip(los_id_array, ra_array, dec_array, z_qso_array)
    ):
        if globals.true_z_qso is not None:
            w = los_ids_catalogue == los_id
            z_qso = true_z_list[w]
            if len(z_qso)==0:
                continue
            z_qso = z_qso[0]
        mask = ~np.isnan(deltas_array[i])
        tracers[i] = Tracer(
            healpix_id, los_id, ra, dec, z_qso, projection_order, deltas_array[i][mask],
            weights_array[i][mask], log_lambda[mask], z[mask], need_distortion
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
    dwave = hdul[1].read_header()['DELTA_LAMBDA']

    tracers = []
    wave_solution = None
    for hdu in hdul[1:]:
        header = hdu.read_header()

        los_id = header["LOS_ID"]
        ra = header['RA']
        dec = header['DEC']
        z_qso = header['Z']

        delta = hdu["DELTA"][:].astype(float)
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

        tracers.append(Tracer(
            healpix_id, los_id, ra, dec, z_qso, projection_order, delta, weights,
            log_lambda, z, need_distortion
            ))

    return np.array(tracers), wave_solution, dwave
