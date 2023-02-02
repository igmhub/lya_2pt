import numpy as np
from numba import njit

from lya_2pt.errors import ParserError, FindBinsError

SMALL_ANGLE_CUT_OFF = 2./3600.*np.pi/180. # 2 arcsec


def parse_config(config, defaults, accepted_options):
    """Parse the given configuration

    Check that all required variables are present
    Load default values for missing optional variables

    Arguments
    ---------
    config: configparser.SectionProxy
    Configuration options

    defaults: dict
    The default options for the given config section

    accepted_options: list of str
    The accepted keys for the given config section

    Return
    ------
    config: configparser.SectionProxy
    Parsed options to initialize class
    """
    # update the section adding the default choices when necessary
    for key, value in defaults.items():
        if key not in config:
            config[key] = str(value)

    # make sure all the required variables are present
    for key in accepted_options:
        if key not in config:
            raise ParserError(f"Missing option {key}")

    # check that all arguments are valid
    for key in config:
        if key not in accepted_options:
            raise ParserError(
                f"Unrecognised option. Found: '{key}'. Accepted options are "
                f"{accepted_options}")

    return config


def compute_ang_max(cosmo, rt_max, z_min, z_min2=None):
    """Computes the maximum anglular separation the correlation should be
    calculated to.

    This angle is given by the maximum transverse separation and the fiducial
    cosmology

    Arguments
    ---------
    comso: Cosmology
    Fiducial cosmology used to go from angles and redshift to distances

    rt_max: float
    Maximum transverse separation

    z_min: float
    Minimum redshift of the first set of data

    z_min2: float or None - default: None
    Minimum redshift of the second set of data. If None, use z_min

    Return
    ------
    ang_max: float
    The maximum anglular separation
    """
    if z_min2 is None:
        z_min2 = z_min

    r_min = cosmo.comoving_transverse_distance(z_min)
    r_min2 = cosmo.comoving_transverse_distance(z_min2)
    r_sum = r_min + r_min2

    if r_sum < rt_max:
        ang_max = np.pi
    else:
        ang_max = 2. * np.arcsin(rt_max / r_sum)

    return ang_max


@njit
def get_angle(x1, y1, z1, ra1, dec1, x2, y2, z2, ra2, dec2):
    """Compute angle between two tracers"""
    cos = x1 * x2 + y1 * y2 + z1 * z2
    if cos >= 1.:
        cos = 1.
    elif cos <= -1.:
        cos = -1.
    angle = np.arccos(cos)

    if ((np.abs(ra2 - ra1) < SMALL_ANGLE_CUT_OFF) & (np.abs(dec2 - dec1) < SMALL_ANGLE_CUT_OFF)):
        angle = np.sqrt((dec2 - dec1)**2 + (np.cos(dec1) * (ra2 - ra1))**2)

    return angle


# def get_angle(tracer1, tracer2):
#     """Compute angle between two tracers of Tracer type

#     Arguments
#     ---------
#     tracer1 : Tracer
#     First tracer

#     tracer2 : Tracer
#     Second tracer

#     Return
#     ------
#     angle: float
#     Angle between tracer1 and tracer2
#     """
#     cos = (tracer2.x_cart * tracer1.x_cart + tracer2.y_cart * tracer1.y_cart
#            + tracer2.z_cart * tracer1.z_cart)

#     if cos >= 1.:
#         cos = 1.
#     elif cos <= -1.:
#         cos = -1.
#     angle = np.arccos(cos)

#     if ((np.abs(tracer2.ra - tracer1.ra) < SMALL_ANGLE_CUT_OFF)
#       & (np.abs(tracer2.dec - tracer1.dec) < SMALL_ANGLE_CUT_OFF)):
#         angle = np.sqrt((tracer2.dec - tracer1.dec)**2
#                         + (np.cos(tracer1.dec) * (tracer2.ra - tracer1.ra))**2)

#     return angle


@njit()
def find_bins(original_array, grid_array, wave_solution):
    """For each element in original_array, find the corresponding bin in grid_array
    Arguments
    ---------
    original_array: array of float
    Read array, e.g. forest.log_lambda
    grid_array: array of float
    Common array, e.g. Forest.log_lambda_grid
    wave_solution: "log" or "lin"
    Specifies whether we want to construct a wavelength grid that is evenly
    spaced on wavelength (lin) or on the logarithm of the wavelength (log)
    Return
    ------
    found_bin: array of int
    An array of size original_array.size filled with values smaller than
    grid_array.size with the bins correspondance
    """
    if wave_solution == "log":
        pass
    elif wave_solution == "lin":
        original_array = 10**original_array
        grid_array = 10**grid_array
    else:
        raise FindBinsError(
            "Error in function find_bins from py/picca/delta_extraction/utils.py"
            "expected wavelength solution to be either 'log' or 'lin'. ")
    original_array_size = original_array.size
    grid_array_size = grid_array.size
    found_bin = np.zeros(original_array_size, dtype=np.int64)
    for index1 in range(original_array_size):
        min_dist = np.finfo(np.float64).max
        for index2 in range(grid_array_size):
            dist = np.abs(grid_array[index2] - original_array[index1])
            if dist < min_dist:
                min_dist = dist
                found_bin[index1] = index2
            else:
                break
    return found_bin
