import os
from os import makedirs
from pathlib import Path

import numpy as np

import lya_2pt
from lya_2pt.errors import ParserError


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
    """Computes the maximum angular separation we need to look for neighbours

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

    r_min = cosmo.get_dist_m(z_min)
    r_min2 = cosmo.get_dist_m(z_min2)
    r_sum = r_min + r_min2

    if r_sum < rt_max:
        ang_max = np.pi
    else:
        ang_max = 2. * np.arcsin(rt_max / r_sum)

    return ang_max


def find_path(path, enforce=True):
    """ Find paths on the system.

    Parameters
    ----------
    path : string
        Input path. Can be absolute or relative to lya_2pt
    enforce : bool
        Flag for enforcing that the path exists
    """
    input_path = Path(os.path.expandvars(path))

    # First check if it's an absolute path
    if input_path.exists():
        return input_path.resolve()

    # Get the vega path and check inside vega (this returns vega/vega)
    lya2pt_path = Path(os.path.dirname(lya_2pt.__file__))

    # Check the lya2pt folder
    in_lya2pt = lya2pt_path / input_path
    if in_lya2pt.exists():
        return in_lya2pt.resolve()

    # Check if it's something used for tests
    in_tests = lya2pt_path.parents[0] / 'tests' / input_path
    if in_tests.exists():
        return in_tests.resolve()

    # Check from the main lya2pt folder
    in_main = lya2pt_path.parents[0] / input_path
    if in_main.exists():
        return in_main.resolve()

    # Check the lya2pt bin folder
    in_bin = lya2pt_path.parents[0] / 'bin' / input_path
    if in_bin.exists():
        return in_bin.resolve()

    if not enforce:
        print(f'Warning, the path/file was not found: {input_path}')
        return input_path
    else:
        raise RuntimeError(f'The path/file does not exist: {input_path}')


def check_dir(dir: Path):
    """
    Checks that a directory exists, and that its permission group is DESI.
    Args:
        dir: Path
            Directory to check
    """
    makedirs(dir, exist_ok=True)
