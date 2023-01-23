import numpy as np


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


def get_angle(tracer1, tracer2):
    pass


class ParserError(Exception):
    pass


class CosmologyError(Exception):
    pass


class MPIError(Exception):
    pass
