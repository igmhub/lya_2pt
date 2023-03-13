import numpy as np
from math import isclose
from configparser import ConfigParser

from lya_2pt.forest_healpix_reader import ForestHealpixReader
from lya_2pt.tracer2_reader import Tracer2Reader
from lya_2pt.correlation import compute_xi
from lya_2pt.cosmo import Cosmology
from lya_2pt.utils import compute_ang_max, find_path


def read_tracers(config, file, cosmo, healpix_id, nside, ang_max, auto_flag, z_min, z_max):
    """Read the tracers

    Arguments
    ---------
    config: configparser.ConfigParser
    Configuration options

    file: str
    Main healpix file

    cosmo: Cosmology
    Fiducial cosmology used to go from angles and redshift to distances

    Return
    ------
    tracer1: array of Tracer
    First set of tracers, with neightbours computed

    tracer2: array of Tracers
    Second set of tracers
    """
    # read tracers 1
    forest_reader = ForestHealpixReader(config["tracer1"], file, cosmo, 1, healpix_id)
    healpix_neighbours_ids = forest_reader.find_healpix_neighbours(nside, ang_max)

    # read tracers 2 - auto correlation
    if auto_flag:
        forest_reader.auto_flag = True
        tracer2_reader = Tracer2Reader(config["tracer1"], healpix_neighbours_ids, cosmo, 1)
        tracer2_reader.add_tracers(forest_reader)
    # read tracers 2 - cross correlation
    else:
        tracer2_reader = Tracer2Reader(
            config["tracer2"], healpix_neighbours_ids, cosmo, 1)

    forest_reader.find_neighbours(tracer2_reader, z_min, z_max, ang_max, 1)

    tracers1 = forest_reader.tracers
    tracers2 = tracer2_reader.tracers

    return tracers1, tracers2, forest_reader.blinding


def test_cf():
    config = ConfigParser()
    config.read(find_path("configs/lyaxlya_cf.ini"))

    # intialize cosmology
    cosmo = Cosmology(config["cosmology"])

    # parse config
    settings = config["settings"]
    z_min = settings.getfloat("z_min")
    z_max = settings.getfloat("z_max")
    nside = settings.getint("nside")
    num_cpu = settings.getint("num processors")
    # maximum angle for two lines-of-sight to have neightbours
    ang_max = compute_ang_max(cosmo, settings.getfloat('rt_max'), z_min)
    # check if we are working with an auto-correlation
    auto_flag = "tracer2" not in config

    # Find files
    input_directory = find_path(config["tracer1"].get("input directory"))
    files = np.array(sorted(input_directory.glob('*fits*')))

    output = []    
    for file in files:
        # Figure out healpix id of the file
        healpix_id = int(file.name.split("delta-")[-1].split(".fits")[0])

        tracers1, tracers2, blinding = read_tracers(config, file, cosmo, healpix_id, nside,
                                                    ang_max, auto_flag, z_min, z_max)

        # do the actual computation
        output.append(compute_xi(tracers1, tracers2, config['settings'], 1))

    assert isclose(np.sum(output[0][0]), -0.018011306751563733)
    assert isclose(np.sum(output[1][0]), -0.24107177381673406)

    assert isclose(np.sum(output[0][1]), 3474396919.7082357)
    assert isclose(np.sum(output[1][1]), 3161197176.2617044)

    assert isclose(np.sum(output[0][2]), 165020.50799382737)
    assert isclose(np.sum(output[1][2]), 147679.3687192793)

    assert isclose(np.sum(output[0][3]), 108795.79014303378)
    assert isclose(np.sum(output[1][3]), 86392.82815556126)

    assert isclose(np.sum(output[0][4]), 4017.489007665773)
    assert isclose(np.sum(output[1][4]), 3442.8489486736116)

    assert isclose(np.sum(output[0][5]), 7610311)
    assert isclose(np.sum(output[1][5]), 8255152)
