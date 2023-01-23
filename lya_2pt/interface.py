import glob
import numpy as np
from mpi4py import MPI

from lya_2pt.errors import MPIError
from lya_2pt.forest_healpix_reader import ForestHealpixReader
from lya_2pt.tracer2_reader import Tracer2Reader
from lya_2pt.correlation import compute_xi
from lya_2pt.cosmo import Cosmology
from lya_2pt.utils import parse_config, compute_ang_max

accepted_options = [
    "num_bins_rp", "num_bins_rt", "nside", "rp_min", "rp_max", "rt_max",
    "z_min", "z_max",
]

defaults = {
    "z_min": 0,
    "z_max": 10,
    "nside": 16,
    "rp_min": 0,
    "rp_max": 200,
    "rt_max": 200,
    "num_bins_rp": 50,
    "num_bins_rt": 50
}


class Interface:
    """Interface for lya_2pt package
    Read ini files
    Handle parallezation
        - Read data
        - Call individual compute functions
    Write outputs

    Methods
    -------
    __init__
    read_tracers
    run_computation
    write_healpix_output

    Attributes
    ----------
    ang_max: float
    Maximum angle for two lines-of-sight to have neightbours

    auto_flag: bool
    True if we are working with an auto-correlation, False for cross-correlation
    Initialized to False

    mpi_comm: ?
    ?

    mpi_rank: ?
    ?

    mpi_size: ?
    ?

    nside: int
    Nside parameter to construct the healpix pixels

    z_max: float
    Maximum redshfit of the tracers

    z_min: float
    Minimum redshift of the tracers
    """
    def __init__(self, config):
        """Initialize class instance

        Arugments
        ---------
        config: configparser.ConfigParser
        Configuration options
        """
        # Initilize MPI objects
        self.mpi_comm = MPI.COMM_WORLD
        self.mpi_rank = self.mpi_comm.Get_rank()
        self.mpi_size = self.mpi_comm.Get_size()

        # intialize cosmology
        cosmo = Cosmology(config["cosmology"])

        # parse config
        settings = parse_config(config['settings'], defaults, accepted_options)
        self.z_min = settings.getfloat('z_min')
        self.z_max = settings.getfloat('z_max')
        self.nside = settings.getint('nside')
        # maximum angle for two lines-of-sight to have neightbours
        self.ang_max = compute_ang_max(
            cosmo, settings.getfloat('rt_max'), self.z_min)
        # check if we are working with an auto-correlation
        self.auto_flag = "tracer2" in config

        # Find files
        input_directory = config['tracer1'].get('input_directory')
        files = np.array(glob.glob(input_directory + '/*fits*'))

        if len(files) < self.mpi_size:
            raise MPIError(f"Less files in {input_directory} than MPI processes. "
                           f"Found {len(files)} healpix files and running "
                           f"{self.mpi_size} MPI processes. This is wasteful. "
                            "Please lower the numper of MPI processes.")

        num_tasks_per_proc = len(files) / self.mpi_size
        remainder = len(files) % self.mpi_size
        if self.mpi_rank < remainder:
            start = self.mpi_rank * (num_tasks_per_proc + 1)
            stop = start + num_tasks_per_proc + 1
        else:
            start = self.mpi_rank * num_tasks_per_proc + remainder
            stop = start + num_tasks_per_proc

        # Loop over the healpixes this thread is responsible for
        for file in files[start:stop]:
            # read tracers
            tracers1, tracers2 = self.read_tracers(config, file, cosmo)

            # do the actual computation
            output = self.run_computation(config['compute'], tracers1, tracers2)

            # write output
            self.write_healpix_output(output)

    def read_tracers(self, config, file, cosmo):
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
        forest_reader = ForestHealpixReader(config["tracer1"], file, cosmo)
        healpix_neighbours_ids = forest_reader.find_healpix_neighbours(
            self.nside, self.ang_max)

        # read tracers 2 - auto correlation
        if self.auto_flag:
            tracer1.auto_flag = True
            tracer2_reader = Tracer2Reader(
                config["tracer1"], healpix_neighbours_ids, cosmo)
            # TODO: check this
            # Ignasi: I'm worried we are adding this twice
            # If healpix_neighbours_ids has the healpix id of tracers1
            # then we are. I presume it's the case as otherwise, we are
            # not reading the main healpix for tracer2 in the cross-correlation
            # case.
            # To fix this, we should use auto_flag in method find_healpix_neighbours
            # to not include the main healpix in the list of healpixes
            # We could then initialize forest_reader.auto_flag in that function
            tracer2_reader.add_tracers1(forest_reader)
        # read tracers 2 - cross correlation
        else:
            tracer2_reader = Tracer2Reader(
                config["tracer2"], healpix_neighbours_ids, cosmo)

        forest_reader.find_neighbours(tracer2_reader, self.z_min, self.z_max)

        tracers1 = forest_reader.tracers
        tracers2 = tracer2_reader.tracers

        return tracers1, tracers2

    def run_computation(config, tracers1, tracers2):
        """Run the computation

        This can include the correlation function, the distortion matrix,
        and/or the metal distortion matrix, depending on the configuration

        Arguments
        ---------
        config: configparser.SectionProxy
        Configuration options

        tracer1: array of Tracer
        First set of tracers, with neightbours computed

        tracer2: array of Tracers
        Second set of tracers

        Return
        ------
        output: array
        The output from the computation
        """
        if config.getboolean('compute xi'):
            output = compute_xi(tracers1, tracers2, config)


    def write_healpix_output(self, output):
        pass
