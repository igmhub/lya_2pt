import glob
import numpy as np
from mpi4py import MPI

from lya_2pt.forest_healpix_reader import ForestHealpixReader
from lya_2pt.tracer2_reader import Tracer2Reader
from lya_2pt.correlation import compute_xi
from lya_2pt.cosmo import Cosmology
from lya_2pt.utils import MPIError, parse_config, compute_ang_max

accepted_options = ["z_min", "z_max", "nside", "rp_min", "rp_max", "rt_max",
                    "num_bins_rp", "num_bins_rt"]

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
    """
    def __init__(self, config):
        # Initilize MPI objects
        self.mpi_comm = MPI.COMM_WORLD
        self.mpi_rank = self.mpi_comm.Get_rank()
        self.mpi_size = self.mpi_comm.Get_size()

        # intialize cosmology
        cosmo = Cosmology(config["cosmology"])

        settings = parse_config(config['settings'], defaults, accepted_options)

        self.z_min = settings.getfloat('z_min')
        self.z_max = settings.getfloat('z_max')
        self.nside = settings.getint('nside')
        self.ang_max = compute_ang_max(cosmo, settings.getfloat('rt_max'), self.z_min)

        # Find files
        input_directory = config['data'].get('input_directory')
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

        # Read computation data
        for file in files[start:stop]:
            forest_reader = ForestHealpixReader(config['reader'], file, cosmo)
            neighbour_ids = forest_reader.find_healpix_neighbours(self.nside, self.ang_max)

            tracer2_reader = Tracer2Reader(config, neighbour_ids, cosmo)

            # Check if we are working with an auto-correlation
            if tracer2_reader.auto_flag:
                tracer2_reader.add_tracers1(forest_reader.tracers)

            tracers2 = tracer2_reader.tracers
            forest_reader.find_neighbours(tracers2, self.z_min, self.z_max)
            tracers1 = forest_reader.tracers

            output = None
            if config['compute'].getboolean('compute xi'):
                output = compute_xi(tracers1, tracers2, config)

            self.write_healpix_output(output)

    def write_healpix_output(self, output):
        pass
