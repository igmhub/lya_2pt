import glob
import numpy as np
from mpi4py import MPI

from lya_2pt.forest_healpix_reader import ForestHealpixReader
from lya_2pt.tracer2_reader import Tracer2Reader
from lya_2pt.correlation import compute_xi


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

        # Find files
        files = np.array(glob.glob(config['reader'].get('input directory')))

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
            forest_reader = ForestHealpixReader(config, file)
            neighbour_ids = forest_reader.find_healpix_neighbours()

            tracer2_reader = Tracer2Reader(config, neighbour_ids)

            # Check if we are working with an auto-correlation
            if True:
                tracer2_reader.add_tracers1(forest_reader.tracers)

            tracers2 = tracer2_reader.tracers
            forest_reader.find_neighbours(tracers2)
            tracers1 = forest_reader.tracers

            output = None
            if config['compute'].getboolean('compute xi'):
                output = compute_xi(tracers1, tracers2, config)

            self.write_healpix_output(output)

    def write_healpix_output(self, output):
        pass
