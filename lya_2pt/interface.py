from mpi4py import MPI
import numpy as np
from lya_2pt.forest_reader import ForestReader
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
        self.mpi_rank = self.comm.Get_rank()
        self.mpi_size = self.comm.Get_size()

        # Find files
        files = glob.glob(config['reader'].get('input directory'))

        # Read computation data
        for rank, file in enumerate(files):
            if rank == self.mpi_rank:   
                forest_reader = ForestReader(config, file)
                neighbour_ids = forest_reader.find_healpix_neighbours()

                tracer2_reader = Tracer2Reader(config, neighbour_ids)

                tracers2 = tracer2_reader.tracers
                forest_reader.find_neighbours(tracer2)
                tracers1 = forest_reader.tracers

                output = None
                if config['compute'].getboolean('compute xi'):
                    output = compute_xi(tracers1, tracers2, config)

                self.write_healpix_output(output)

    def write_healpix_output(self, output):
        pass
