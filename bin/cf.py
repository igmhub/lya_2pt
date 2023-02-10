import argparse
from mpi4py import MPI
from configparser import ConfigParser

from lya_2pt.interface import Interface


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description=('Compute auto-correlation function'))

    parser.add_argument('-i', '--config', type=str, default=None,
                        help=('Path to config file'))
    
    parser.add_argument('-t', '--test-run', action='store_true', required=False,
                        help=('Flag for test run.'))

    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.config)

    # Initilize MPI objects
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    lya2pt_interface = Interface(config, mpi_rank, mpi_size)
    lya2pt_interface.run(test_run=args.test_run)
