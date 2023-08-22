#!/usr/bin/env python3

import argparse
import time
from mpi4py import MPI
from configparser import ConfigParser

from lya_2pt.interface import Interface
from lya_2pt.errors import MPIError


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description=('Compute auto-correlation function'))

    parser.add_argument('-i', '--config', type=str, default=None,
                        help=('Path to config file'))

    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.config)

    # Initilize MPI objects
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    lya2pt = Interface(config)

    if len(lya2pt.files) < mpi_size:
        raise MPIError(f"Less files than MPI processes. "
                       f"Found {len(lya2pt.files)} healpix files and running "
                       f"{mpi_size} MPI processes. This is wasteful. "
                       "Please lower the numper of MPI processes.")

    num_tasks_per_proc = len(lya2pt.files) // mpi_size
    remainder = len(lya2pt.files) % mpi_size
    if mpi_rank < remainder:
        start = int(mpi_rank * (num_tasks_per_proc + 1))
        stop = int(start + num_tasks_per_proc + 1)
    else:
        start = int(mpi_rank * num_tasks_per_proc + remainder)
        stop = int(start + num_tasks_per_proc)

    if mpi_rank == 0:
        total_t1 = time.time()
        print('Starting computation...', flush=True)

    lya2pt.read_tracers(lya2pt.files[start:stop])
    lya2pt.run(mpi_rank=mpi_rank)
    lya2pt.write_results(mpi_rank=mpi_rank)

    if mpi_rank == 0:
        total_t2 = time.time()
        print(f'Total time: {(total_t2-total_t1):.3f} sec', flush=True)
        print('Done', flush=True)


if __name__ == '__main__':
    main()
