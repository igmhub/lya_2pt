#!/usr/bin/env python3

import argparse
import time
from configparser import ConfigParser

from lya_2pt.interface import Interface
from lya_2pt.errors import MPIError


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description=('Compute auto-correlation function'))

    parser.add_argument('-i', '--config', type=str, default=None,
                        help=('Path to config file'))

    parser.add_argument('-r', '--rank', type=int, default=0, help=('Rank of the job'))
    parser.add_argument('-s', '--size', type=int, default=1, help=('Number of jobs'))

    args = parser.parse_args()

    if args.rank >= args.size:
        raise MPIError(f"Rank {args.rank} is greater than the number of MPI processes {args.size}")

    config = ConfigParser()
    config.read(args.config)

    lya2pt = Interface(config)

    if len(lya2pt.files) < args.size:
        raise MPIError(f"Less files than MPI processes. "
                       f"Found {len(lya2pt.files)} healpix files and running "
                       f"{args.size} MPI processes. This is wasteful. "
                       "Please lower the numper of MPI processes.")

    total_t1 = time.time()

    print(f'Rank {args.rank}: Reading tracers...', flush=True)
    lya2pt.read_tracers(mpi_rank=args.rank)

    print(f'Rank {args.rank}: Starting computation...', flush=True)
    lya2pt.run(mpi_size=args.size, mpi_rank=args.rank)

    print(f'Rank {args.rank}: Writing results...', flush=True)
    lya2pt.write_results(mpi_rank=args.rank)

    total_t2 = time.time()
    print(f'Rank {args.rank}: Total time: {(total_t2-total_t1):.3f} sec', flush=True)
    print(f'Rank {args.rank}: Done', flush=True)


if __name__ == '__main__':
    main()
