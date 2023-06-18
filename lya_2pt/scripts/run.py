#!/usr/bin/env python3
import argparse
import time
from configparser import ConfigParser

from lya_2pt.interface import Interface


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description=('run lya-2pt'))

    parser.add_argument('-i', '--config', type=str, default=None,
                        help=('Path to config file'))

    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.config)

    print('Initializing')
    lya2pt = Interface(config)

    print('Reading tracers')
    t1 = time.time()
    lya2pt.read_tracers()
    t2 = time.time()
    print(f'Time reading: {(t2-t1):.3f}')

    print('Computing correlation')
    t1 = time.time()
    lya2pt.run()
    t2 = time.time()
    print(f'Time computing: {(t2-t1):.3f}')

    print('Writing results')
    lya2pt.write_results()

    print('Exporting')
    lya2pt.export.run(lya2pt.config, lya2pt.settings)
    print('Done')


if __name__ == '__main__':
    main()
