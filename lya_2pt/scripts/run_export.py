#!/usr/bin/env python3
import argparse
from configparser import ConfigParser

from lya_2pt import Interface


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description=('Compute auto-correlation function'))

    parser.add_argument('-i', '--config', type=str, default=None,
                        help=('Path to config file'))

    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.config)

    print('Initializing')
    lya2pt = Interface(config)

    print('Exporting')
    lya2pt.export.run(lya2pt.config, lya2pt.settings)
    print('Done')


if __name__ == '__main__':
    main()
