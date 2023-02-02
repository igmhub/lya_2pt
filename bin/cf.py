import argparse
from configparser import ConfigParser

from lya_2pt.interface import Interface

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description=('Compute auto-correlation function'))

    parser.add_argument('-i', '--config', type=str, default=None,
                        help=('Path to config file'))

    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.config)

    Interface(config)
