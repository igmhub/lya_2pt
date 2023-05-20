import argparse
from configparser import ConfigParser

from lya_2pt.interface import Interface


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description=('Compute auto-correlation function'))

    parser.add_argument('-i', '--config', type=str, default=None,
                        help=('Path to config file'))

    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.config)

    lya2pt = Interface(config)
    lya2pt.read_tracers()
    lya2pt.run()
    lya2pt.write_results()


if __name__ == '__main__':
    main()
