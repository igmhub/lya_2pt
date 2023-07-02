#!/usr/bin/env python3
import argparse
import time
from configparser import ConfigParser

from lya_2pt.interface import Interface


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=('Compute distortion matrix for auto-correlation function'))

    parser.add_argument('-i', '--input-dir', type=str, default=None,
                        help=('Inpute directory with delta files'))
    parser.add_argument('-o', '--output-dir', type=str, default=None,
                        help=('Output directory to store the correlation in'))
    parser.add_argument('-n', '--name', type=str, default=None,
                        help=('Name of run'))
    parser.add_argument('--rp-min', type=float, default=0, required=False,
                        help=('Minimum line-of-sight separation'))
    parser.add_argument('--rp-max', type=float, default=200, required=False,
                        help=('Maximum line-of-sight separation'))
    parser.add_argument('--rt-max', type=float, default=200, required=False,
                        help=('Maximum transverse separation'))
    parser.add_argument('--num-bins-rp', type=int, default=50, required=False,
                        help=('Number of bins in line-of-sight separation'))
    parser.add_argument('--num-bins-rt', type=int, default=50, required=False,
                        help=('Number of bins in transverse separation'))
    parser.add_argument('--num-bins-rp-model', type=int, default=50, required=False,
                        help=('Number of model bins in line-of-sight separation'))
    parser.add_argument('--num-bins-rt-model', type=int, default=50, required=False,
                        help=('Number of model bins in transverse separation'))
    parser.add_argument('--z-cut-min', type=float, default=0, required=False,
                        help=('Minimum redshift'))
    parser.add_argument('--z-cut-max', type=float, default=10, required=False,
                        help=('Maximum redshift'))
    parser.add_argument('--absorption-line', type=str, default='LYA', required=False,
                        help=('Name of absorption line'))
    parser.add_argument('--Omega-m', type=float, default=0.315, required=False,
                        help=('Matter density parameter'))
    parser.add_argument('--Omega-r', type=float, default=0, required=False,
                        help=('Radiation density parameter'))
    parser.add_argument('--rejection-fraction', type=float, default=0.99, required=False,
                        help=('Fraction of forest pairs to skip when computing the distortion'))
    parser.add_argument('--projection-order', type=int, default=1, required=False,
                        help=('Order of the polynomial used to build the projection'))
    parser.add_argument('--get-new-distortion', action='store_true', required=False,
                        help=('Compute the distortion matrix using the new formalism'))
    parser.add_argument('--rebin-factor', type=int, default=1, required=False,
                        help=('Factor for rebinning forests into coarser lambda bins'))
    parser.add_argument('--num-cpu', type=int, default=1, required=False,
                        help=('Number of cpus when running in parallel'))

    args = parser.parse_args()

    config = ConfigParser()
    config['tracer1'] = {'input-dir': args.input_dir,
                         'tracer-type': 'continuous',
                         'absorption-line': args.absorption_line,
                         'projection-order': str(args.projection_order),
                         'rebin': str(args.rebin_factor)}

    if args.get_new_distortion:
        config['tracer1']['project-deltas'] = 'True'
        config['tracer1']['use-old-projection'] = 'False'

    config['settings'] = {'num-cpu': str(args.num_cpu),
                          'z_min': str(args.z_cut_min),
                          'z_max': str(args.z_cut_max),
                          'rp_min': str(args.rp_min),
                          'rp_max': str(args.rp_max),
                          'rt_max': str(args.rt_max),
                          'num_bins_rp': str(args.num_bins_rp),
                          'num_bins_rt': str(args.num_bins_rt),
                          'num_bins_rp_model': str(args.num_bins_rp_model),
                          'num_bins_rt_model': str(args.num_bins_rt_model),
                          'rejection_fraction': str(args.rejection_fraction),
                          'get-old-distortion': str(not args.get_new_distortion)}

    config['cosmology'] = {'Omega_m': str(args.Omega_m),
                           'Omega_r': str(args.Omega_r)}

    config['compute'] = {'compute-distortion-matrix': 'True'}

    config['output'] = {'name': args.name,
                        'output-dir': args.output_dir}

    config['export'] = {'export-distortion': 'True'}

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
