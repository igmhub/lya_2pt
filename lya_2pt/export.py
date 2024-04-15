import fitsio
import numpy as np
from multiprocessing import Pool

from lya_2pt.utils import parse_config
from lya_2pt import defaults

# accepted_options = [
#     "export-correlation", "export-distortion", "smooth-covariance"
# ]

# defaults = {
#     "export-correlation": False,
#     "export-distortion": False,
#     "smooth-covariance": True,
# }


class Export:
    """Class for handling export operations
    Reads output healpix files with each correaltion/distortion matrix
    Computes the mean and covariance matrix of the samples
    Writes final correlation
    """
    def __init__(self, config, name, output_directory, num_cpu):
        self.config = parse_config(config, defaults.export)

        self.num_cpu = num_cpu
        self.name = name
        self.output_directory = output_directory
        self.healpix_dir = self.output_directory / f'healpix_files_{self.name}'
        assert self.healpix_dir.is_dir()

        self.export_correlation = self.config.getboolean('export-correlation')
        self.export_distortion = self.config.getboolean('export-distortion')

    def run(self, global_config, settings):
        if self.export_correlation:
            self.read_correlations()

            # TODO Add more other covariance options
            self.compute_covariance()

            self.write_correlation(global_config, settings)

        if self.export_distortion:
            self.read_distortion()
            self.write_distortion(global_config, settings)

        # self.distortion = None
        # self.distortion_flag = config.getboolean('distortion')
        # if self.distortion_flag:
        #     distortion_dir = config.get('distortion dir')
        #     # TODO
        #     pass

        # self.metal_matrix_flag = config.getboolean('metal matrices')
        # if self.metal_matrix_flag:
        #     metal_matrix_dir = config.get('metal matrix dir')
        #     # TODO
        #     pass

    def read_correlations(self):
        files = np.array(list(self.healpix_dir.glob('correlation*fits*')))

        with fitsio.FITS(files[0]) as hdul:
            header = hdul[1].read_header()
            self.r_par_min = header['R_PAR_MIN']
            self.r_par_max = header['R_PAR_MAX']
            self.r_trans_max = header['R_TRANS_MAX']
            self.num_bins_r_par = header['NUM_BINS_R_PAR']
            self.num_bins_r_trans = header['NUM_BINS_R_TRANS']

        self.delta_r_par = (self.r_par_max - self.r_par_min) / self.num_bins_r_par
        self.delta_r_trans = self.r_trans_max / self.num_bins_r_trans

        if self.num_cpu > 1:
            with Pool(processes=self.num_cpu) as pool:
                results = pool.map(self._read_correlation, files)
        else:
            results = [self._read_correlation(file) for file in files]

        results = np.array(results)
        self.correlations = results[:, 0, :]
        self.weights = results[:, 1, :]
        self.mean_correlation = np.sum(self.correlations * self.weights, axis=0)
        self.r_par = np.sum(results[:, 2, :] * self.weights, axis=0)
        self.r_trans = np.sum(results[:, 3, :] * self.weights, axis=0)
        self.z_grid = np.sum(results[:, 4, :] * self.weights, axis=0)
        self.num_pairs = np.sum(results[:, 5, :], axis=0)

        self.sum_weights = np.sum(self.weights, axis=0)
        w = self.sum_weights > 0
        self.mean_correlation[w] /= self.sum_weights[w]
        self.r_par[w] /= self.sum_weights[w]
        self.r_trans[w] /= self.sum_weights[w]
        self.z_grid[w] /= self.sum_weights[w]

    def _read_correlation(self, file):
        with fitsio.FITS(file) as hdul:
            rp = hdul[1]['R_PAR'][:]
            rt = hdul[1]['R_TRANS'][:]
            z = hdul[1]['Z'][:]
            num_pairs = hdul[1]['NUM_PAIRS'][:]

            # TODO implement blinding support
            correlation = hdul[2]['CORRELATION'][:]
            weights = hdul[2]['WEIGHT_SUM'][:]

        return correlation, weights, rp, rt, z, num_pairs

    def read_distortion(self):
        files = np.array(list(self.healpix_dir.glob('distortion*fits*')))

        with fitsio.FITS(files[0]) as hdul:
            header = hdul[1].read_header()
            self.dist_rp_min = header['R_PAR_MIN']
            self.dist_rp_max = header['R_PAR_MAX']
            self.dist_rt_max = header['R_TRANS_MAX']
            self.dist_rp_size = header['NUM_BINS_R_PAR']
            self.dist_rt_size = header['NUM_BINS_R_TRANS']

        if self.num_cpu > 1:
            with Pool(processes=self.num_cpu) as pool:
                results = pool.map(self._read_distortion, files)
        else:
            results = [self._read_distortion(file) for file in files]

        results = list(results)
        self.distortion = np.array([item[0] for item in results]).sum(axis=0)
        self.dist_weights = np.array([item[1] for item in results]).sum(axis=0)
        self.dist_rp = np.array([item[2] for item in results]).sum(axis=0)
        self.dist_rt = np.array([item[3] for item in results]).sum(axis=0)
        self.dist_z = np.array([item[4] for item in results]).sum(axis=0)
        self.dist_eff_weights = np.array([item[5] for item in results]).sum(axis=0)
        self.dist_num_pairs = np.array([item[6] for item in results]).sum(axis=0)
        self.dist_num_pairs_used = np.array([item[7] for item in results]).sum(axis=0)

        w = self.dist_weights > 0
        self.distortion[w] /= self.dist_weights[w, None]

        w = self.dist_eff_weights > 0
        self.dist_rp[w] /= self.dist_eff_weights[w]
        self.dist_rt[w] /= self.dist_eff_weights[w]
        self.dist_z[w] /= self.dist_eff_weights[w]

    def _read_distortion(self, file):
        with fitsio.FITS(file) as hdul:
            header = hdul[1].read_header()
            num_pairs = header['NUM_PAIRS']
            num_pairs_used = header['PAIRS_USED']

            rp = hdul[1]['R_PAR'][:]
            rt = hdul[1]['R_TRANS'][:]
            z = hdul[1]['Z'][:]
            eff_weights = hdul[1]['EFF_WEIGHTS'][:]

            # TODO implement blinding support
            distortion = hdul[2]['DISTORTION'][:]
            weights = hdul[2]['DISTORTION_WEIGHTS'][:]

        return distortion, weights, rp, rt, z, eff_weights, num_pairs, num_pairs_used

    def compute_covariance(self):
        meanless_xi_times_weights = self.weights * (self.correlations - self.mean_correlation)

        covariance = meanless_xi_times_weights.T.dot(meanless_xi_times_weights)
        sum_weights_squared = np.outer(self.sum_weights, self.sum_weights)
        w = sum_weights_squared > 0.
        covariance[w] /= sum_weights_squared[w]

        if self.config.getboolean('smooth-covariance'):
            print('Smoothing covariance matrix')
            covariance = self.smooth_covariance(covariance)

        self.covariance = covariance

    def smooth_covariance(self, covariance):
        num_bins = covariance.shape[1]
        var = np.diagonal(covariance)
        if np.any(var == 0):
            raise ValueError('Covariance has at least one 0 on the diagonal. Cannot smooth.')
        elif np.any(var < 0):
            raise ValueError('Covariance has at least one negative value on the diagonal. '
                             'Cannot smooth.')

        correlation = covariance / np.outer(np.sqrt(var), np.sqrt(var))
        correlation_smooth = np.zeros([num_bins, num_bins])

        # add together the correlation from bins with similar separations in
        # parallel and perpendicular distances
        sum_correlation = {}
        counts_correlation = {}
        for i in range(num_bins):
            print("\rSmoothing {}".format(i + 1), end="")
            for j in range(i + 1, num_bins):
                ind_drp = round(abs(self.r_par[j] - self.r_par[i]) / self.delta_r_par)
                ind_drt = round(abs(self.r_trans[i] - self.r_trans[j]) / self.delta_r_trans)
                if (ind_drp, ind_drt) not in sum_correlation:
                    sum_correlation[(ind_drp, ind_drt)] = 0
                    counts_correlation[(ind_drp, ind_drt)] = 0

                sum_correlation[(ind_drp, ind_drt)] += correlation[i, j]
                counts_correlation[(ind_drp, ind_drt)] += 1

        for i in range(num_bins):
            correlation_smooth[i, i] = 1.
            for j in range(i + 1, num_bins):
                ind_drp = round(abs(self.r_par[j] - self.r_par[i]) / self.delta_r_par)
                ind_drt = round(abs(self.r_trans[i] - self.r_trans[j]) / self.delta_r_trans)
                correlation_smooth[i, j] = (sum_correlation[(ind_drp, ind_drt)]
                                            / counts_correlation[(ind_drp, ind_drt)])
                correlation_smooth[j, i] = correlation_smooth[i, j]

        print("\n")
        covariance_smooth = correlation_smooth * np.outer(np.sqrt(var), np.sqrt(var))
        return covariance_smooth

    def write_correlation(self, global_config, settings):
        output_file = self.output_directory / f'{self.name}-exp.fits.gz'
        results = fitsio.FITS(output_file, 'rw', clobber=True)

        # distortion = self.distortion
        distortion = None
        if distortion is None:
            distortion = np.eye(len(self.covariance))

        header = [{
            'name': 'R_PAR_MIN',
            'value': settings.getfloat('rp_min'),
            'comment': 'Minimum r-parallel [h^-1 Mpc]'
        }, {
            'name': 'R_PAR_MAX',
            'value': settings.getfloat('rp_max'),
            'comment': 'Maximum r-parallel [h^-1 Mpc]'
        }, {
            'name': 'R_TRANS_MAX',
            'value': settings.getfloat('rt_max'),
            'comment': 'Maximum r-transverse [h^-1 Mpc]'
        }, {
            'name': 'NUM_BINS_R_PAR',
            'value': settings.getint('num_bins_rp'),
            'comment': 'Number of bins in r-parallel'
        }, {
            'name': 'NUM_BINS_R_TRANS',
            'value': settings.getint('num_bins_rt'),
            'comment': 'Number of bins in r-transverse'
        }, {
            'name': 'Z_MIN',
            'value': settings.getfloat('z_min'),
            'comment': 'Minimum redshift of pairs'
        }, {
            'name': 'Z_MAX',
            'value': settings.getfloat('z_max'),
            'comment': 'Maximum redshift of pairs'
        }, {
            'name': 'OMEGA_M',
            'value': global_config['cosmology'].getfloat('Omega_m'),
            'comment': 'Omega_matter(z=0) of fiducial LambdaCDM cosmology'
        }, {
            'name': "BLINDING",
            'value': 'placeholder',  # TODO Correct this once blinding implemented
            'comment': 'String specifying the blinding strategy'
        }]

        comment = ['R-parallel', 'R-transverse', 'Redshift', 'Correlation',
                   'Covariance matrix', 'Distortion matrix', 'Number of pairs']
        results.write(
            [self.r_par, self.r_trans, self.z_grid, self.mean_correlation,
             self.covariance, distortion, self.num_pairs],
            names=['RP', 'RT', 'Z', 'DA', 'CO', 'DM', 'NB'],
            comment=comment,
            header=header,
            extname='COR'
        )

        comment = ['R-parallel model', 'R-transverse model', 'Redshift model']
        results.write(
            [self.r_par, self.r_trans, self.z_grid],
            names=['DMRP', 'DMRT', 'DMZ'],
            comment=comment,
            extname='DMATRIX'
        )
        results.close()

    def write_distortion(self, global_config, settings):
        output_file = self.output_directory / f'dmat_{self.name}-exp.fits.gz'
        results = fitsio.FITS(output_file, 'rw', clobber=True)

        header = [{
            'name': 'R_PAR_MIN',
            'value': settings.getfloat('rp_min'),
            'comment': 'Minimum r-parallel [h^-1 Mpc]'
        }, {
            'name': 'R_PAR_MAX',
            'value': settings.getfloat('rp_max'),
            'comment': 'Maximum r-parallel [h^-1 Mpc]'
        }, {
            'name': 'R_TRANS_MAX',
            'value': settings.getfloat('rt_max'),
            'comment': 'Maximum r-transverse [h^-1 Mpc]'
        }, {
            'name': 'NUM_BINS_R_PAR',
            'value': settings.getint('num_bins_rp'),
            'comment': 'Number of bins in r-parallel'
        }, {
            'name': 'NUM_BINS_R_TRANS',
            'value': settings.getint('num_bins_rt'),
            'comment': 'Number of bins in r-transverse'
        }, {
            'name': 'Z_MIN',
            'value': settings.getfloat('z_min'),
            'comment': 'Minimum redshift of pairs'
        }, {
            'name': 'Z_MAX',
            'value': settings.getfloat('z_max'),
            'comment': 'Maximum redshift of pairs'
        }, {
            'name': 'REJECTION_FRAC',
            'value': settings.getfloat('rejection_fraction'),
            'comment': 'Rejection fraction when computing distortion'
        }, {
            'name': 'NUM_PAIRS',
            'value': self.dist_num_pairs,
            'comment': 'Healpix nside'
        }, {
            'name': 'PAIRS_USED',
            'value': self.dist_num_pairs_used,
            'comment': 'Healpix nside'
        }, {
            'name': 'OMEGA_M',
            'value': global_config['cosmology'].getfloat('Omega_m'),
            'comment': 'Omega_matter(z=0) of fiducial LambdaCDM cosmology'
        }, {
            'name': "BLINDING",
            'value': 'placeholder',  # TODO Correct this once blinding implemented
            'comment': 'String specifying the blinding strategy'
        }]

        results.write(
            [self.dist_weights, self.distortion],
            names=['WDM', 'DM'],
            comment=['Sum of weights', 'Distortion matrix'],
            header=header,
            extname='COR'
        )

        results.write(
            [self.dist_rp, self.dist_rt, self.dist_z],
            names=['RP', 'RT', 'Z'],
            comment=['R-parallel', 'R-transverse', 'Redshift'],
            units=['h^-1 Mpc', 'h^-1 Mpc', ''],
            extname='DMATRIX'
        )
        results.close()
