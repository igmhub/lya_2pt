import fitsio

from lya_2pt.utils import check_dir, parse_config, find_path

accepted_options = [
    "name", "output-dir"
]

defaults = {
    "name": "lyaxlya",
}


class Output:
    def __init__(self, config):
        self.config = parse_config(config, defaults, accepted_options)
        self.name = self.config.get('name')
        self.output_directory = find_path(self.config.get('output-dir'), enforce=False)
        check_dir(self.output_directory)

        self.blinding = None
        self.healpix_dir = self.output_directory / f'healpix_files_{self.name}'
        check_dir(self.healpix_dir)

    def write_cf_healpix(self, output, healpix_id, global_config, settings):
        """Write computation output for the main healpix

        Arguments
        ---------
        config: configparser.SectionProxy
        Configuration options

        file: str
        Name of the read file, used to construct the output file
        """
        filename = self.healpix_dir / f"correlation-{healpix_id}.fits.gz"

        # save data
        results = fitsio.FITS(filename, 'rw', clobber=True)
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
            'name': 'NSIDE',
            'value': settings.getint('nside'),
            'comment': 'Healpix nside'
        }, {
            'name': 'OMEGA_M',
            'value': global_config['cosmology'].getfloat('Omega_m'),
            'comment': 'Omega_matter(z=0) of fiducial LambdaCDM cosmology'
        }, {
            'name': "BLINDING",
            'value': self.blinding,
            'comment': 'String specifying the blinding strategy'
        }
        ]
        results.write(
            [output[2], output[3], output[4], output[5]],
            names=['R_PAR', 'R_TRANS', 'Z', 'NUM_PAIRS'],
            comment=['R-parallel', 'R-transverse', 'Redshift', 'Number of pairs'],
            units=['h^-1 Mpc', 'h^-1 Mpc', '', ''],
            header=header,
            extname='ATTRIBUTES')

        header2 = [{
            'name': 'HEALPIX_ID',
            'value': healpix_id,
            'comment': 'Healpix id'
        }]
        correlation_name = "CORRELATION"
        if self.blinding != "none":
            correlation_name += "_BLIND"

        results.write(
            [output[0], output[1], output[6]],
            names=[correlation_name, "WEIGHT_SUM", 'GAMMA_GAMMA'],
            comment=['unnormalized correlation', 'Sum of weight','gamma autocorrelation'],
            header=header2,
            extname='CORRELATION'
        )

        results.close()

    def write_dmat_healpix(self, output, healpix_id, global_config, settings):
        """Write computation output for the main healpix

        Arguments
        ---------
        config: configparser.SectionProxy
        Configuration options

        file: str
        Name of the read file, used to construct the output file
        """
        filename = self.healpix_dir / f"distortion-{healpix_id}.fits.gz"

        # save data
        results = fitsio.FITS(filename, 'rw', clobber=True)
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
            'name': 'NSIDE',
            'value': settings.getint('nside'),
            'comment': 'Healpix nside'
        }, {
            'name': 'NUM_PAIRS',
            'value': output[6],
            'comment': 'Healpix nside'
        }, {
            'name': 'PAIRS_USED',
            'value': output[7],
            'comment': 'Healpix nside'
        }, {
            'name': 'OMEGA_M',
            'value': global_config['cosmology'].getfloat('Omega_m'),
            'comment': 'Omega_matter(z=0) of fiducial LambdaCDM cosmology'
        }, {
            'name': "BLINDING",
            'value': self.blinding,
            'comment': 'String specifying the blinding strategy'
        }
        ]
        results.write(
            [output[2], output[3], output[4], output[5]],
            names=['R_PAR', 'R_TRANS', 'Z', 'EFF_WEIGHTS'],
            comment=['R-parallel', 'R-transverse', 'Redshift', 'Effective weights'],
            units=['h^-1 Mpc', 'h^-1 Mpc', '', ''],
            header=header,
            extname='ATTRIBUTES')

        header2 = [{
            'name': 'HEALPIX_ID',
            'value': healpix_id,
            'comment': 'Healpix id'
        }]

        results.write(
            [output[0], output[1]],
            names=["DISTORTION", "DISTORTION_WEIGHTS"],
            comment=['unnormalized distortion', 'distortion weights'],
            header=header2,
            extname='DISTORTION'
        )

        results.close()
