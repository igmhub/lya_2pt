from multiprocessing import Pool

import fitsio
import numpy as np

from lya_2pt.output import get_coordinate_columns, get_coordinate_system, get_grid_header
from lya_2pt.utils import parse_config

accepted_options = ["export-correlation", "export-distortion", "smooth-covariance"]

defaults = {
    "export-correlation": False,
    "export-distortion": False,
    "smooth-covariance": True,
}


class Export:
    """Class for handling export operations
    Reads output healpix files with each correaltion/distortion matrix
    Computes the mean and covariance matrix of the samples
    Writes final correlation
    """

    def __init__(self, config, name, output_directory, num_cpu):
        self.config = parse_config(config, defaults, accepted_options)

        self.num_cpu = num_cpu
        self.name = name
        self.output_directory = output_directory
        self.healpix_dir = self.output_directory / f"healpix_files_{self.name}"
        assert self.healpix_dir.is_dir()

        self.export_correlation = self.config.getboolean("export-correlation")
        self.export_distortion = self.config.getboolean("export-distortion")

    @staticmethod
    def _coordinate_system_from_header(header):
        if "COORDSYS" not in header:
            return "RP_RT"
        if header["COORDSYS"] != "R_MU":
            raise ValueError(f"Unsupported FITS coordinate system: {header['COORDSYS']}")
        return "R_MU"

    def _set_coordinate_system(self, header):
        coordinate_system = self._coordinate_system_from_header(header)
        expected = getattr(self, "expected_coordinate_system", coordinate_system)
        if coordinate_system != expected:
            raise ValueError("Configuration and healpix FITS coordinate systems do not match")
        if hasattr(self, "coordinate_system") and self.coordinate_system != coordinate_system:
            raise ValueError("Cannot export correlation and distortion files on different grids")

        self.coordinate_system = coordinate_system
        if coordinate_system == "R_MU":
            self.coordinate1_min = header["R_MIN"]
            self.coordinate1_max = header["R_MAX"]
            self.coordinate2_min = header["MU_MIN"]
            self.coordinate2_max = header["MU_MAX"]
            self.num_bins_coordinate1 = header["NUM_BINS_R"]
            self.num_bins_coordinate2 = header["NUM_BINS_MU"]
        else:
            self.coordinate1_min = header["R_PAR_MIN"]
            self.coordinate1_max = header["R_PAR_MAX"]
            self.coordinate2_min = 0.0
            self.coordinate2_max = header["R_TRANS_MAX"]
            self.num_bins_coordinate1 = header["NUM_BINS_R_PAR"]
            self.num_bins_coordinate2 = header["NUM_BINS_R_TRANS"]

    def run(self, global_config, settings):
        self.expected_coordinate_system = get_coordinate_system(settings)
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
        files = np.array(list(self.healpix_dir.glob("correlation*fits*")))

        with fitsio.FITS(files[0]) as hdul:
            header = hdul[1].read_header()
            self._set_coordinate_system(header)

        self.delta_coordinate1 = (
            self.coordinate1_max - self.coordinate1_min
        ) / self.num_bins_coordinate1
        self.delta_coordinate2 = (
            self.coordinate2_max - self.coordinate2_min
        ) / self.num_bins_coordinate2

        if self.num_cpu > 1:
            with Pool(processes=self.num_cpu) as pool:
                results = pool.map(self._read_correlation, files)
        else:
            results = [self._read_correlation(file) for file in files]

        results = np.array(results)
        self.correlations = results[:, 0, :]
        self.weights = results[:, 1, :]
        self.mean_correlation = np.sum(self.correlations * self.weights, axis=0)
        self.coordinate1 = np.sum(results[:, 2, :] * self.weights, axis=0)
        self.coordinate2 = np.sum(results[:, 3, :] * self.weights, axis=0)
        self.z_grid = np.sum(results[:, 4, :] * self.weights, axis=0)
        self.num_pairs = np.sum(results[:, 5, :], axis=0)

        self.sum_weights = np.sum(self.weights, axis=0)
        w = self.sum_weights > 0
        self.mean_correlation[w] /= self.sum_weights[w]
        self.coordinate1[w] /= self.sum_weights[w]
        self.coordinate2[w] /= self.sum_weights[w]
        self.z_grid[w] /= self.sum_weights[w]

    def _read_correlation(self, file):
        with fitsio.FITS(file) as hdul:
            coordinate_system = self._coordinate_system_from_header(hdul[1].read_header())
            if coordinate_system != self.coordinate_system:
                raise ValueError("Cannot export mixed coordinate-system healpix files")
            coordinate_names, _, _ = get_coordinate_columns(coordinate_system)
            coordinate1 = hdul[1][coordinate_names[0]][:]
            coordinate2 = hdul[1][coordinate_names[1]][:]
            z = hdul[1]["Z"][:]
            num_pairs = hdul[1]["NUM_PAIRS"][:]

            # TODO implement blinding support
            correlation = hdul[2]["CORRELATION"][:]
            weights = hdul[2]["WEIGHT_SUM"][:]

        return correlation, weights, coordinate1, coordinate2, z, num_pairs

    def read_distortion(self):
        files = np.array(list(self.healpix_dir.glob("distortion*fits*")))

        with fitsio.FITS(files[0]) as hdul:
            header = hdul[1].read_header()
            self._set_coordinate_system(header)

        if self.num_cpu > 1:
            with Pool(processes=self.num_cpu) as pool:
                results = pool.map(self._read_distortion, files)
        else:
            results = [self._read_distortion(file) for file in files]

        results = list(results)
        self.distortion = np.array([item[0] for item in results]).sum(axis=0)
        self.dist_weights = np.array([item[1] for item in results]).sum(axis=0)
        self.dist_coordinate1 = np.array([item[2] for item in results]).sum(axis=0)
        self.dist_coordinate2 = np.array([item[3] for item in results]).sum(axis=0)
        self.dist_z = np.array([item[4] for item in results]).sum(axis=0)
        self.dist_eff_weights = np.array([item[5] for item in results]).sum(axis=0)
        self.dist_num_pairs = np.array([item[6] for item in results]).sum(axis=0)
        self.dist_num_pairs_used = np.array([item[7] for item in results]).sum(axis=0)

        w = self.dist_weights > 0
        self.distortion[w] /= self.dist_weights[w, None]

        w = self.dist_eff_weights > 0
        self.dist_coordinate1[w] /= self.dist_eff_weights[w]
        self.dist_coordinate2[w] /= self.dist_eff_weights[w]
        self.dist_z[w] /= self.dist_eff_weights[w]

    def _read_distortion(self, file):
        with fitsio.FITS(file) as hdul:
            header = hdul[1].read_header()
            coordinate_system = self._coordinate_system_from_header(header)
            if coordinate_system != self.coordinate_system:
                raise ValueError("Cannot export mixed coordinate-system healpix files")
            num_pairs = header["NUM_PAIRS"]
            num_pairs_used = header["PAIRS_USED"]

            coordinate_names, _, _ = get_coordinate_columns(coordinate_system)
            coordinate1 = hdul[1][coordinate_names[0]][:]
            coordinate2 = hdul[1][coordinate_names[1]][:]
            z = hdul[1]["Z"][:]
            eff_weights = hdul[1]["EFF_WEIGHTS"][:]

            # TODO implement blinding support
            distortion = hdul[2]["DISTORTION"][:]
            weights = hdul[2]["DISTORTION_WEIGHTS"][:]

        return (
            distortion,
            weights,
            coordinate1,
            coordinate2,
            z,
            eff_weights,
            num_pairs,
            num_pairs_used,
        )

    def compute_covariance(self):
        meanless_xi_times_weights = self.weights * (self.correlations - self.mean_correlation)

        covariance = meanless_xi_times_weights.T.dot(meanless_xi_times_weights)
        sum_weights_squared = np.outer(self.sum_weights, self.sum_weights)
        w = sum_weights_squared > 0.0
        covariance[w] /= sum_weights_squared[w]

        if self.config.getboolean("smooth-covariance"):
            print("Smoothing covariance matrix")
            covariance = self.smooth_covariance(covariance)

        self.covariance = covariance

    def smooth_covariance(self, covariance):
        num_bins = covariance.shape[1]
        var = np.diagonal(covariance)
        if np.any(var == 0):
            raise ValueError("Covariance has at least one 0 on the diagonal. Cannot smooth.")
        elif np.any(var < 0):
            raise ValueError(
                "Covariance has at least one negative value on the diagonal. Cannot smooth."
            )

        correlation = covariance / np.outer(np.sqrt(var), np.sqrt(var))
        correlation_smooth = np.zeros([num_bins, num_bins])

        # add together the correlation from bins with similar separations in
        # parallel and perpendicular distances
        sum_correlation = {}
        counts_correlation = {}
        for i in range(num_bins):
            print("\rSmoothing {}".format(i + 1), end="")
            for j in range(i + 1, num_bins):
                ind_coordinate1 = round(
                    abs(self.coordinate1[j] - self.coordinate1[i]) / self.delta_coordinate1
                )
                ind_coordinate2 = round(
                    abs(self.coordinate2[i] - self.coordinate2[j]) / self.delta_coordinate2
                )
                if (ind_coordinate1, ind_coordinate2) not in sum_correlation:
                    sum_correlation[(ind_coordinate1, ind_coordinate2)] = 0
                    counts_correlation[(ind_coordinate1, ind_coordinate2)] = 0

                sum_correlation[(ind_coordinate1, ind_coordinate2)] += correlation[i, j]
                counts_correlation[(ind_coordinate1, ind_coordinate2)] += 1

        for i in range(num_bins):
            correlation_smooth[i, i] = 1.0
            for j in range(i + 1, num_bins):
                ind_coordinate1 = round(
                    abs(self.coordinate1[j] - self.coordinate1[i]) / self.delta_coordinate1
                )
                ind_coordinate2 = round(
                    abs(self.coordinate2[i] - self.coordinate2[j]) / self.delta_coordinate2
                )
                correlation_smooth[i, j] = (
                    sum_correlation[(ind_coordinate1, ind_coordinate2)]
                    / counts_correlation[(ind_coordinate1, ind_coordinate2)]
                )
                correlation_smooth[j, i] = correlation_smooth[i, j]

        print("\n")
        covariance_smooth = correlation_smooth * np.outer(np.sqrt(var), np.sqrt(var))
        return covariance_smooth

    def write_correlation(self, global_config, settings):
        output_file = self.output_directory / f"{self.name}-exp.fits.gz"
        results = fitsio.FITS(output_file, "rw", clobber=True)

        # distortion = self.distortion
        distortion = None
        if distortion is None:
            distortion = np.eye(len(self.covariance))

        coordinate_system = get_coordinate_system(settings)
        header = get_grid_header(settings) + [
            {
                "name": "Z_MIN",
                "value": settings.getfloat("z_min"),
                "comment": "Minimum redshift of pairs",
            },
            {
                "name": "Z_MAX",
                "value": settings.getfloat("z_max"),
                "comment": "Maximum redshift of pairs",
            },
            {
                "name": "OMEGA_M",
                "value": global_config["cosmology"].getfloat("Omega_m"),
                "comment": "Omega_matter(z=0) of fiducial LambdaCDM cosmology",
            },
            {
                "name": "BLINDING",
                "value": "placeholder",  # TODO Correct this once blinding implemented
                "comment": "String specifying the blinding strategy",
            },
        ]

        if coordinate_system == "R_MU":
            coordinate_names = ["R", "MU"]
            model_coordinate_names = ["DMR", "DMMU"]
            coordinate_comments = ["Separation", "Cosine to line of sight"]
            model_coordinate_comments = ["Separation model", "Cosine to line of sight model"]
        else:
            coordinate_names = ["RP", "RT"]
            model_coordinate_names = ["DMRP", "DMRT"]
            coordinate_comments = ["R-parallel", "R-transverse"]
            model_coordinate_comments = ["R-parallel model", "R-transverse model"]

        comment = [
            *coordinate_comments,
            "Redshift",
            "Correlation",
            "Covariance matrix",
            "Distortion matrix",
            "Number of pairs",
        ]
        results.write(
            [
                self.coordinate1,
                self.coordinate2,
                self.z_grid,
                self.mean_correlation,
                self.covariance,
                distortion,
                self.num_pairs,
            ],
            names=[*coordinate_names, "Z", "DA", "CO", "DM", "NB"],
            comment=comment,
            header=header,
            extname="COR",
        )

        comment = [*model_coordinate_comments, "Redshift model"]
        results.write(
            [self.coordinate1, self.coordinate2, self.z_grid],
            names=[*model_coordinate_names, "DMZ"],
            comment=comment,
            extname="DMATRIX",
        )
        results.close()

    def write_distortion(self, global_config, settings):
        output_file = self.output_directory / f"dmat_{self.name}-exp.fits.gz"
        results = fitsio.FITS(output_file, "rw", clobber=True)

        coordinate_system = get_coordinate_system(settings)
        header = get_grid_header(settings) + [
            {
                "name": "Z_MIN",
                "value": settings.getfloat("z_min"),
                "comment": "Minimum redshift of pairs",
            },
            {
                "name": "Z_MAX",
                "value": settings.getfloat("z_max"),
                "comment": "Maximum redshift of pairs",
            },
            {
                "name": "REJECTION_FRAC",
                "value": settings.getfloat("rejection_fraction"),
                "comment": "Rejection fraction when computing distortion",
            },
            {"name": "NUM_PAIRS", "value": self.dist_num_pairs, "comment": "Healpix nside"},
            {"name": "PAIRS_USED", "value": self.dist_num_pairs_used, "comment": "Healpix nside"},
            {
                "name": "OMEGA_M",
                "value": global_config["cosmology"].getfloat("Omega_m"),
                "comment": "Omega_matter(z=0) of fiducial LambdaCDM cosmology",
            },
            {
                "name": "BLINDING",
                "value": "placeholder",  # TODO Correct this once blinding implemented
                "comment": "String specifying the blinding strategy",
            },
        ]

        results.write(
            [self.dist_weights, self.distortion],
            names=["WDM", "DM"],
            comment=["Sum of weights", "Distortion matrix"],
            header=header,
            extname="COR",
        )

        if coordinate_system == "R_MU":
            coordinate_names = ["R", "MU"]
            coordinate_comments = ["Separation", "Cosine to line of sight"]
            coordinate_units = ["h^-1 Mpc", ""]
        else:
            coordinate_names = ["RP", "RT"]
            coordinate_comments = ["R-parallel", "R-transverse"]
            coordinate_units = ["h^-1 Mpc", "h^-1 Mpc"]

        results.write(
            [self.dist_coordinate1, self.dist_coordinate2, self.dist_z],
            names=[*coordinate_names, "Z"],
            comment=[*coordinate_comments, "Redshift"],
            units=[*coordinate_units, ""],
            extname="DMATRIX",
        )
        results.close()
