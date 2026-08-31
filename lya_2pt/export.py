import os.path
from multiprocessing import Pool

import fitsio
import h5py
import numpy as np
import scipy.interpolate

from lya_2pt.constants import ACCEPTED_BLIND_CORRELATION_TYPES, ACCEPTED_BLINDING_STRATEGIES
from lya_2pt.output import get_coordinate_columns, get_coordinate_system, get_grid_header
from lya_2pt.utils import parse_config

UNBLINDABLE_STRATEGIES = ["none", "desi_m2", "desi_y1", "desi_y3"]

accepted_options = [
    "export-correlation",
    "export-distortion",
    "smooth-covariance",
    "blind-corr-type",
]

defaults = {
    "export-correlation": False,
    "export-distortion": False,
    "smooth-covariance": True,
    "blind-corr-type": "",
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
        self.blind_corr_type = self.config.get("blind-corr-type")
        if self.blind_corr_type and self.blind_corr_type not in ACCEPTED_BLIND_CORRELATION_TYPES:
            raise ValueError(
                "Expected blind-corr-type to be one of "
                f"{ACCEPTED_BLIND_CORRELATION_TYPES}. Found {self.blind_corr_type}."
            )

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

    def _set_blinding(self, blindings):
        """Validate and store the common blinding strategy from HEALPix files."""
        unique_blindings = set(blindings)
        if len(unique_blindings) != 1:
            raise ValueError("Cannot export HEALPix files with different blinding strategies")

        blinding = unique_blindings.pop()
        if isinstance(blinding, np.number):
            if not blinding.is_integer() or not 0 <= blinding < len(ACCEPTED_BLINDING_STRATEGIES):
                raise ValueError(
                    "Expected blinding strategy to be one of "
                    f"{ACCEPTED_BLINDING_STRATEGIES}. Found {blinding}."
                )
            blinding = ACCEPTED_BLINDING_STRATEGIES[int(blinding)]
        if blinding not in ACCEPTED_BLINDING_STRATEGIES:
            raise ValueError(
                "Expected blinding strategy to be one of "
                f"{ACCEPTED_BLINDING_STRATEGIES}. Found {blinding}."
            )
        if hasattr(self, "blinding") and self.blinding != blinding:
            raise ValueError(
                "Cannot export correlation and distortion files with different blinding"
            )
        if self.coordinate_system == "R_MU" and blinding not in UNBLINDABLE_STRATEGIES:
            raise ValueError("Blinded export is only supported for rp-rt coordinate grids")

        self.blinding = blinding

    def _output_is_blinded(self):
        return self.blinding not in UNBLINDABLE_STRATEGIES

    def _apply_blinding(self, xi):
        """Apply the DESI blinding template to an aggregated correlation."""
        blinding = self.blinding
        if blinding in UNBLINDABLE_STRATEGIES:
            print(f"'{blinding}' correlations are not blinded.")
            return xi

        blinding_dir = "/global/cfs/projectdirs/desi/science/lya/lya_blinding/bao/"
        blinding_templates = {
            "desi_dr3": {
                "standard": "dr3_blinding_v4_standard_28_05_2026.h5",
                "grid": "dr3_blinding_v4_regular_grid_28_05_2026.h5",
            }
        }

        if blinding in blinding_templates:
            print(f"Blinding using seed for {blinding}")
        else:
            raise ValueError(
                f"Expected blinding to be one of {blinding_templates.keys()}. Found {blinding}."
            )

        if not self.blind_corr_type:
            raise ValueError("Blinding requires export option blind-corr-type.")

        blind_corr_type = self.blind_corr_type
        # Match the name expected in the blinding template file.
        if blind_corr_type == "qsoxlya":
            blind_corr_type = "lyaxqso"
        if blind_corr_type == "qsoxlyb":
            blind_corr_type = "lybxqso"

        # Check type of correlation and get size and regular binning.
        if blind_corr_type in ["lyaxlya", "lyaxlyb"]:
            corr_size = 2500
            rp_interp_grid = np.arange(2.0, 202.0, 4)
            rt_interp_grid = np.arange(2.0, 202.0, 4)
        elif blind_corr_type in ["lyaxqso", "lybxqso"]:
            corr_size = 5000
            rp_interp_grid = np.arange(-197.99, 202.01, 4)
            rt_interp_grid = np.arange(2.0, 202, 4)
        else:
            raise ValueError("Unknown correlation type: {}".format(blind_corr_type))

        if corr_size == self.num_bins_coordinate1 * self.num_bins_coordinate2:
            # Read the blinding file and get the right template.
            blinding_filename = blinding_dir + blinding_templates[blinding]["standard"]
        else:
            # Read the regular-grid blinding file and get the right template.
            blinding_filename = blinding_dir + blinding_templates[blinding]["grid"]

        if not os.path.isfile(blinding_filename):
            raise RuntimeError(
                "Missing blinding file. Make sure you are running at"
                " NERSC or contact picca developers"
            )
        with h5py.File(blinding_filename, "r") as blinding_file:
            hex_diff = np.array(blinding_file["blinding"][blind_corr_type]).astype(str)
        diff_grid = np.array([float.fromhex(x) for x in hex_diff])

        if corr_size == self.num_bins_coordinate1 * self.num_bins_coordinate2:
            diff = diff_grid
        else:
            # Interpolate the blinding template on the regular grid.
            interp = scipy.interpolate.RectBivariateSpline(
                rp_interp_grid,
                rt_interp_grid,
                diff_grid.reshape(len(rp_interp_grid), len(rt_interp_grid)),
                kx=3,
                ky=3,
            )
            diff = interp.ev(self.coordinate1, self.coordinate2)

        # Check that the shapes match.
        if np.shape(xi) != np.shape(diff):
            raise RuntimeError(
                "Unknown binning or wrong correlation type. Cannot blind."
                " Please raise an issue or contact picca developers."
            )

        # Add blinding.
        return xi + diff

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
        self._set_blinding(results[:, 6, :].ravel())
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
            header = hdul[1].read_header()
            coordinate_system = self._coordinate_system_from_header(header)
            if coordinate_system != self.coordinate_system:
                raise ValueError("Cannot export mixed coordinate-system healpix files")
            coordinate_names, _, _ = get_coordinate_columns(coordinate_system)
            coordinate1 = hdul[1][coordinate_names[0]][:]
            coordinate2 = hdul[1][coordinate_names[1]][:]
            z = hdul[1]["Z"][:]
            num_pairs = hdul[1]["NUM_PAIRS"][:]

            blinding = header["BLINDING"] if "BLINDING" in header else "none"
            if blinding not in ACCEPTED_BLINDING_STRATEGIES:
                raise ValueError(
                    "Expected blinding strategy to be one of "
                    f"{ACCEPTED_BLINDING_STRATEGIES}. Found {blinding}."
                )
            correlation_name = "CORRELATION"
            if "CORRELATION_BLIND" in hdul[2].get_colnames():
                correlation_name += "_BLIND"
            correlation = hdul[2][correlation_name][:]
            weights = hdul[2]["WEIGHT_SUM"][:]

        blinding_code = ACCEPTED_BLINDING_STRATEGIES.index(blinding)
        blinding_codes = np.full(correlation.shape, blinding_code)
        return correlation, weights, coordinate1, coordinate2, z, num_pairs, blinding_codes

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
        self._set_blinding([result[8] for result in results])
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

            blinding = header["BLINDING"] if "BLINDING" in header else "none"
            distortion_name = "DISTORTION"
            if "DISTORTION_BLIND" in hdul[2].get_colnames():
                distortion_name += "_BLIND"
            distortion = hdul[2][distortion_name][:]
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
            blinding,
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
        xi = self._apply_blinding(self.mean_correlation.copy())
        correlation_name = "DA_BLIND" if self._output_is_blinded() else "DA"
        distortion_name = "DM_EMPTY" if self._output_is_blinded() else "DM"

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
                "value": self.blinding,
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
                xi,
                self.covariance,
                distortion,
                self.num_pairs,
            ],
            names=[*coordinate_names, "Z", correlation_name, "CO", distortion_name, "NB"],
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
        distortion_name = "DM_BLIND" if self._output_is_blinded() else "DM"
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
                "value": self.blinding,
                "comment": "String specifying the blinding strategy",
            },
        ]

        results.write(
            [self.dist_weights, self.distortion],
            names=["WDM", distortion_name],
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
