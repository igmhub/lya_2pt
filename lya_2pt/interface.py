import multiprocessing

import numpy as np
import healpy as hp
import tqdm

import lya_2pt.global_data as globals
from lya_2pt.correlation import compute_xi
from lya_2pt.distortion import compute_dmat
from lya_2pt.optimal_estimator import compute_xi_and_fisher
from lya_2pt.cosmo import Cosmology
from lya_2pt.forest_healpix_reader import ForestHealpixReader
from lya_2pt.tracer2_reader import Tracer2Reader
from lya_2pt.tracer_utils import find_healpix_neighbours
from lya_2pt.utils import find_path, parse_config, compute_ang_max
from lya_2pt.output import Output
from lya_2pt.export import Export

accepted_options = [
    "input-nside", "output-nside", "num-cpu", "z_min", "z_max", "rp_min", "rp_max", "rt_max",
    "num_bins_rp", "num_bins_rt", "num_bins_rp_model", "num_bins_rt_model",
    "rejection_fraction", "get-old-distortion"
]

defaults = {
    "input-nside": 16,
    "output-nside": 16,
    "num-cpu": 1,
    "z_min": 0,
    "z_max": 10,
    "rp_min": 0,
    "rp_max": 200,
    "rt_max": 200,
    "num_bins_rp": 50,
    "num_bins_rt": 50,
    "num_bins_rp_model": 50,
    "num_bins_rt_model": 50,
    "rejection_fraction": 0.99,
    "get-old-distortion": True
}


class Interface:
    """Interface for lya_2pt package
    Read ini files
    Handle parallezation
        - Read data
        - Call individual compute functions
    Write outputs

    Methods
    -------
    __init__
    read_tracers
    run_computation
    write_healpix_output

    Attributes
    ----------
    ang_max: float
    Maximum angle for two lines-of-sight to have neightbours

    auto_flag: bool
    True if we are working with an auto-correlation, False for cross-correlation
    Initialized to False

    nside: int
    Nside parameter to construct the healpix pixels

    z_max: float
    Maximum redshfit of the tracers

    z_min: float
    Minimum redshift of the tracers
    """
    def __init__(self, config):
        """Initialize class instance

        Arugments
        ---------
        config: configparser.ConfigParser
        Configuration options
        """
        self.config = config

        # intialize cosmology
        self.cosmo = Cosmology(config["cosmology"])

        # parse config
        self.settings = parse_config(config["settings"], defaults, accepted_options)
        globals.z_min = self.settings.getfloat("z_min")
        globals.z_max = self.settings.getfloat("z_max")
        globals.rp_min = self.settings.getfloat('rp_min')
        globals.rp_max = self.settings.getfloat("rp_max")
        globals.rt_max = self.settings.getfloat("rt_max")
        globals.num_bins_rp = self.settings.getint('num_bins_rp')
        globals.num_bins_rt = self.settings.getint('num_bins_rt')
        globals.num_bins_rp_model = self.settings.getint('num_bins_rp_model')
        globals.num_bins_rt_model = self.settings.getint('num_bins_rt_model')
        globals.rejection_fraction = self.settings.getfloat('rejection_fraction')
        globals.get_old_distortion = self.settings.getboolean('get-old-distortion')

        self.input_nside = self.settings.getint("input-nside")
        self.output_nside = self.settings.getint("output-nside")
        self.num_cpu = self.settings.getint("num-cpu")

        # TODO The default value here is z=1.7. We should adjust if we ever run at lower redshift
        self.ang_max = compute_ang_max(self.cosmo, self.settings.getfloat('rt_max'), 1.7)

        # check if we are working with an auto-correlation
        self.auto_flag = "tracer2" not in config
        globals.auto_flag = self.auto_flag
        self.need_distortion = self.config['compute'].getboolean('compute-distortion-matrix', False)

        # Find files
        input_directory = find_path(config["tracer1"].get("input-dir"))
        self.files = np.array(list(input_directory.glob('*.fits*')))

        self.output = Output(config["output"])
        self.export = Export(
            config["export"], self.output.name, self.output.output_directory, self.num_cpu)

    def read_tracers(self, files=None):
        """Read the tracers

        Arguments
        ---------
        files: array[Path], optional
        List of delta files to read, by default None
        """
        if files is None:
            files = self.files

        if self.num_cpu > 1:
            with multiprocessing.Pool(processes=self.num_cpu) as pool:
                results = list(tqdm.tqdm(pool.imap(self.read_tracer1, files), total=len(files)))
        else:
            results = [self.read_tracer1(file) for file in files]

        if self.output.blinding is None:
            self.output.blinding = results[0].blinding
        else:
            assert self.output.blinding == results[0].blinding

        forest_readers = {reader.healpix_id: reader for reader in results}
        del results

        healpix_neighbours = {
            reader.healpix_id: find_healpix_neighbours(
                reader.tracers, reader.healpix_id, self.input_nside, self.ang_max, self.auto_flag)
            for reader in forest_readers.values()
        }

        unique_healpix_neighbours = np.unique(np.hstack([
            neigh for neigh in healpix_neighbours.values()]))

        if self.auto_flag:
            auto_healpix_neighbours = unique_healpix_neighbours[
                ~np.isin(unique_healpix_neighbours, list(forest_readers.keys()))]

            tracer2_reader = Tracer2Reader(
                self.config["tracer1"], auto_healpix_neighbours, self.cosmo,
                self.num_cpu, self.need_distortion
                )
            for forest_reader in forest_readers.values():
                tracer2_reader.add_tracers(forest_reader)
        else:
            tracer2_reader = Tracer2Reader(
                self.config["tracer2"], unique_healpix_neighbours, self.cosmo,
                self.num_cpu, self.need_distortion
                )

        if self.input_nside != self.output_nside:
            globals.tracers1 = self.new_healpix_nside(
                self.output_nside,
                [forest_reader.tracers for forest_reader in forest_readers.values()]
            )
            globals.tracers2 = self.new_healpix_nside(
                self.output_nside, [tracers for tracers in tracer2_reader.tracers.values()]
            )
        else:
            globals.tracers1 = {hp_id: forest_reader.tracers
                                for hp_id, forest_reader in forest_readers.items()}
            globals.tracers2 = tracer2_reader.tracers

        globals.healpix_neighbours = {
            healpix_id: find_healpix_neighbours(
                tracers, healpix_id, self.output_nside, self.ang_max, self.auto_flag)
            for healpix_id, tracers in globals.tracers1.items()
        }

        globals.num_tracers = np.sum(
            [len(tracers)for tracers in tracer2_reader.tracers.values()])
        self.healpix_ids = np.array(list(globals.tracers1.keys()))

    def read_tracer1(self, file):
        forest_reader = ForestHealpixReader(
            self.config["tracer1"], file, self.cosmo, self.auto_flag, self.need_distortion)

        return forest_reader

    def new_healpix_nside(self, new_nside, tracers):
        new_tracers = {}

        for hp_tracers in tracers:
            phi = [tracer.ra for tracer in hp_tracers]
            theta = [np.pi / 2. - tracer.dec for tracer in hp_tracers]
            healpixs = hp.ang2pix(new_nside, theta, phi)

            for tracer, healpix_id in zip(hp_tracers, healpixs):

                if healpix_id not in new_tracers:
                    new_tracers[healpix_id] = []
                new_tracers[healpix_id].append(tracer)

        return new_tracers

    @staticmethod
    def reset_global_counter():
        globals.counter = multiprocessing.Value('i', 0)
        globals.lock = multiprocessing.Lock()

    def run(self, healpix_ids=None):
        """Run the computation

        This can include the correlation function, the distortion matrix,
        and/or the metal distortion matrix, depending on the configuration

        Arguments
        ---------
        healpix_ids: array[int], optional
        List of healpix_ids to run on, by default None
        """
        if healpix_ids is None:
            healpix_ids = self.healpix_ids
        else:
            for id in healpix_ids:
                if id not in self.healpix_ids:
                    raise ValueError(f'HEALPix ID {id} not found. '
                                     f'Currently stored IDs: {self.healpix_ids}')

        self.xi_output = {}
        if self.config['compute'].getboolean('compute-correlation', False):
            self.reset_global_counter()
            if self.num_cpu > 1:
                context = multiprocessing.get_context('fork')
                with context.Pool(processes=self.num_cpu) as pool:
                    results = pool.map(compute_xi, healpix_ids)

                for hp_id, res in results:
                    self.xi_output[hp_id] = res
            else:
                for healpix_id in healpix_ids:
                    self.xi_output[healpix_id] = compute_xi(healpix_id)[1]

        self.dmat_output = {}
        if self.config['compute'].getboolean('compute-distortion-matrix', False):
            self.reset_global_counter()
            if self.num_cpu > 1:
                context = multiprocessing.get_context('fork')
                with context.Pool(processes=self.num_cpu) as pool:
                    results = pool.map(compute_dmat, healpix_ids)

                for hp_id, res in results:
                    self.dmat_output[hp_id] = res
            else:
                for healpix_id in healpix_ids:
                    self.dmat_output[healpix_id] = compute_dmat(healpix_id)[1]

        self.optimal_xi_output = {}
        if self.config['compute'].getboolean('compute-optimal-correlation', False):
            self.reset_global_counter()

            total_size = int(globals.num_bins_rp * globals.num_bins_rt)
            xi_est = np.zeros(total_size)
            fisher_est = np.zeros((total_size, total_size))

            if self.num_cpu > 1:
                context = multiprocessing.get_context('fork')
                with context.Pool(processes=self.num_cpu) as pool:
                    results = list(tqdm.tqdm(
                        pool.imap_unordered(compute_xi_and_fisher, healpix_ids),
                        total=len(healpix_ids)
                    ))

                for hp_id, res in results:
                    self.optimal_xi_output[hp_id] = res
                    xi_est += res[0]
                    fisher_est += res[1]
            else:
                for healpix_id in healpix_ids:
                    self.optimal_xi_output[healpix_id] = compute_xi_and_fisher(healpix_id)[1]

                    xi_est += self.optimal_xi_output[healpix_id][0]
                    fisher_est += self.optimal_xi_output[healpix_id][1]

            # xiall = np.vstack([v for (v, _) in self.optimal_xi_output.values()]).sum(axis=0)

        # TODO: add other computations

    def write_results(self):
        if self.config['compute'].getboolean('compute-correlation', False):
            for healpix_id, result in self.xi_output.items():
                self.output.write_cf_healpix(result, healpix_id, self.config, self.settings)

        if self.config['compute'].getboolean('compute-distortion-matrix', False):
            for healpix_id, result in self.dmat_output.items():
                self.output.write_dmat_healpix(result, healpix_id, self.config, self.settings)

        if self.config['compute'].getboolean('compute-optimal-correlation', False):
            self.output.write_optimal_cf(self.optimal_xi_output, self.config, self.settings)

        # TODO: add other modes
