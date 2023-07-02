from multiprocessing import Pool

import numpy as np
import tqdm

from lya_2pt.correlation import compute_xi
from lya_2pt.distortion import compute_dmat
from lya_2pt.cosmo import Cosmology
from lya_2pt.forest_healpix_reader import ForestHealpixReader
from lya_2pt.tracer2_reader import Tracer2Reader
from lya_2pt.utils import find_path, parse_config, compute_ang_max
from lya_2pt.output import Output
from lya_2pt.export import Export

accepted_options = [
    "nside", "num-cpu", "z_min", "z_max", "rp_min", "rp_max", "rt_max",
    "num_bins_rp", "num_bins_rt", "num_bins_rp_model", "num_bins_rt_model",
    "rejection_fraction", "get-old-distortion"
]

defaults = {
    "nside": 16,
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
        self.z_min = self.settings.getfloat("z_min")
        self.z_max = self.settings.getfloat("z_max")
        self.rp_max = self.settings.getfloat("rp_max")
        self.rt_max = self.settings.getfloat("rt_max")
        self.nside = self.settings.getint("nside")
        self.num_cpu = self.settings.getint("num-cpu")

        # TODO The default value here is z=1.7. We should adjust if we ever run at lower redshift
        self.ang_max = compute_ang_max(self.cosmo, self.settings.getfloat('rt_max'), 1.7)

        # check if we are working with an auto-correlation
        self.auto_flag = "tracer2" not in config
        self.need_distortion = self.config['compute'].getboolean('compute-distortion-matrix', False)

        # Find files
        input_directory = find_path(config["tracer1"].get("input-dir"))
        self.files = np.array(list(input_directory.glob('*fits*')))

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
            with Pool(processes=self.num_cpu) as pool:
                results = list(tqdm.tqdm(pool.imap(self.read_tracer1, files), total=len(files)))
        else:
            results = [self.read_tracer1(file) for file in files]

        if self.output.blinding is None:
            self.output.blinding = results[0].blinding
        else:
            assert self.output.blinding == results[0].blinding

        forest_readers = {reader.healpix_id: reader for reader in results}
        del results

        self.healpix_neighbours = {}
        for reader in forest_readers.values():
           self.healpix_neighbours[reader.healpix_id] = reader.find_healpix_neighbours(
               self.nside, self.ang_max)


        unique_healpix_neighbours = np.unique(np.hstack([
            neigh for neigh in self.healpix_neighbours.values()]))

        if self.auto_flag:
            healpix_neighbours = unique_healpix_neighbours[~np.isin(unique_healpix_neighbours,
                                                             list(forest_readers.keys()))]
            self.tracer2_reader = Tracer2Reader(
                self.config["tracer1"], healpix_neighbours, self.cosmo,
                self.num_cpu, self.need_distortion
                )
            for forest_reader in forest_readers.values():
                self.tracer2_reader.add_tracers(forest_reader)
        else:
            self.tracer2_reader = Tracer2Reader(
                self.config["tracer2"], unique_healpix_neighbours, self.cosmo,
                self.num_cpu, self.need_distortion
                )

        self.tracers1 = {hp_id: forest_reader.tracers
                         for hp_id, forest_reader in forest_readers.items()}
        self.tracers2 = self.tracer2_reader.tracers
        self.healpix_ids = np.array(list(self.tracers1.keys()))

        if len(files) > 1 and self.num_cpu > 1:
            with Pool(processes=self.num_cpu) as pool:
                _ = tqdm.tqdm(pool.imap(self.find_neighbours, self.healpix_ids),
                              total=len(self.healpix_ids))
        else:
            for healpix_id in self.healpix_ids:
                self.find_neighbours(healpix_id)

    def read_tracer1(self, file):
        forest_reader = ForestHealpixReader(
            self.config["tracer1"], file, self.cosmo, self.auto_flag, self.need_distortion)

        return forest_reader

    def find_neighbours(self, healpix_id):
        # forest_reader.find_neighbours(
        #     self.tracer2_reader, self.z_min, self.z_max, self.rp_max, self.rt_max)
        # return forest_reader
        hp_neighs = [other_hp for other_hp in self.healpix_neighbours[healpix_id]
                     if other_hp in self.tracers2]
        hp_neighs += [healpix_id]

        for tracer1 in self.tracers1[healpix_id]:
            neighbours = [tracer2 for hp in hp_neighs for tracer2 in self.tracers2[hp]]

            if self.auto_flag:
                neighbours = [tracer2 for tracer2 in neighbours if tracer1.ra > tracer2.ra]

            tracer1.add_neighbours(
                neighbours, self.auto_flag, self.z_min, self.z_max, self.rp_max, self.rt_max)

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
            healpix_ids = list(self.tracers1.keys())

        self.xi_output = {}
        if self.config['compute'].getboolean('compute-correlation', False):
            if self.num_cpu > 1:
                arguments = [(tracers1, self.tracers2, self.settings, self.auto_flag)
                             for tracers1 in self.tracers1.values()]

                with Pool(processes=self.num_cpu) as pool:
                    results = pool.starmap(compute_xi,
                                           tqdm.tqdm(arguments, total=len(self.tracers1)))

                for hp_id, res in zip(self.tracers1.keys(), results):
                    self.xi_output[hp_id] = res
            else:
                for hp_id, tracers1 in self.tracers1.items():
                    self.xi_output[hp_id] = compute_xi(
                        tracers1, self.tracers2, self.settings, self.auto_flag)

        self.dmat_output = {}
        if self.config['compute'].getboolean('compute-distortion-matrix', False):
            if self.num_cpu > 1:
                arguments = [(tracers1, self.tracers2, self.settings)
                             for tracers1 in self.tracers1.values()]

                with Pool(processes=self.num_cpu) as pool:
                    results = pool.starmap(compute_dmat,
                                           tqdm.tqdm(arguments, total=len(self.tracers1)))

                for hp_id, res in zip(self.tracers1.keys(), results):
                    self.dmat_output[hp_id] = res
            else:
                for hp_id, tracers1 in self.tracers1.items():
                    self.dmat_output[hp_id] = compute_dmat(
                        tracers1, self.tracers2, self.settings, self.auto_flag)

        # TODO: add other computations

    def write_results(self):
        if self.config['compute'].getboolean('compute-correlation', False):
            for healpix_id, result in self.xi_output.items():
                self.output.write_cf_healpix(result, healpix_id, self.config, self.settings)

        if self.config['compute'].getboolean('compute-distortion-matrix', False):
            for healpix_id, result in self.dmat_output.items():
                self.output.write_dmat_healpix(result, healpix_id, self.config, self.settings)

        # TODO: add other modes
