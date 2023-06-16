from multiprocessing import Pool

import numpy as np

from lya_2pt.correlation import compute_xi
from lya_2pt.distortion import compute_dmat
from lya_2pt.cosmo import Cosmology, PiccaCosmo
from lya_2pt.forest_healpix_reader import ForestHealpixReader
from lya_2pt.tracer2_reader import Tracer2Reader
from lya_2pt.utils import find_path, parse_config, compute_ang_max
from lya_2pt.output import Output
from lya_2pt.export import Export

accepted_options = [
    "num_bins_rp", "num_bins_rt", "num_bins_rp_model", "num_bins_rt_model",
    "nside", "rp_min", "rp_max", "rt_max", "z_min", "z_max", "num processors",
    "rejection_fraction"
]

defaults = {
    "nside": 16,
    "z_min": 0,
    "z_max": 10,
    "rp_min": 0,
    "rp_max": 200,
    "rt_max": 200,
    "num_bins_rp": 50,
    "num_bins_rt": 50,
    "num_bins_rp_model": 50,
    "num_bins_rt_model": 50,
    "num processors": 1,
    "rejection_fraction": 0.99
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
        use_picca_cosmo = config["cosmology"].getboolean('use_picca_cosmo', False)
        if use_picca_cosmo:
            self.cosmo = PiccaCosmo(config["cosmology"])
        else:
            self.cosmo = Cosmology(config["cosmology"])

        # parse config
        self.settings = parse_config(config["settings"], defaults, accepted_options)
        self.z_min = self.settings.getfloat("z_min")
        self.z_max = self.settings.getfloat("z_max")
        self.rp_max = self.settings.getfloat("rp_max")
        self.rt_max = self.settings.getfloat("rt_max")
        self.nside = self.settings.getint("nside")
        self.num_cpu = self.settings.getint("num processors")
        self.ang_max = compute_ang_max(self.cosmo, self.settings.getfloat('rt_max'), 1.7)

        # check if we are working with an auto-correlation
        self.auto_flag = "tracer2" not in config

        # Find files
        input_directory = find_path(config["tracer1"].get("input directory"))
        self.files = np.array(list(input_directory.glob('*fits*')))

        self.output = Output(config["output"])
        self.export = Export(config["export"], self.output.name, self.output.output_directory,
                             self.num_cpu)

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
                results = pool.map(self.read_tracer1, files)
        else:
            results = [self.read_tracer1(file) for file in files]

        forest_readers = {}
        healpix_neighbours = []
        for res in results:
            hp_id = res[0].healpix_id
            forest_readers[hp_id] = res[0]
            healpix_neighbours.append(res[1])

            if self.output.blinding is None:
                self.output.blinding = res[0].blinding
            else:
                assert self.output.blinding == res[0].blinding

        healpix_neighbours = np.unique(np.hstack(healpix_neighbours))

        if self.auto_flag:
            healpix_neighbours = healpix_neighbours[~np.isin(healpix_neighbours,
                                                             list(forest_readers.keys()))]
            self.tracer2_reader = Tracer2Reader(self.config["tracer1"], healpix_neighbours,
                                                self.cosmo, self.num_cpu)
            for forest_reader in forest_readers.values():
                self.tracer2_reader.add_tracers(forest_reader)
        else:
            self.tracer2_reader = Tracer2Reader(self.config["tracer2"], healpix_neighbours,
                                                self.cosmo, self.num_cpu)

        if len(files) > 1 and self.num_cpu > 1:
            with Pool(processes=self.num_cpu) as pool:
                results = pool.map(self.find_neighbours, forest_readers.values())

            for res in results:
                forest_readers[res.healpix_id] = res
        else:
            for hp_id, forest_reader in forest_readers.items():
                forest_readers[hp_id] = self.find_neighbours(forest_reader)

        self.tracers1 = {hp_id: forest_reader.tracers
                         for hp_id, forest_reader in forest_readers.items()}
        self.tracers2 = self.tracer2_reader.tracers

    def read_tracer1(self, file):
        forest_reader = ForestHealpixReader(self.config["tracer1"], file, self.cosmo,
                                            self.auto_flag)
        healpix_neighbours = forest_reader.find_healpix_neighbours(self.nside, self.ang_max)

        return forest_reader, healpix_neighbours

    def find_neighbours(self, forest_reader):
        forest_reader.find_neighbours(self.tracer2_reader, self.z_min, self.z_max,
                                      self.rp_max, self.rt_max)
        return forest_reader

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
        if self.config['compute'].getboolean('compute-correlation'):
            if self.num_cpu > 1:
                arguments = [(tracers1, self.tracers2, self.settings, self.auto_flag)
                             for tracers1 in self.tracers1.values()]

                with Pool(processes=self.num_cpu) as pool:
                    results = pool.starmap(compute_xi, arguments)

                for hp_id, res in zip(self.tracers1.keys(), results):
                    self.xi_output[hp_id] = res
            else:
                for hp_id, tracers1 in self.tracers1.items():
                    self.xi_output[hp_id] = compute_xi(tracers1, self.tracers2, self.settings,
                                                       self.auto_flag)

        self.dmat_output = {}
        if self.config['compute'].getboolean('compute-distortion-matrix'):
            if self.num_cpu > 1:
                arguments = [(tracers1, self.tracers2, self.settings)
                             for tracers1 in self.tracers1.values()]

                with Pool(processes=self.num_cpu) as pool:
                    results = pool.starmap(compute_dmat, arguments)

                for hp_id, res in zip(self.tracers1.keys(), results):
                    self.dmat_output[hp_id] = res
            else:
                for hp_id, tracers1 in self.tracers1.items():
                    self.dmat_output[hp_id] = compute_dmat(tracers1, self.tracers2, self.settings,
                                                           self.auto_flag)

        # TODO: add other computations

    def write_results(self):
        if self.config['compute'].getboolean('compute-correlation'):
            for healpix_id, result in self.xi_output.items():
                self.output.write_cf_healpix(result, healpix_id, self.config, self.settings)

        if self.config['compute'].getboolean('compute-distortion-matrix'):
            for healpix_id, result in self.dmat_output.items():
                self.output.write_dmat_healpix(result, healpix_id, self.config, self.settings)

        # TODO: add other modes
