import glob
import numpy as np
from multiprocessing import Pool

from lya_2pt.utils import parse_config
from lya_2pt.errors import ReaderException
from lya_2pt import forest_healpix_reader
from lya_2pt.forest_healpix_reader import ForestHealpixReader
from lya_2pt.tracer import Tracer

# Read defaults from the healpix reader and add tracer2 specific options
accepted_options = forest_healpix_reader.accepted_options
accepted_options += []

defaults = forest_healpix_reader.defaults
defaults |= {}

class Tracer2Reader:
    """Read neighbouring healpix files and store a tracers2 list

    Methods
    -------
    __init__
    add_tracers1
    read_catalogue
    read_forests

    Attributes
    ----------
    """
    def __init__(self, config, healpix_neighbours_ids, cosmo, num_cpu):
        """Initialize class instance

        Arguments
        ---------
        config: configparser.SectionProxy
        Configuration options

        healpix_neighbours_ids: array of int
        Healpix ids to load

        cosmo: Cosmology
        Fiducial cosmology used to go from angles and redshift to distances
        """
        # parse configuration
        reader_config = parse_config(config, defaults, accepted_options)

        tracer2_type = config.get('type')
        if tracer2_type == 'continuous':
            self.read_forests(reader_config, healpix_neighbours_ids, cosmo, num_cpu)
        elif tracer2_type == 'discrete':
            self.read_catalogue(reader_config, healpix_neighbours_ids)
        else:
            raise ReaderException(
                "Unknown tracer2 type. Must be 'continuous' or 'discrete'.")

    
    def _read_one_healpix(self, config, file, cosmo, num_cpu):
        return ForestHealpixReader(config, file, cosmo, num_cpu)

    def read_forests(self, config, healpix_neighbours_ids, cosmo, num_cpu):
        """Read continuous tracers from healpix delta files

        Arguments
        ---------
        config: configparser.SectionProxy
        Configuration options

        healpix_neighbours_ids: array of int
        Healpix ids to load

        cosmo: Cosmology
        Fiducial cosmology used to go from angles and redshift to distances
        """
        in_dir = config.get('input directory')
        files = np.array(glob.glob(in_dir + '/*fits*'))

        self.tracers = np.array([], dtype=Tracer)

        neighbour_files = [in_dir + f'delta-{healpix_id}.fits.gz'
                           for healpix_id in healpix_neighbours_ids]
        neighbour_files = [file for file in neighbour_files if file in files]
            # else:
            #     # Print some error message? Or just a warning?
            #     # Ignasi: I would add a warning as this will happen occasionally
            #     # in the edges of the footprint
            #     # need to setup logging first, though
            #     pass

        # if num_cpu > 1:
        #     arguments = [(config, file, cosmo, int(1)) for file in neighbour_files]
        #     with Pool(processes=num_cpu) as pool:
        #         results = pool.starmap(self._read_one_healpix, arguments)
        # else:
        results = [ForestHealpixReader(config, file, cosmo, num_cpu)
                    for file in neighbour_files]

        for healpix_reader in results:
            self.add_tracers(healpix_reader)

    def read_catalogue(self, reader_config, healpix_neighbours_ids):
        """Read discrete tracers from catalogue

        Arguments
        ---------
        neighbour_ids: array of int
        Healpix ids to load
        """
        pass

    def add_tracers(self, healpix_reader):
        """Add the tracers in healpix_reader to the array of tracers

        Arguments
        ---------
        healpix_reader : ForestHealpixReader
        ForestHealpixReader instance with read tracers
        """
        self.tracers = np.concatenate((self.tracers, healpix_reader.tracers))
