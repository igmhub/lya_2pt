import glob
import numpy as np

from lya_2pt.tracer import Tracer
from lya_2pt.forest_healpix_reader import ForestHealpixReader

class Tracer2Reader:
    """Read neighbouring healpix files and store a tracers2 list
    """
    def __init__(self, config, neighbour_ids, cosmo):
        self.config = config

        tracer2_type = config['data'].get('tracer2_type', 'continuous')
        self.auto_flag = self.config['data'].getboolean('auto_correlation', True)

        if tracer2_type == 'continuous':
            self.read_forests(neighbour_ids, cosmo)
        elif tracer2_type == 'discrete':
            self.read_catalogue(neighbour_ids)
        else:
            raise ValueError("Unknown tracer2 type. Must be 'continuous' or 'discrete'.")

    def read_forests(self, neighbour_ids, cosmo):
        """Read continuous tracers from healpix delta files

        Parameters
        ----------
        neighbour_ids : ArrayLike
            List of Healpix IDs for neighbouring pixels
        """
        in_dir = None
        if self.auto_flag:
            if 'input_directory2' in self.config['data']:
                # Print some error message? Or just a warning?
                pass
            
            in_dir = self.config['data'].get('input_directory')
        else:
            in_dir = self.config['data'].get('input_directory2')

        files = glob.glob(in_dir)

        self.tracers = np.array([], dtype=Tracer)
        for healpix_id in neighbour_ids:
            file = in_dir + f'delta-{healpix_id}.fits.gz'
            if file in files:
                healpix_reader = ForestHealpixReader(self.config, file, cosmo)
                self.tracers = np.concatenate(self.tracers, healpix_reader.tracers)
            else:
                # Print some error message? Or just a warning?
                pass
        
    def read_catalogue(self, neighbour_ids):
        """Read discrete tracers from catalogue

        Parameters
        ----------
        neighbour_ids : ArrayLike
            List of Healpix IDs for neighbouring pixels
        """
        pass

    def add_tracers1(self, tracers1):
        """Add tracers1 array to tracers2.
        Only used when computing an auto-correlation

        Parameters
        ----------
        tracers1 : ArrayLike
            Array containing tracers1 read by the original ForestHealpixReader
        """
        assert tracers1.dtype == Tracer
        self.tracers = np.concatenate(tracers1, self.tracers)
