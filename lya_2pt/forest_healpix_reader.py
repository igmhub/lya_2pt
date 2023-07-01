import fitsio
import numpy as np
from healpy import query_disc

from lya_2pt.constants import ACCEPTED_BLINDING_STRATEGIES
from lya_2pt.errors import ReaderException
from lya_2pt.utils import parse_config
from lya_2pt.read_io import read_from_image, read_from_hdu

accepted_options = [
    "input-dir", "tracer-type", "absorption-line", "project-deltas",
    "projection-order", "use-old-projection", "rebin",
    "redshift-evolution", "reference-redshift"
]

defaults = {
    "tracer-type": "continuous",
    "absorption-line": "LYA",
    "project-deltas": True,
    "projection-order": 1,
    "use-old-projection": True,
    "rebin": 1,
    "redshift-evolution": 2.9,
    "reference-redshift": 2.25,
}


class ForestHealpixReader:
    """Class to read the data of a forest-like tracer

    This class will automatically deduce the data format and call the
    relevant methods.
    Two data formats are accepted (from picca.delta_extraction):
    - an HDU per forest
    - image table
    Read data will be formatted as a list of tracers

    Methods
    -------
    __init__
    find_healpix_neighbours
    find_neighbours

    Attributes
    ----------
    auto_flag: bool
    True if we are working with an auto-correlation, False for cross-correlation

    blinding: str
    A string specifying the chosen blinding strategies. Must be one of the
    accepted values in ACCEPTED_BLINDING_STRATEGIES

    tracers: array of Tracer
    The set of tracers for this healpix
    """
    def __init__(self, config, file, cosmo, auto_flag=False, need_distortion=False):
        """Initialize class instance

        Arguments
        ---------
        config: configparser.SectionProxy
        Configuration options

        file: Path
        Path of the file to read

        cosmo: Cosmology
        Fiducial cosmology used to go from angles and redshift to distances

        Raise
        -----
        ReaderException if the tracer type is not continuous
        ReaderException if the blinding strategy is not valid
        """
        # parse configuration
        reader_config = parse_config(config, defaults, accepted_options)
        self.healpix_id = int(file.name.split("delta-")[-1].split(".fits")[0])

        # extract parameters from config
        absorption_line = reader_config.get("absorption-line")
        tracer1_type = config.get('tracer-type')
        if tracer1_type != 'continuous':
            raise ReaderException(
                f"Tracer type must be 'continuous'. Found: '{tracer1_type}'")

        self.auto_flag = auto_flag

        # read data
        self.tracers = None
        hdul = fitsio.FITS(file)
        # image format
        if "METADATA" in hdul:
            self.tracers, self.wave_solution, self.dwave = read_from_image(
                hdul, absorption_line, self.healpix_id, need_distortion,
                reader_config.getint("projection-order"))
            self.blinding = hdul["METADATA"].read_header()["BLINDING"]
        # HDU per forest
        else:
            self.tracers, self.wave_solution, self.dwave = read_from_hdu(
                hdul, absorption_line, self.healpix_id, need_distortion,
                reader_config.getint("projection-order"))
            self.blinding = hdul[1].read_header()["BLINDING"]

        if self.blinding not in ACCEPTED_BLINDING_STRATEGIES:
            raise ReaderException(
                "Expected blinding strategy fo be one of: " +
                " ".join(ACCEPTED_BLINDING_STRATEGIES) +
                f" Found: {self.blinding}"
            )

        # Apply redshift evolution correction
        reference_z = reader_config.getfloat("reference-redshift")
        redshift_evol = reader_config.getfloat("redshift-evolution")
        for tracer in self.tracers:
            tracer.apply_z_evol_to_weights(redshift_evol, reference_z)

        # rebin
        rebin_factor = reader_config.getint("rebin")
        if rebin_factor > 1:
            for tracer in self.tracers:
                tracer.rebin(rebin_factor, self.dwave, absorption_line)

        # project
        if reader_config.getboolean("project-deltas"):
            for tracer in self.tracers:
                tracer.project(reader_config.getboolean("use-old-projection"))

        # add distances
        for tracer in self.tracers:
            tracer.compute_comoving_distances(cosmo)

    def find_healpix_neighbours(self, nside, ang_max):
        """Find the healpix neighbours

        Arguments
        ---------
        nside: int
        Nside parameter to construct the healpix pixels

        ang_max: float
        Maximum angle for two lines-of-sight to have neightbours

        Return
        ------
        healpix_ids: array of int
        The healpix id of the neighbouring healpixes

        Raise
        -----
        ReaderException if the self.tracers is None
        """
        if self.tracers is None:
            raise ReaderException(
                "In ForestHealpixReader, self.tracer should not be None")

        neighbour_ids = set()
        for tracer in self.tracers:
            tracer_neighbour_ids = query_disc(nside, [tracer.x_cart, tracer.y_cart, tracer.z_cart],
                                              ang_max, inclusive=True)
            neighbour_ids = neighbour_ids.union(set(tracer_neighbour_ids))

        neighbour_ids = np.array(list(neighbour_ids))
        if self.healpix_id in neighbour_ids and self.auto_flag:
            neighbour_ids = np.delete(neighbour_ids, np.where(neighbour_ids == self.healpix_id))

        return neighbour_ids

    def find_neighbours(self, other, z_min, z_max, rp_max, rt_max):
        """For each tracer, find neighbouring tracers. Keep the results in
        tracer.neighbours

        Arguments
        ---------
        other: Tracer2Reader
        Other tracers

        z_min: float
        Minimum redshift of the tracers

        z_max: float
        Maximum redshfit of the tracers

        Raise
        -----
        ReaderException if the self.tracers is None
        ReaderException if the other.tracers is None
        """
        if self.tracers is None:
            raise ReaderException(
                "In ForestHealpixReader, self.tracer should not be None")
        if other.tracers is None:
            raise ReaderException(
                "In ForestHealpixReader, other.tracer should not be None")

        for tracer1 in self.tracers:
            neighbour_mask = np.full(other.tracers.shape, False)

            for index2, tracer2 in enumerate(other.tracers):
                if tracer1.is_neighbour(tracer2, self.auto_flag, z_min, z_max, rp_max, rt_max):
                    neighbour_mask[index2] = True

            tracer1.add_neighbours(neighbour_mask)
