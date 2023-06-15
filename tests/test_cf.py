import fitsio
import numpy as np
from math import isclose
from configparser import ConfigParser

from lya_2pt import Interface
from lya_2pt.utils import find_path


def test_cf():
    config = ConfigParser()
    config.read(find_path("configs/lyaxlya_cf.ini"))

    print('Initializing')
    lya2pt = Interface(config)
    lya2pt.read_tracers()
    lya2pt.run()
    lya2pt.write_results()
    lya2pt.export.run(lya2pt.config, lya2pt.settings)

    cf_file = lya2pt.export.output_directory / f'{lya2pt.export.name}-exp.fits.gz'
    hdul_cf = fitsio.FITS(cf_file)
    assert isclose(np.sum(hdul_cf[1]['DA'][:]), 0.0024777050405812573)
    assert isclose(np.sum(hdul_cf[1]['CO'][:]), 0.00030179352264525305)
    assert isclose(np.sum(hdul_cf[1]['RP'][:]), 165028.4959600392)
    assert isclose(np.sum(hdul_cf[1]['RT'][:]), 108809.82787843319)
    assert isclose(np.sum(hdul_cf[1]['Z'][:]), 3980.341712981344)
    assert isclose(np.sum(hdul_cf[1]['NB'][:]), 15856904.0)

    dmat_file = lya2pt.export.output_directory / f'dmat_{lya2pt.export.name}-exp.fits.gz'
    hdul_dmat = fitsio.FITS(dmat_file)
    assert isclose(np.sum(hdul_dmat[1]['DM'][:]), 145.19076424842885)
    assert isclose(np.sum(hdul_dmat[1]['WDM'][:]), 6632386505.4460745)
    assert isclose(np.sum(hdul_dmat[2]['RP'][:]), 165028.4959600392)
    assert isclose(np.sum(hdul_dmat[2]['RT'][:]), 108809.82787843319)
    assert isclose(np.sum(hdul_dmat[2]['Z'][:]), 3980.341712981344)
