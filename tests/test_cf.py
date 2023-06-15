import fitsio
import numpy as np
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
    cf_file_test = find_path('output/' + f'{lya2pt.export.name}-exp.fits.gz')
    hdul_cf_test = fitsio.FITS(cf_file_test)

    assert np.allclose(hdul_cf[1]['DA'][:], hdul_cf_test[1]['DA'][:])
    assert np.allclose(hdul_cf[1]['CO'][:], hdul_cf_test[1]['CO'][:])
    assert np.allclose(hdul_cf[1]['RP'][:], hdul_cf_test[1]['RP'][:])
    assert np.allclose(hdul_cf[1]['RT'][:], hdul_cf_test[1]['RT'][:])
    assert np.allclose(hdul_cf[1]['Z'][:], hdul_cf_test[1]['Z'][:])
    assert np.allclose(hdul_cf[1]['NB'][:], hdul_cf_test[1]['NB'][:])

    dmat_file = lya2pt.export.output_directory / f'dmat_{lya2pt.export.name}-exp.fits.gz'
    hdul_dmat = fitsio.FITS(dmat_file)
    dmat_file_test = find_path('output/' + f'dmat_{lya2pt.export.name}-exp.fits.gz')
    hdul_dmat_test = fitsio.FITS(dmat_file_test)
    assert np.allclose(hdul_dmat[1]['DM'][:], hdul_dmat_test[1]['DM'][:])
    assert np.allclose(hdul_dmat[1]['WDM'][:], hdul_dmat_test[1]['WDM'][:])
    assert np.allclose(hdul_dmat[2]['RP'][:], hdul_dmat_test[2]['RP'][:])
    assert np.allclose(hdul_dmat[2]['RT'][:], hdul_dmat_test[2]['RT'][:])
    assert np.allclose(hdul_dmat[2]['Z'][:], hdul_dmat_test[2]['Z'][:])
