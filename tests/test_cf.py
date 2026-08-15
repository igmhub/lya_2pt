from configparser import ConfigParser
from pathlib import Path

import fitsio
import numpy as np
import pytest

import lya_2pt.global_data as global_data
from lya_2pt import Interface

TESTS_DIR = Path(__file__).parent


def test_cf(tmp_path):
    config = ConfigParser()
    config.read(TESTS_DIR / "configs" / "lyaxlya_cf.ini")
    output_dir = tmp_path / "products"
    config["tracer1"]["input-dir"] = str((TESTS_DIR / "deltas").resolve())
    config["output"]["output-dir"] = str(output_dir)

    print("Initializing")
    lya2pt = Interface(config)
    assert global_data.rmu_binning is False
    assert isinstance(global_data.rp_min, float)
    assert isinstance(global_data.num_bins_rp, int)

    lya2pt.read_tracers()
    lya2pt.run()
    lya2pt.write_results()
    lya2pt.export.run(lya2pt.config, lya2pt.settings)

    cf_file = lya2pt.export.output_directory / f"{lya2pt.export.name}-exp.fits.gz"
    hdul_cf = fitsio.FITS(cf_file)
    cf_file_test = TESTS_DIR / "output" / f"{lya2pt.export.name}-exp.fits.gz"
    hdul_cf_test = fitsio.FITS(cf_file_test)

    assert np.allclose(hdul_cf[1]["DA"][:], hdul_cf_test[1]["DA"][:])
    assert np.allclose(hdul_cf[1]["CO"][:], hdul_cf_test[1]["CO"][:])
    assert np.allclose(hdul_cf[1]["RP"][:], hdul_cf_test[1]["RP"][:])
    assert np.allclose(hdul_cf[1]["RT"][:], hdul_cf_test[1]["RT"][:])
    assert np.allclose(hdul_cf[1]["Z"][:], hdul_cf_test[1]["Z"][:])
    assert np.allclose(hdul_cf[1]["NB"][:], hdul_cf_test[1]["NB"][:])

    dmat_file = lya2pt.export.output_directory / f"dmat_{lya2pt.export.name}-exp.fits.gz"
    hdul_dmat = fitsio.FITS(dmat_file)
    dmat_file_test = TESTS_DIR / "output" / f"dmat_{lya2pt.export.name}-exp.fits.gz"
    hdul_dmat_test = fitsio.FITS(dmat_file_test)
    assert np.allclose(hdul_dmat[1]["DM"][:], hdul_dmat_test[1]["DM"][:])
    assert np.allclose(hdul_dmat[1]["WDM"][:], hdul_dmat_test[1]["WDM"][:])
    assert np.allclose(hdul_dmat[2]["RP"][:], hdul_dmat_test[2]["RP"][:])
    assert np.allclose(hdul_dmat[2]["RT"][:], hdul_dmat_test[2]["RT"][:])
    assert np.allclose(hdul_dmat[2]["Z"][:], hdul_dmat_test[2]["Z"][:])


@pytest.mark.parametrize("get_old_distortion", [True, False])
def test_rmu_cf_and_distortion(tmp_path, get_old_distortion):
    config = ConfigParser()
    config.read(TESTS_DIR / "configs" / "lyaxlya_rmu_cf.ini")
    output_dir = tmp_path / "products"
    config["tracer1"]["input-dir"] = str((TESTS_DIR / "deltas").resolve())
    config["output"]["output-dir"] = str(output_dir)
    config["settings"]["get-old-distortion"] = str(get_old_distortion)
    config["tracer1"]["use-old-projection"] = str(get_old_distortion)

    lya2pt = Interface(config)
    assert global_data.rmu_binning is True
    assert isinstance(global_data.r_min, float)
    assert isinstance(global_data.mu_max, float)
    assert isinstance(global_data.num_bins_r_model, int)
    lya2pt.read_tracers()
    lya2pt.run()
    lya2pt.write_results()
    lya2pt.export.run(lya2pt.config, lya2pt.settings)
    lya2pt.export.run(lya2pt.config, lya2pt.settings)

    cf_file = lya2pt.export.output_directory / f"{lya2pt.export.name}-exp.fits.gz"
    with fitsio.FITS(cf_file) as hdul_cf:
        header = hdul_cf[1].read_header()
        assert header["COORDSYS"] == "R_MU"
        assert header["NUM_BINS_R_MODEL"] == 4
        assert header["NUM_BINS_MU_MODEL"] == 4
        assert {"R", "MU"}.issubset(hdul_cf[1].get_colnames())
        assert {"DMR", "DMMU"}.issubset(hdul_cf[2].get_colnames())
        populated = hdul_cf[1]["NB"][:] > 0
        assert np.all(np.isfinite(hdul_cf[1]["DA"][:]))
        assert np.all((hdul_cf[1]["MU"][:][populated] >= 0) & (hdul_cf[1]["MU"][:][populated] < 1))

    dmat_file = lya2pt.export.output_directory / f"dmat_{lya2pt.export.name}-exp.fits.gz"
    with fitsio.FITS(dmat_file) as hdul_dmat:
        assert hdul_dmat[1].read_header()["COORDSYS"] == "R_MU"
        assert hdul_dmat[1]["DM"][:].shape == (16, 16)
        assert {"R", "MU"}.issubset(hdul_dmat[2].get_colnames())
        assert np.all(np.isfinite(hdul_dmat[1]["DM"][:]))
