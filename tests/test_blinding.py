from configparser import ConfigParser
from pathlib import Path

import fitsio
import numpy as np
import pytest

from lya_2pt import Interface
from lya_2pt.errors import ReaderException
from lya_2pt.export import UNBLINDABLE_STRATEGIES, Export

TESTS_DIR = Path(__file__).parent


def _make_export(tmp_path, blind_corr_type=""):
    output_directory = tmp_path / "products"
    output_directory.mkdir(exist_ok=True)
    (output_directory / "healpix_files_test").mkdir(exist_ok=True)

    config = ConfigParser()
    config["export"] = {
        "export-correlation": "True",
        "export-distortion": "False",
        "smooth-covariance": "False",
        "blind-corr-type": blind_corr_type,
    }
    return Export(config["export"], "test", output_directory, 1)


def _copy_blinded_deltas(tmp_path, blindings, use_delta_blind=False):
    input_directory = tmp_path / "deltas"
    input_directory.mkdir()
    source_files = sorted((TESTS_DIR / "deltas").glob("*.fits.gz"))

    for source, blinding in zip(source_files, blindings):
        target = input_directory / source.name.removesuffix(".gz")
        with fitsio.FITS(source) as source_hdul, fitsio.FITS(target, "rw", clobber=True) as hdul:
            for extension_index in range(1, len(source_hdul)):
                source_hdu = source_hdul[extension_index]
                header = source_hdu.read_header()
                if extension_index == 1:
                    header["BLINDING"] = blinding
                columns = source_hdu.get_colnames()
                if blinding != "none" or use_delta_blind:
                    output_columns = [
                        "DELTA_BLIND" if column == "DELTA" else column for column in columns
                    ]
                else:
                    output_columns = columns
                hdul.write(
                    [source_hdu[column][:] for column in columns],
                    names=output_columns,
                    header=header,
                    extname=source_hdu.get_extname(),
                )

    return input_directory


def _make_interface(tmp_path, input_directory, blind_corr_type="lyaxlya"):
    config = ConfigParser()
    config.read(TESTS_DIR / "configs" / "lyaxlya_cf.ini")
    config["tracer1"]["input-dir"] = str(input_directory)
    config["output"]["output-dir"] = str(tmp_path / "products")
    if blind_corr_type:
        config["export"]["blind-corr-type"] = blind_corr_type
    return Interface(config)


def _write_image_delta(tmp_path, blinding, delta_name):
    source = TESTS_DIR / "deltas" / "delta-1952.fits.gz"
    target = tmp_path / "delta-1952.fits"

    with fitsio.FITS(source) as source_hdul, fitsio.FITS(target, "rw", clobber=True) as hdul:
        source_hdu = source_hdul[1]
        header = source_hdu.read_header()
        metadata_names = ["LOS_ID", "RA", "DEC", "Z"]
        hdul.write(
            [np.array([header[name]]) for name in metadata_names],
            names=metadata_names,
            header=[{"name": "BLINDING", "value": blinding}],
            extname="METADATA",
        )
        hdul.write(
            source_hdu["LAMBDA"][:],
            header=[{"name": "DELTA_LAMBDA", "value": header["DELTA_LAMBDA"]}],
            extname="LAMBDA",
        )
        hdul.write(np.array([source_hdu["DELTA"][:]]), extname=delta_name)
        hdul.write(np.array([source_hdu["WEIGHT"][:]]), extname="WEIGHT")

    return target


def test_blinding_configuration_and_export_validation(tmp_path):
    with pytest.raises(ValueError, match="blind-corr-type"):
        _make_export(tmp_path, "invalid")

    export = _make_export(tmp_path, "lyaxlya")
    export.coordinate_system = "RP_RT"
    export._set_blinding(["desi_dr3", "desi_dr3"])
    assert export.blinding == "desi_dr3"
    assert export._output_is_blinded()

    with pytest.raises(ValueError, match="different blinding"):
        export._set_blinding(["none", "desi_dr3"])

    with pytest.raises(ValueError, match="Expected blinding strategy"):
        export._set_blinding(["not-a-strategy"])

    export.coordinate_system = "R_MU"
    with pytest.raises(ValueError, match="rp-rt"):
        export._set_blinding(["desi_dr3"])


@pytest.mark.parametrize("blinding", UNBLINDABLE_STRATEGIES)
def test_rmu_allows_unblindable_strategies(tmp_path, blinding):
    export = _make_export(tmp_path, "lyaxlya")
    export.coordinate_system = "R_MU"

    export._set_blinding([blinding])

    assert export.blinding == blinding
    assert not export._output_is_blinded()


def test_reader_rejects_mixed_blinding_strategies(tmp_path):
    input_directory = _copy_blinded_deltas(tmp_path, ["none", "desi_dr3"])
    lya2pt = _make_interface(tmp_path, input_directory)

    with pytest.raises(ValueError, match="same blinding strategy"):
        lya2pt.read_tracers()


def test_reader_rejects_delta_blind_with_no_blinding(tmp_path):
    input_directory = _copy_blinded_deltas(tmp_path, ["none", "none"], use_delta_blind=True)
    lya2pt = _make_interface(tmp_path, input_directory)

    with pytest.raises(ReaderException, match="DELTA_BLIND.*BLINDING='none'"):
        lya2pt.read_tracers()


def test_image_reader_uses_delta_blind(tmp_path):
    delta_file = _write_image_delta(tmp_path, "desi_y1", "DELTA_BLIND")
    lya2pt = _make_interface(tmp_path, tmp_path)

    reader = lya2pt.read_tracer1(delta_file)

    assert reader.blinding == "desi_y1"
    assert len(reader.tracers) == 1


def test_image_reader_rejects_delta_blind_with_no_blinding(tmp_path):
    delta_file = _write_image_delta(tmp_path, "none", "DELTA_BLIND")
    lya2pt = _make_interface(tmp_path, tmp_path)

    with pytest.raises(ReaderException, match="DELTA_BLIND.*BLINDING='none'"):
        lya2pt.read_tracer1(delta_file)
