"""Unit tests for configuration and numerical utility helpers."""

from configparser import ConfigParser

import numpy as np
import pytest

from lya_2pt.compute_utils import fast_dot_product, fast_outer_product, get_bin
from lya_2pt.errors import ParserError
from lya_2pt.utils import check_dir, compute_ang_max, parse_config


class _Cosmology:
    def get_dist_m(self, redshift):
        return 100.0 * redshift


def test_parse_config_adds_defaults():
    config = ConfigParser()
    config["settings"] = {"required": "value"}

    parsed = parse_config(config["settings"], {"optional": 3}, ["required", "optional"])

    assert parsed.get("optional") == "3"


def test_parse_config_rejects_missing_and_unknown_options():
    config = ConfigParser()
    config["settings"] = {}
    with pytest.raises(ParserError, match="Missing option required"):
        parse_config(config["settings"], {}, ["required"])

    config["settings"] = {"required": "value", "unknown": "value"}
    with pytest.raises(ParserError, match="Unrecognised option"):
        parse_config(config["settings"], {}, ["required"])


def test_compute_ang_max_and_check_dir(tmp_path):
    cosmo = _Cosmology()
    assert compute_ang_max(cosmo, rt_max=20.0, z_min=1.0) == pytest.approx(2 * np.arcsin(0.1))
    assert compute_ang_max(cosmo, rt_max=200.0, z_min=1.0) == np.pi

    output_dir = tmp_path / "new-output"
    check_dir(output_dir)
    assert output_dir.is_dir()


def test_numba_helpers():
    vector1 = np.array([1.0, 2.0])
    vector2 = np.array([3.0, 4.0])

    assert fast_dot_product(vector1, vector2) == 11.0
    np.testing.assert_array_equal(fast_outer_product(vector1, vector2), [3.0, 4.0, 6.0, 8.0])
    assert get_bin(5.0, 0.0, 10.0, 10) == 5
