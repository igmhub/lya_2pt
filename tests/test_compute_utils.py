import numpy as np
import pytest

import lya_2pt.global_data as global_data
from lya_2pt.compute_utils import get_pixel_pairs_rmu_auto, get_pixel_pairs_rmu_cross
from lya_2pt.export import Export


def configure_rmu_globals():
    global_data.r_min = 10.0
    global_data.r_max = 200.0
    global_data.mu_min = -1.0
    global_data.mu_max = 1.0
    global_data.num_bins_r = 4
    global_data.num_bins_mu = 4
    global_data.num_bins_r_model = 4
    global_data.num_bins_mu_model = 4


def test_rmu_pair_binning_and_boundaries():
    configure_rmu_globals()
    zero_distance = np.array([[0.0, 1.0]])
    boundary_distance = np.array([[200.0, 1.0]])

    for distances1, distances2 in (
        (zero_distance, zero_distance),
        (boundary_distance, zero_distance),
    ):
        pixel_pairs, coordinate_pairs = get_pixel_pairs_rmu_cross(distances1, distances2, 0.0)
        assert pixel_pairs.size == 0
        assert coordinate_pairs.size == 0


def test_rmu_cross_pairs_keep_signed_mu():
    configure_rmu_globals()
    distances1 = np.array([[100.0, 3.0]])
    distances2 = np.array([[150.0, 3.0]])

    pixel_pairs, coordinate_pairs = get_pixel_pairs_rmu_cross(distances1, distances2, 0.0)

    assert pixel_pairs.tolist() == [[0, 0, 0, 0]]
    assert np.allclose(coordinate_pairs, [[50.0, -1.0]])


def test_rmu_auto_pairs_use_r_and_mu_bins():
    configure_rmu_globals()
    distances1 = np.array([[30.0, 10.0]])
    distances2 = np.array([[90.0, 10.0]])

    pixel_pairs, coordinate_pairs = get_pixel_pairs_rmu_auto(distances1, distances2, np.pi / 2)

    assert pixel_pairs.tolist() == [[0, 0, 3, 3]]
    assert np.allclose(coordinate_pairs, [[np.sqrt(2000), 3 / np.sqrt(10)]])


def test_export_rejects_coordinate_system_mismatch():
    export = Export.__new__(Export)
    export.expected_coordinate_system = "RP_RT"

    with pytest.raises(ValueError, match="do not match"):
        export._set_coordinate_system({"COORDSYS": "R_MU"})
