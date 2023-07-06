import numpy as np
from numba import njit

import lya_2pt.global_data as globals


@njit
def fast_dot_product(vector1, vector2):
    dot_product = 0
    for v1, v2 in zip(vector1, vector2):
        dot_product += v1 * v2
    return dot_product


@njit
def fast_outer_product(vector1, vector2):
    outer_vector = np.zeros(vector1.size * vector2.size)
    for i, v1 in enumerate(vector1):
        for j, v2 in enumerate(vector2):
            outer_vector[j + i * vector2.size] = v1 * v2
    return outer_vector


@njit
def get_bin(r, r_min, r_max, r_size):
    return np.int32(np.floor((r - r_min) / (r_max - r_min) * r_size))


@njit
def get_pixel_pairs_auto(distances1, distances2, angle):
    sin_angle = np.sin(angle / 2)
    cos_angle = np.cos(angle / 2)

    num_pairs = int(0)
    for dc1, dm1 in distances1:
        for dc2, dm2 in distances2:
            rp = np.abs((dc1 - dc2) * cos_angle)
            rt = (dm1 + dm2) * sin_angle

            if (rp < globals.rp_min) or (rp >= globals.rp_max) or (rt >= globals.rt_max):
                continue

            num_pairs += 1

    pixel_pairs = np.zeros((num_pairs, 4), dtype=np.int32)
    rp_rt_pairs = np.zeros((num_pairs, 2))

    k = np.int64(0)
    for (i, (dc1, dm1)) in enumerate(distances1):
        for (j, (dc2, dm2)) in enumerate(distances2):
            rp = np.abs((dc1 - dc2) * cos_angle)
            rt = (dm1 + dm2) * sin_angle

            if (rp < globals.rp_min) or (rp >= globals.rp_max) or (rt >= globals.rt_max):
                continue

            bin_rp_model = get_bin(rp, globals.rp_min, globals.rp_max, globals.num_bins_rp_model)
            bin_rt_model = get_bin(rt, 0., globals.rt_max, globals.num_bins_rt_model)
            bin_rp = get_bin(rp, globals.rp_min, globals.rp_max, globals.num_bins_rp)
            bin_rt = get_bin(rt, 0., globals.rt_max, globals.num_bins_rt)

            rp_rt_pairs[k] = rp, rt
            pixel_pairs[k] = (i, j, bin_rt_model + globals.num_bins_rt_model * bin_rp_model,
                              bin_rt + globals.num_bins_rt * bin_rp)
            k += 1

    return pixel_pairs, rp_rt_pairs


@njit
def get_pixel_pairs_cross(distances1, distances2, angle):
    sin_angle = np.sin(angle / 2)
    cos_angle = np.cos(angle / 2)

    num_pairs = int(0)
    for dc1, dm1 in distances1:
        for dc2, dm2 in distances2:
            rp = (dc1 - dc2) * cos_angle
            rt = (dm1 + dm2) * sin_angle

            if (rp < globals.rp_min) or (rp >= globals.rp_max) or (rt >= globals.rt_max):
                continue

            num_pairs += 1

    pixel_pairs = np.zeros((num_pairs, 4), dtype=np.int32)
    rp_rt_pairs = np.zeros((num_pairs, 2))

    k = np.int64(0)
    for (i, (dc1, dm1)) in enumerate(distances1):
        for (j, (dc2, dm2)) in enumerate(distances2):
            rp = (dc1 - dc2) * cos_angle
            rt = (dm1 + dm2) * sin_angle

            if (rp < globals.rp_min) or (rp >= globals.rp_max) or (rt >= globals.rt_max):
                continue

            bin_rp_model = get_bin(rp, globals.rp_min, globals.rp_max, globals.num_bins_rp_model)
            bin_rt_model = get_bin(rt, 0., globals.rt_max, globals.num_bins_rt_model)
            bin_rp = get_bin(rp, globals.rp_min, globals.rp_max, globals.num_bins_rp)
            bin_rt = get_bin(rt, 0., globals.rt_max, globals.num_bins_rt)

            rp_rt_pairs[k] = rp, rt
            pixel_pairs[k] = (i, j, bin_rt_model + globals.num_bins_rt_model * bin_rp_model,
                              bin_rt + globals.num_bins_rt * bin_rp)
            k += 1

    return pixel_pairs, rp_rt_pairs
