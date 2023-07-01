import numpy as np
from numba import njit


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
def compute_rp(dc1, dc2, i, j, cos_angle, auto_flag):
    if auto_flag:
        return np.abs((dc1[i] - dc2[j]) * cos_angle)

    return (dc1[i] - dc2[j]) * cos_angle


@njit
def compute_rt(dm1, dm2, i, j, sin_angle):
    return (dm1[i] + dm2[j]) * sin_angle


@njit
def get_num_pairs_auto(distances1, distances2, sin_angle, cos_angle, rp_min, rp_max, rt_max):
    count = int(0)
    for dc1, dm1 in distances1:
        for dc2, dm2 in distances2:
            rp = np.abs((dc1 - dc2) * cos_angle)
            rt = (dm1 + dm2) * sin_angle

            if (rp < rp_min) or (rp >= rp_max) or (rt >= rt_max):
                continue

            count += 1

    return count


@njit
def get_num_pairs_cross(distances1, distances2, sin_angle, cos_angle, rp_min, rp_max, rt_max):
    count = int(0)
    for dc1, dm1 in distances1:
        for dc2, dm2 in distances2:
            rp = (dc1 - dc2) * cos_angle
            rt = (dm1 + dm2) * sin_angle

            if (rp < rp_min) or (rp >= rp_max) or (rt >= rt_max):
                continue

            count += 1

    return count


@njit
def get_pixel_pairs_auto(
    distances1, distances2, angle, rp_min, rp_max, rt_max,
    rp_size, rt_size, rp_size_model, rt_size_model
):
    sin_angle = np.sin(angle / 2)
    cos_angle = np.cos(angle / 2)

    num_pairs = get_num_pairs_auto(
        distances1, distances2, sin_angle, cos_angle, rp_min, rp_max, rt_max)

    pixel_pairs = np.zeros((num_pairs, 4), dtype=np.int32)
    rp_rt_pairs = np.zeros((num_pairs, 2))

    k = np.int64(0)
    for (i, (dc1, dm1)) in enumerate(distances1):
        for (j, (dc2, dm2)) in enumerate(distances2):
            rp = np.abs((dc1 - dc2) * cos_angle)
            rt = (dm1 + dm2) * sin_angle

            if (rp < rp_min) or (rp >= rp_max) or (rt >= rt_max):
                continue

            bin_rp_model = get_bin(rp, rp_min, rp_max, rp_size_model)
            bin_rt_model = get_bin(rt, 0., rt_max, rt_size_model)
            bin_rp = get_bin(rp, rp_min, rp_max, rp_size)
            bin_rt = get_bin(rt, 0., rt_max, rt_size)

            rp_rt_pairs[k] = rp, rt
            pixel_pairs[k] = (i, j, bin_rt_model + rt_size_model * bin_rp_model,
                              bin_rt + rt_size * bin_rp)
            k += 1

    return pixel_pairs, rp_rt_pairs


@njit
def get_pixel_pairs_cross(
    distances1, distances2, angle, rp_min, rp_max, rt_max,
    rp_size, rt_size, rp_size_model, rt_size_model
):
    sin_angle = np.sin(angle / 2)
    cos_angle = np.cos(angle / 2)

    num_pairs = get_num_pairs_cross(
        distances1, distances2, sin_angle, cos_angle, rp_min, rp_max, rt_max)

    pixel_pairs = np.zeros((num_pairs, 4), dtype=np.int32)
    rp_rt_pairs = np.zeros((num_pairs, 2))

    k = np.int64(0)
    for (i, (dc1, dm1)) in enumerate(distances1):
        for (j, (dc2, dm2)) in enumerate(distances2):
            rp = (dc1 - dc2) * cos_angle
            rt = (dm1 + dm2) * sin_angle

            if (rp < rp_min) or (rp >= rp_max) or (rt >= rt_max):
                continue

            bin_rp_model = get_bin(rp, rp_min, rp_max, rp_size_model)
            bin_rt_model = get_bin(rt, 0., rt_max, rt_size_model)
            bin_rp = get_bin(rp, rp_min, rp_max, rp_size)
            bin_rt = get_bin(rt, 0., rt_max, rt_size)

            rp_rt_pairs[k] = rp, rt
            pixel_pairs[k] = (i, j, bin_rt_model + rt_size_model * bin_rp_model,
                              bin_rt + rt_size * bin_rp)
            k += 1

    return pixel_pairs, rp_rt_pairs
