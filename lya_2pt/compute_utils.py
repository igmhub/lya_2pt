import numpy as np
from numba import njit


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
def get_num_pairs(distances1, distances2, sin_angle, cos_angle, rp_min, rp_max, rt_max, auto_flag):
    count = int(0)
    for dc1, dm1 in distances1:
        for dc2, dm2 in distances2:
            rp = (dc1 - dc2) * cos_angle
            rt = (dm1 + dm2) * sin_angle

            if auto_flag:
                rp = np.abs(rp)

            if (rp < rp_min) or (rp >= rp_max) or (rt >= rt_max):
                continue

            count += 1

    return count
