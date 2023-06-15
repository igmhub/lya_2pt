import numpy as np
from numba import njit

from lya_2pt.tracer_utils import get_angle


def compute_xi(tracers1, tracers2, config, auto_flag=False):
    rp_min = config.getfloat('rp_min')
    rp_max = config.getfloat('rp_max')
    rt_max = config.getfloat('rt_max')
    num_bins_rp = config.getint('num_bins_rp')
    num_bins_rt = config.getint('num_bins_rt')
    total_size = int(num_bins_rp * num_bins_rt)

    xi_grid = np.zeros(total_size)
    weights_grid = np.zeros(total_size)
    rp_grid = np.zeros(total_size)
    rt_grid = np.zeros(total_size)
    z_grid = np.zeros(total_size)
    num_pairs_grid = np.zeros(total_size, dtype=np.int32)

    for tracer1 in tracers1:
        assert tracer1.neighbours is not None
        for tracer2 in tracers2[tracer1.neighbours]:
            angle = get_angle(
                tracer1.x_cart, tracer1.y_cart, tracer1.z_cart, tracer1.ra, tracer1.dec,
                tracer2.x_cart, tracer2.y_cart, tracer2.z_cart, tracer2.ra, tracer2.dec
                )

            compute_xi_pair(
                tracer1.deltas, tracer1.weights, tracer1.z,
                tracer1.comoving_distance, tracer1.comoving_transverse_distance,
                tracer2.deltas, tracer2.weights, tracer2.z,
                tracer2.comoving_distance, tracer2.comoving_transverse_distance,
                angle, auto_flag, rp_min, rp_max, rt_max, num_bins_rp, num_bins_rt,
                xi_grid, weights_grid, rp_grid, rt_grid, z_grid, num_pairs_grid
                )

    w = weights_grid > 0
    xi_grid[w] /= weights_grid[w]
    rp_grid[w] /= weights_grid[w]
    rt_grid[w] /= weights_grid[w]
    z_grid[w] /= weights_grid[w]

    return xi_grid, weights_grid, rp_grid, rt_grid, z_grid, num_pairs_grid


@njit
def compute_xi_pair(
        deltas1, weights1, z1, dc1, dm1,
        deltas2, weights2, z2, dc2, dm2, angle,
        auto_flag, rp_min, rp_max, rt_max, rp_size, rt_size,
        xi_grid, weights_grid, rp_grid, rt_grid, z_grid, num_pairs_grid):

    sin_angle = np.sin(angle / 2)
    cos_angle = np.cos(angle / 2)

    for i in range(deltas1.size):
        if weights1[i] == 0:
            continue
        for j in range(deltas2.size):
            if weights2[j] == 0:
                continue

            # Abs is optional (only for auto)
            rp = (dc1[i] - dc2[j]) * cos_angle
            rt = (dm1[i] + dm2[j]) * sin_angle
            if auto_flag:
                rp = np.abs(rp)

            if (rp < rp_min) or (rp >= rp_max) or (rt >= rt_max):
                continue

            bins_rp = np.floor((rp - rp_min) / (rp_max - rp_min) * rp_size)
            bins_rt = np.floor(rt / rt_max * rt_size)
            bins = int(bins_rt + rt_size * bins_rp)

            weight12 = weights1[i] * weights2[j]
            xi_grid[bins] += deltas1[i] * deltas2[j] * weight12
            weights_grid[bins] += weight12
            rp_grid[bins] += rp * weight12
            rt_grid[bins] += rt * weight12
            z_grid[bins] += (z1[i] + z2[j]) / 2 * weight12
            num_pairs_grid[bins] += 1
