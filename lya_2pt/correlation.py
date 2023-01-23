import numpy as np
from numba import jit, int32
from lya_2pt.utils import get_angle


def compute_xi(tracers1, tracers2, config):
    rp_min = config.getfloat('rp_min')
    rp_max = config.getfloat('rp_max')
    rt_max = config.getfloat('rt_max')
    num_bins_rp = config.getfloat('num_bins_rp')
    num_bins_rt = config.getfloat('num_bins_rt')
    total_size = num_bins_rp * num_bins_rt

    xi_grid = np.zeros(total_size)
    weights_grid = np.zeros(total_size)
    rp_grid = np.zeros(total_size)
    rt_grid = np.zeros(total_size)
    z_grid = np.zeros(total_size)
    num_pairs_grid = np.zeros(total_size, dtype=np.int64)

    for tracer1 in tracers1:
        for tracer2 in tracers2[tracer1.neighbours]:
            angle = get_angle(tracer1, tracer2)

            compute_xi_pair_vectorized(tracer1.deltas, tracer1.weights, tracer1.z,
                tracer1.comoving_distance, tracer1.comoving_transverse_distance,
                tracer2.deltas, tracer2.weights, tracer2.z,
                tracer2.comoving_distance, tracer2.comoving_transverse_distance,
                angle, rp_min, rp_max, rt_max, num_bins_rp, num_bins_rt,
                xi_grid, weights_grid, rp_grid, rt_grid, z_grid, num_pairs_grid)

    w = weights_grid > 0
    xi_grid[w] /= weights_grid[w]
    rp_grid[w] /= weights_grid[w]
    rt_grid[w] /= weights_grid[w]
    z_grid[w] /= weights_grid[w]

    return xi_grid, weights_grid, rp_grid, rt_grid, z_grid, num_pairs_grid


def compute_xi_pair_vectorized(deltas1, weights1, z1, dc1, dm1, deltas2, weights2, z2, dc2, dm2,
                               angle, rp_min, rp_max, rt_max, rp_size, rt_size,
                               xi_grid, weights_grid, rp_grid, rt_grid, z_grid, num_pairs_grid):

    # Abs is optional (only for auto)
    rp = np.abs(np.subtract.outer(dc1, dc2) * np.cos(angle/2))
    rt = np.add.outer(dm1, dm2) * np.sin(angle/2)
    z = np.add.outer(z1, z2) / 2

    xi = np.outer(deltas1 * weights1, deltas2 * weights2)
    xi_weights = np.outer(weights1, weights2)

    mask = (rp >= rp_min) & (rp < rp_max) & (rt < rt_max)

    rp = rp[mask]
    rt = rt[mask]
    z = z[mask]
    xi = xi[mask]
    xi_weights = xi_weights[mask]

    bins_rp = np.floor((rp - rp_min) / (rp_max - rp_min) * rp_size).astype(int)
    bins_rt = np.floor(rt / rt_max * rt_size).astype(int)
    bins = (bins_rt + rt_size * bins_rp).astype(int).flatten()

    rebin_xi = np.bincount(bins, weights=xi.flatten())
    rebin_weight = np.bincount(bins, weights=xi_weights.flatten())
    rebin_rp = np.bincount(bins, weights=(rp * xi_weights).flatten())
    rebin_rt = np.bincount(bins, weights=(rt * xi_weights).flatten())
    rebin_z = np.bincount(bins, weights=(z * xi_weights).flatten())
    rebin_num_pairs = np.bincount(bins, weights=(xi_weights > 0.).flatten())

    rebin_size = len(rebin_xi)
    total_size = rp_size * rt_size

    xi_grid[:rebin_size] += rebin_xi[:total_size]
    weights_grid[:rebin_size] += rebin_weight[:total_size]
    rp_grid[:rebin_size] += rebin_rp[:total_size]
    rt_grid[:rebin_size] += rebin_rt[:total_size]
    z_grid[:rebin_size] += rebin_z[:total_size]
    num_pairs_grid[:rebin_size] += rebin_num_pairs[:total_size]
