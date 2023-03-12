import numpy as np
from numba import njit, int32
from multiprocessing import Pool
from lya_2pt.utils import get_angle


def _compute_dmat_kernel(tracers1, tracers2, config):
    rejection_fraction = config.getfloat('rejection_fraction')
    rp_min = config.getfloat('rp_min')
    rp_max = config.getfloat('rp_max')
    rt_max = config.getfloat('rt_max')
    num_bins_rp = config.getint('num_bins_rp')
    num_bins_rt = config.getint('num_bins_rt')
    num_bins_rp_model = config.getint('num_bins_rp_model')
    num_bins_rt_model = config.getint('num_bins_rt_model')
    total_size = int(num_bins_rp * num_bins_rt)
    total_size_model = int(num_bins_rp_model * num_bins_rt_model)

    distortion = np.zeros((total_size, total_size_model))
    weights_dmat = np.zeros(total_size)
    rp_grid = np.zeros(total_size_model)
    rt_grid = np.zeros(total_size_model)
    z_grid = np.zeros(total_size_model)
    weights_grid = np.zeros(total_size_model)

    num_pairs = 0
    num_pairs_used = 0
    for tracer1 in tracers1:
        w = np.random.rand(tracer1.neighbours.size) > rejection_fraction
        num_pairs += tracer1.neighbours.size
        num_pairs_used += w.sum()

        for tracer2 in tracers2[tracer1.neighbours[w]]:
            angle = get_angle(tracer1.x_cart, tracer1.y_cart, tracer1.z_cart, tracer1.ra,
                              tracer1.dec, tracer2.x_cart, tracer2.y_cart, tracer2.z_cart,
                              tracer2.ra, tracer2.dec)

            compute_dmat_pair(tracer1.weights, tracer1.z, tracer1.comoving_distance,
                tracer1.comoving_transverse_distance, tracer2.weights, tracer2.z,
                tracer2.comoving_distance, tracer2.comoving_transverse_distance,
                angle, rp_min, rp_max, rt_max, num_bins_rp, num_bins_rt,
                num_bins_rp_model, num_bins_rt_model, distortion, weights_dmat, 
                rp_grid, rt_grid, z_grid, weights_grid)

    return (distortion, weights_dmat, rp_grid, rt_grid, z_grid,
            weights_grid, num_pairs, num_pairs_used)


def compute_dmat(tracers1, tracers2, config, num_cpu):
    if num_cpu < 2:
        return _compute_dmat_kernel(tracers1, tracers2, config)

    split_tracers1 = np.array_split(tracers1, num_cpu)
    arguments = [(local_tracers1, tracers2, config) for local_tracers1 in split_tracers1]
    with Pool(processes=num_cpu) as pool:
        results = pool.starmap(_compute_dmat_kernel, arguments)

    results = np.array(results)
    # distortion = np.sum(results[:, 0, :] * results[:, 1, :], axis=0)
    # weights_dmat = np.sum(results[:, 1, :], axis=0)
    # rp_grid = np.sum(results[:, 2, :] * results[:, 1, :], axis=0)
    # rt_grid = np.sum(results[:, 3, :] * results[:, 1, :], axis=0)
    # z_grid = np.sum(results[:, 4, :] * results[:, 1, :], axis=0)
    # weights_grid = np.sum(results[:, 5, :], axis=0)

    # w = weights_grid > 0
    # xi_grid[w] /= weights_grid[w]
    # rp_grid[w] /= weights_grid[w]
    # rt_grid[w] /= weights_grid[w]
    # z_grid[w] /= weights_grid[w]

    # return xi_grid, weights_grid, rp_grid, rt_grid, z_grid, num_pairs_grid


@njit
def get_num_pairs(distances1, distances2, sin_angle, cos_angle, rp_min, rp_max, rt_max):
    count = 0
    for dc1, dm1 in distances1:
        for dc2, dm2 in distances2:
            rp = np.abs((dc1 - dc2) * cos_angle)
            rt = (dm1 + dm2) * sin_angle

            if (rp < rp_min) or (rp >= rp_max) or (rt >= rt_max):
                continue

            count += 1

    return count


@njit
def get_bin(r, r_min, r_max, r_size):
    return np.int32(np.floor((r - r_min) / (r_max - r_min) * r_size))


@njit
def get_pixel_pairs(distances1, distances2, sin_angle, cos_angle, rp_min, rp_max,
                    rt_max, rp_size, rt_size, rp_size_model, rt_size_model, num_pairs):
    rp_rt_pairs = np.zeros((num_pairs, 2))
    pixel_pairs = np.zeros((num_pairs, 4), dtype=np.int32)

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
            pixel_pairs[k] = (i, j, bin_rt_model + rt_size_model * bin_rp_model + 1,
                              bin_rt + rt_size * bin_rp + 1)
            k += 1

    return pixel_pairs, rp_rt_pairs


@njit
def get_indeces(pixel_pairs, unique_model_bins, unique_data_bins):
    # TODOD The searchsorted function might not work with numba
    for i in range(pixel_pairs.shape[0]):
        pixel_pairs[i, 3] = np.searchsorted(unique_model_bins, pixel_pairs[i, 3])
        pixel_pairs[i, 4] = np.searchsorted(unique_data_bins, pixel_pairs[i, 4])


@njit
def get_etas(weights1, weights2, z1, z2, unique_model_bins, pixel_pairs, rp_rt_pairs):
    num_model_bins = unique_model_bins.size
    pair_rp_eff = np.zeros(num_model_bins)
    pair_rt_eff = np.zeros(num_model_bins)
    pair_z_eff = np.zeros(num_model_bins)

    eta2 = np.zeros((weights1.size, num_model_bins))
    eta3 = np.zeros((weights2.size, num_model_bins))
    eta4 = np.zeros(num_model_bins)
    pixel_pairs_weights = np.zeros(pixel_pairs.shape[0])

    sum_weights1 = np.sum(weights1)
    sum_weights2 = np.sum(weights2)

    for (k, (i, j, mbin, _)) in enumerate(pixel_pairs):
        weight1 = weights1[i]
        weight2 = weights2[j]
        weight12 = weight1 * weight2

        eta2[i, mbin] += weight2 / sum_weights2
        eta3[j, mbin] += weight1 / sum_weights1
        eta4[mbin] += weight12 / sum_weights1 / sum_weights2

        pixel_pairs_weights[k] = weight12
        pair_rp_eff[mbin] += rp_rt_pairs[k][0] * weight12
        pair_rt_eff[mbin] += rp_rt_pairs[k][1] * weight12
        pair_z_eff[mbin] += (z1[i] + z2[j]) / 2 * weight12

    return eta2, eta3, eta4, pixel_pairs_weights, pair_rp_eff, pair_rt_eff, pair_z_eff


@njit
def write_etas(dmat_view, eta2_view, eta3_view, eta4, weight12, num_model_bins):
    for i in range(num_model_bins):
        dmat_view[i] += (eta4[i] - eta2_view[i] - eta3_view[i]) * weight12


@njit
def get_pair_dmat(pixel_pairs, pixel_pairs_weights, unique_model_bins,
                  unique_data_bins, eta2, eta3, eta4):
    num_model_bins = unique_model_bins.size
    pair_dmat = np.zeros((unique_data_bins.size, num_model_bins))
    pair_wdmat = np.zeros(unique_data_bins.size)
    pair_weights_eff = np.zeros(num_model_bins)

    for ((i, j, mbin_index, dbin_index), weight12) in zip(pixel_pairs, pixel_pairs_weights):
        pair_weights_eff[mbin_index] += weight12

        pair_dmat[dbin_index, mbin_index] += weight12
        pair_wdmat[dbin_index] += weight12
        write_etas(pair_dmat[dbin_index, :], eta2[i, :], eta3[j, :], eta4, weight12, num_model_bins)

    return pair_dmat, pair_wdmat, pair_weights_eff


@njit
def compute_dmat_pair(weights1, z1, dc1, dm1, weights2, z2, dc2, dm2, angle, rp_min, rp_max,
                      rt_max, rp_size, rt_size, rp_size_model, rt_size_model,
                      distortion, weights_dmat, rp_grid, rt_grid, z_grid, weights_grid):
    
    sin_angle = np.sin(angle / 2)
    cos_angle = np.cos(angle / 2)

    num_pairs = get_num_pairs(np.c_[dc1, dm1], np.c_[dc2, dm2], sin_angle, cos_angle,
                              rp_min, rp_max, rt_max)

    pixel_pairs, rp_rt_pairs = get_pixel_pairs(np.c_[dc1, dm1], np.c_[dc2, dm2], sin_angle,
                                               cos_angle, rp_min, rp_max, rt_max, rp_size, rt_size,
                                               rp_size_model, rt_size_model, num_pairs)

    unique_model_bins = np.sort(np.unique(pixel_pairs[:, 3]))
    unique_data_bins = np.sort(np.unique(pixel_pairs[:, 4]))
    get_indeces(pixel_pairs, unique_model_bins, unique_data_bins)

    (eta2, eta3, eta4, pixel_pairs_weights,
        pair_rp_eff, pair_rt_eff, pair_z_eff) = get_etas(weights1, weights2, z1, z2,
                                                         unique_model_bins, pixel_pairs,
                                                         rp_rt_pairs)

    rp_grid[unique_model_bins] += pair_rp_eff
    rt_grid[unique_model_bins] += pair_rt_eff
    z_grid[unique_model_bins] += pair_z_eff

    pair_dmat, pair_wdmat, pair_weights_eff = get_pair_dmat(pixel_pairs, pixel_pairs_weights,
                                                            unique_model_bins, unique_data_bins,
                                                            eta2, eta3, eta4)

    distortion[unique_data_bins, unique_model_bins] += pair_dmat
    weights_dmat[unique_data_bins] += pair_wdmat
    weights_grid[unique_model_bins] += pair_weights_eff
