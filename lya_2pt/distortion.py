import numpy as np
from numba import njit
from lya_2pt.tracer_utils import get_angle
from lya_2pt.compute_utils import get_num_pairs, get_bin


def compute_dmat(tracers1, tracers2, config, auto_flag=False):
    """Compute distortion matrix

    Parameters
    ----------
    tracers1 : array of lya_2pt.tracer.Tracer
        First set of tracers
    tracers2 : array of lya_2pt.tracer.Tracer
        Second set of tracers
    config : ConfigParser
        Internal configuration object containing the settings section
    auto_flag : bool, optional
        Flag for auto-correlation, by default False

    Returns
    -------
    (array, array, array, array, array, array, array, array)
        distortion matrix, weight matrix, line-of-sight separation grid,
        transverse separation grid, redshift grid, sum of weights in each bin,
        total number of pairs per bin, number of pairs used per bin
    """
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
        w = np.random.rand(tracer1.num_neighbours) > rejection_fraction
        num_pairs += tracer1.neighbours.size
        num_pairs_used += w.sum()

        for tracer2 in tracers2[tracer1.neighbours][w]:
            angle = get_angle(
                tracer1.x_cart, tracer1.y_cart, tracer1.z_cart, tracer1.ra, tracer1.dec,
                tracer2.x_cart, tracer2.y_cart, tracer2.z_cart, tracer2.ra, tracer2.dec
                )

            compute_dmat_pair(
                tracer1.weights, tracer1.z, tracer1.distances, tracer1.order,
                tracer1.sum_weights, tracer1.logwave_term, tracer1.term3_norm,
                tracer2.weights, tracer2.z, tracer2.distances, tracer2.order,
                tracer2.sum_weights, tracer2.logwave_term, tracer2.term3_norm,
                angle, auto_flag, rp_min, rp_max, rt_max, num_bins_rp, num_bins_rt,
                num_bins_rp_model, num_bins_rt_model, distortion, weights_dmat,
                rp_grid, rt_grid, z_grid, weights_grid
                )

    return (distortion, weights_dmat, rp_grid, rt_grid, z_grid,
            weights_grid, num_pairs, num_pairs_used)


# TODO This function could benefit from being jitted if we moved the distortion write line outside
# @njit
def compute_dmat_pair(
        weights1, z1, distances1, order1, sum_weights1, logwave_term1, term3_norm1,
        weights2, z2, distances2, order2, sum_weights2, logwave_term2, term3_norm2, angle,
        auto_flag, rp_min, rp_max, rt_max, rp_size, rt_size, rp_size_model, rt_size_model,
        distortion, weights_dmat, rp_grid, rt_grid, z_grid, weights_grid
):
    sin_angle = np.sin(angle / 2)
    cos_angle = np.cos(angle / 2)

    num_pairs = get_num_pairs(
        distances1, distances2, sin_angle, cos_angle, rp_min, rp_max, rt_max, auto_flag)

    pixel_pairs, rp_rt_pairs = get_pixel_pairs(
        distances1, distances2, sin_angle, cos_angle, rp_min, rp_max, rt_max,
        rp_size, rt_size, rp_size_model, rt_size_model, num_pairs
        )

    unique_model_bins = np.sort(np.unique(pixel_pairs[:, 2]))
    unique_data_bins = np.sort(np.unique(pixel_pairs[:, 3]))
    get_indeces(pixel_pairs, unique_model_bins, unique_data_bins)

    etas, pixel_pairs_weights = get_etas(
        weights1, z1, order1, sum_weights1, logwave_term1, term3_norm1,
        weights2, z2, order2, sum_weights2, logwave_term2, term3_norm2,
        unique_model_bins, pixel_pairs, rp_rt_pairs, rp_grid, rt_grid, z_grid
        )

    pair_dmat, pair_wdmat, pair_weights_eff = get_pair_dmat(
        pixel_pairs, pixel_pairs_weights, logwave_term1, logwave_term2,
        unique_model_bins, unique_data_bins, *etas)

    distortion[unique_data_bins, unique_model_bins[:, None]] += pair_dmat
    weights_dmat[unique_data_bins] += pair_wdmat
    weights_grid[unique_model_bins] += pair_weights_eff


@njit
def get_pixel_pairs(
    distances1, distances2, sin_angle, cos_angle, rp_min, rp_max,
    rt_max, rp_size, rt_size, rp_size_model, rt_size_model, num_pairs
):
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
            pixel_pairs[k] = (i, j, bin_rt_model + rt_size_model * bin_rp_model,
                              bin_rt + rt_size * bin_rp)
            k += 1

    return pixel_pairs, rp_rt_pairs


@njit
def get_indeces(pixel_pairs, unique_model_bins, unique_data_bins):
    for i in range(pixel_pairs.shape[0]):
        pixel_pairs[i, 2] = np.searchsorted(unique_model_bins, pixel_pairs[i, 2])
        pixel_pairs[i, 3] = np.searchsorted(unique_data_bins, pixel_pairs[i, 3])


@njit
def get_etas(
    weights1, z1, order1, sum_weights1, logwave_minus_mean1, term3_norm1,
    weights2, z2, order2, sum_weights2, logwave_minus_mean2, term3_norm2,
    unique_model_bins, pixel_pairs, rp_rt_pairs, rp_grid, rt_grid, z_grid
):
    num_model_bins = unique_model_bins.size
    pair_rp_eff = np.zeros(num_model_bins)
    pair_rt_eff = np.zeros(num_model_bins)
    pair_z_eff = np.zeros(num_model_bins)

    # Projection matrix has 3 terms, therefore distortion matrix has 3^2 terms
    # The first term (eta0) is trivial and thus only added at the end
    eta1 = np.zeros((weights1.size, num_model_bins))  # term 1 x term 2
    eta2 = np.zeros((weights2.size, num_model_bins))  # term 2 x term 1
    eta3 = np.zeros((weights1.size, num_model_bins))  # term 1 x term 3
    eta4 = np.zeros((weights2.size, num_model_bins))  # term 3 x term 1
    eta5 = np.zeros(num_model_bins)  # term 2 x term 2
    eta6 = np.zeros(num_model_bins)  # term 2 x term 3
    eta7 = np.zeros(num_model_bins)  # term 3 x term 2
    eta8 = np.zeros(num_model_bins)  # term 3 x term 3
    pixel_pairs_weights = np.zeros(pixel_pairs.shape[0])

    for (k, (i, j, mbin, _)) in enumerate(pixel_pairs):
        weight1 = weights1[i]
        weight2 = weights2[j]
        weight12 = weight1 * weight2

        log_wave_term1 = logwave_minus_mean1[i]
        log_wave_term2 = logwave_minus_mean2[j]

        eta1[i, mbin] -= weight2 / sum_weights2
        eta2[j, mbin] -= weight1 / sum_weights1
        eta5[mbin] += weight12 / sum_weights1 / sum_weights2

        if order2 == 1:
            eta3[i, mbin] -= weight2 * log_wave_term2 / term3_norm2
            eta6[mbin] += weight12 * log_wave_term2 / sum_weights1 / term3_norm2

        if order1 == 1:
            eta4[j, mbin] -= weight1 * log_wave_term1 / term3_norm1
            eta7[mbin] += weight12 * log_wave_term1 / sum_weights2 / term3_norm1

            if order2 == 1:
                eta8[mbin] += weight12 * log_wave_term1 * log_wave_term2 / term3_norm1 / term3_norm2

        pixel_pairs_weights[k] = weight12
        pair_rp_eff[mbin] += rp_rt_pairs[k][0] * weight12
        pair_rt_eff[mbin] += rp_rt_pairs[k][1] * weight12
        pair_z_eff[mbin] += (z1[i] + z2[j]) / 2 * weight12

    rp_grid[unique_model_bins] += pair_rp_eff
    rt_grid[unique_model_bins] += pair_rt_eff
    z_grid[unique_model_bins] += pair_z_eff

    return (eta1, eta2, eta3, eta4, eta5, eta6, eta7, eta8), pixel_pairs_weights


@njit
def write_etas(
    num_model_bins, weight12, log_wave_term1, log_wave_term2,
    eta1_view, eta2_view, eta3_view, eta4_view, eta5, eta6, eta7, eta8, dmat_view
):
    for i in range(num_model_bins):
        dmat_view[i] += weight12 * (
            eta1_view[i]
            + eta2_view[i]
            + eta3_view[i] * log_wave_term2
            + eta4_view[i] * log_wave_term1
            + eta5[i]
            + eta6[i] * log_wave_term2
            + eta7[i] * log_wave_term1
            + eta8[i] * log_wave_term1 * log_wave_term2
            )


@njit
def get_pair_dmat(
    pixel_pairs, pixel_pairs_weights, logwave_minus_mean1, logwave_minus_mean2,
    unique_model_bins, unique_data_bins, eta1, eta2, eta3, eta4, eta5, eta6, eta7, eta8
):
    num_model_bins = unique_model_bins.size
    pair_dmat = np.zeros((unique_data_bins.size, num_model_bins))
    pair_wdmat = np.zeros(unique_data_bins.size)
    pair_weights_eff = np.zeros(num_model_bins)

    for ((i, j, mbin_index, dbin_index), weight12) in zip(pixel_pairs, pixel_pairs_weights):
        log_wave_term1 = logwave_minus_mean1[i]
        log_wave_term2 = logwave_minus_mean2[j]

        pair_weights_eff[mbin_index] += weight12

        pair_dmat[dbin_index, mbin_index] += weight12
        pair_wdmat[dbin_index] += weight12

        write_etas(num_model_bins, weight12, log_wave_term1, log_wave_term2,
                   eta1[i, :], eta2[j, :], eta3[i, :], eta4[j, :],
                   eta5, eta6, eta7, eta8, pair_dmat[dbin_index, :])

    return pair_dmat, pair_wdmat, pair_weights_eff
