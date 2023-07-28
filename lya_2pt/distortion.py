import sys
import numpy as np
from numba import njit

import lya_2pt.global_data as globals
from lya_2pt.tracer_utils import get_angle
from lya_2pt.compute_utils import fast_dot_product, fast_outer_product
from lya_2pt.compute_utils import get_pixel_pairs_auto, get_pixel_pairs_cross


def compute_dmat(healpix_id):
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
    hp_neighs = [other_hp for other_hp in globals.healpix_neighbours[healpix_id]
                 if other_hp in globals.tracers2]
    hp_neighs += [healpix_id]

    total_size = int(globals.num_bins_rp * globals.num_bins_rt)
    total_size_model = int(globals.num_bins_rp_model * globals.num_bins_rt_model)

    distortion = np.zeros((total_size, total_size_model))
    weights_dmat = np.zeros(total_size)
    rp_grid = np.zeros(total_size_model)
    rt_grid = np.zeros(total_size_model)
    z_grid = np.zeros(total_size_model)
    weights_grid = np.zeros(total_size_model)

    num_pairs = 0
    num_pairs_used = 0
    for tracer1 in globals.tracers1[healpix_id]:
        with globals.lock:
            xicounter = round(globals.counter.value * 100. / globals.num_tracers, 2)
            if (globals.counter.value % 1000 == 0):
                print(("computing dmat: {}%").format(xicounter))
                sys.stdout.flush()
            globals.counter.value += 1

        potential_neighbours = [tracer2 for hp in hp_neighs for tracer2 in globals.tracers2[hp]]

        neighbours = tracer1.get_neighbours(
            potential_neighbours, globals.auto_flag,
            globals.z_min, globals.z_max,
            globals.rp_max, globals.rt_max
            )

        w = np.random.rand(neighbours.size) > globals.rejection_fraction
        num_pairs += neighbours.size
        num_pairs_used += w.sum()

        for tracer2 in neighbours[w]:
            compute_tracer_pair_dmat(
                tracer1, tracer2, distortion, weights_dmat,
                weights_grid, rp_grid, rt_grid, z_grid
            )

    return healpix_id, (distortion, weights_dmat, rp_grid, rt_grid, z_grid,
                        weights_grid, num_pairs, num_pairs_used)


def compute_tracer_pair_dmat(
    tracer1, tracer2, distortion, weights_dmat, weights_grid, rp_grid, rt_grid, z_grid
):
    # Compute angle between the two tracers
    angle = get_angle(
        tracer1.x_cart, tracer1.y_cart, tracer1.z_cart, tracer1.ra, tracer1.dec,
        tracer2.x_cart, tracer2.y_cart, tracer2.z_cart, tracer2.ra, tracer2.dec
    )

    # Find and save all relevant pixel pairs
    if globals.auto_flag:
        pixel_pairs, rp_rt_pairs = get_pixel_pairs_auto(
            tracer1.distances, tracer2.distances, angle)
    else:
        pixel_pairs, rp_rt_pairs = get_pixel_pairs_cross(
            tracer1.distances, tracer2.distances, angle)

    # Identify the unique bins in rp/rt space
    unique_model_bins, unique_data_bins = get_unique_bins(pixel_pairs)

    # Compute old distortion matrix
    if globals.get_old_distortion:
        # Compute all eta pairs and effective coordinate grids
        etas, pixel_pairs_weights, pair_rp_eff, pair_rt_eff, pair_z_eff = get_etas(
            tracer1.weights, tracer1.z, tracer1.order,
            tracer1.sum_weights, tracer1.logwave_term, tracer1.term3_norm,
            tracer2.weights, tracer2.z, tracer2.order,
            tracer2.sum_weights, tracer2.logwave_term, tracer2.term3_norm,
            unique_model_bins, pixel_pairs, rp_rt_pairs,
        )

        # Compute distortion matrix for the tracer pair
        pair_dmat, pair_wdmat, pair_weights_eff = get_pair_dmat(
            pixel_pairs, pixel_pairs_weights, tracer1.logwave_term, tracer2.logwave_term,
            unique_model_bins, unique_data_bins, *etas)

    # Compute new distortion matrix
    else:
        kronecker1_forest2, kronecker2_forest1, forest1_forest2, \
            pixel_pairs_weights, pair_rp_eff, pair_rt_eff, pair_z_eff = compute_inner_sums(
                tracer1.weights, tracer2.weights, tracer1.z, tracer2.z,
                unique_model_bins, pixel_pairs, rp_rt_pairs,
                tracer1.proj_vec_mat, tracer2.proj_vec_mat
                )

        pair_dmat, pair_wdmat, pair_weights_eff = get_general_pair_dmat(
                pixel_pairs, pixel_pairs_weights, unique_model_bins, unique_data_bins,
                kronecker1_forest2, kronecker2_forest1, forest1_forest2,
                tracer1.proj_vec_mat, tracer2.proj_vec_mat
                )

    # Write pair distortion matrix into the big matrices
    distortion[unique_data_bins, unique_model_bins[:, None]] += pair_dmat
    weights_dmat[unique_data_bins] += pair_wdmat
    weights_grid[unique_model_bins] += pair_weights_eff
    rp_grid[unique_model_bins] += pair_rp_eff
    rt_grid[unique_model_bins] += pair_rt_eff
    z_grid[unique_model_bins] += pair_z_eff


@njit
def get_unique_bins(pixel_pairs):
    unique_model_bins = np.sort(np.unique(pixel_pairs[:, 2]))
    unique_data_bins = np.sort(np.unique(pixel_pairs[:, 3]))
    get_indeces(pixel_pairs, unique_model_bins, unique_data_bins)
    return unique_model_bins, unique_data_bins


@njit
def get_indeces(pixel_pairs, unique_model_bins, unique_data_bins):
    for i in range(pixel_pairs.shape[0]):
        pixel_pairs[i, 2] = np.searchsorted(unique_model_bins, pixel_pairs[i, 2])
        pixel_pairs[i, 3] = np.searchsorted(unique_data_bins, pixel_pairs[i, 3])


@njit
def get_etas(
    weights1, z1, order1, sum_weights1, logwave_minus_mean1, term3_norm1,
    weights2, z2, order2, sum_weights2, logwave_minus_mean2, term3_norm2,
    unique_model_bins, pixel_pairs, rp_rt_pairs,
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

    return (eta1, eta2, eta3, eta4, eta5, eta6, eta7, eta8), \
        pixel_pairs_weights, pair_rp_eff, pair_rt_eff, pair_z_eff


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


@njit
def compute_inner_sums(
    weights1, weights2, z1, z2, unique_model_bins, pixel_pairs, rp_rt_pairs, vmat1, vmat2
):
    kronecker1_forest2 = np.zeros((vmat1.shape[0], unique_model_bins.size, vmat2.shape[1]))
    kronecker2_forest1 = np.zeros((vmat2.shape[0], unique_model_bins.size, vmat1.shape[1]))
    forest1_forest2 = np.zeros((unique_model_bins.size, vmat1.shape[1] * vmat2.shape[1]))
    pixel_pairs_weights = np.zeros(pixel_pairs.shape[0])

    pair_rp_eff = np.zeros(unique_model_bins.size)
    pair_rt_eff = np.zeros(unique_model_bins.size)
    pair_z_eff = np.zeros(unique_model_bins.size)

    for (k, (i, j, mbin, _)) in enumerate(pixel_pairs):
        kronecker1_forest2[i, mbin, :] -= vmat2[j, :]
        kronecker2_forest1[j, mbin, :] -= vmat1[i, :]

        for N in range(vmat1.shape[1]):
            for M in range(vmat2.shape[1]):
                forest1_forest2[mbin, M + N * vmat2.shape[1]] += vmat1[i, N] * vmat2[j, M]

        weight12 = weights1[i] * weights2[j]
        pixel_pairs_weights[k] = weight12

        pair_rp_eff[mbin] += rp_rt_pairs[k][0] * weight12
        pair_rt_eff[mbin] += rp_rt_pairs[k][1] * weight12
        pair_z_eff[mbin] += (z1[i] + z2[j]) / 2 * weight12

    return kronecker1_forest2, kronecker2_forest1, forest1_forest2, \
        pixel_pairs_weights, pair_rp_eff, pair_rt_eff, pair_z_eff


@njit
def compute_outer_sum(
    pair_dmat_slice, mbin_size, weight12, kronecker1_forest2_slice, kronecker2_forest1_slice,
    forest1_forest2, vmat1_slice, vmat2_slice, vmat1_vmat2
):
    for mbin_prime in range(mbin_size):
        kronecker1_forest2_sum = fast_dot_product(kronecker1_forest2_slice[mbin_prime], vmat2_slice)
        kronecker2_forest1_sum = fast_dot_product(kronecker2_forest1_slice[mbin_prime], vmat1_slice)
        forest1_forest2_sum = fast_dot_product(forest1_forest2[mbin_prime], vmat1_vmat2)

        pair_dmat_slice[mbin_prime] += weight12 * (kronecker1_forest2_sum
                                                   + kronecker2_forest1_sum
                                                   + forest1_forest2_sum)


@njit
def get_general_pair_dmat(
    pixel_pairs, pixel_pairs_weights, unique_model_bins, unique_data_bins,
    kronecker1_forest2, kronecker2_forest1, forest1_forest2, vmat1, vmat2
):
    pair_dmat = np.zeros((unique_data_bins.size, unique_model_bins.size))
    pair_wdmat = np.zeros(unique_data_bins.size)
    pair_weights_eff = np.zeros(unique_model_bins.size)

    for ((i, j, mbin, dbin), weight12) in zip(pixel_pairs, pixel_pairs_weights):
        pair_wdmat[dbin] += weight12
        pair_weights_eff[mbin] += weight12

        pair_dmat[dbin, mbin] += weight12
        vmat1_vmat2 = fast_outer_product(vmat1[i], vmat2[j])

        compute_outer_sum(
            pair_dmat[dbin], unique_model_bins.size, weight12,
            kronecker1_forest2[i], kronecker2_forest1[j],
            forest1_forest2, vmat1[i], vmat2[j], vmat1_vmat2)

    return pair_dmat, pair_wdmat, pair_weights_eff
