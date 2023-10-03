import numpy as np
from numba import njit
from scipy.constants import speed_of_light
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.sparse import coo_array

import lya_2pt.global_data as globals


@njit
def compute_raw(invcov, mask, size1, size2):
    """Multiply inverse covariance by the derivative matrix without computing the
    derivative matrix.
    """
    prod = np.zeros((size1, size2))
    mask_size = len(mask[1])

    start = 0
    end = 0
    for i, col_idx in enumerate(mask[1]):
        if i == (mask_size - 1):
            row_idx = mask[0][start:mask_size]
            prod[:, col_idx] += np.sum(invcov[:, row_idx], axis=1)

            start = i + 1
            end = i + 1
        elif mask[1][i + 1] != col_idx:
            row_idx = mask[0][start:i + 1]
            prod[:, col_idx] += np.sum(invcov[:, row_idx], axis=1)

            start = i + 1
            end = i + 1
        else:
            end += 1

    return prod


def fiducial_Pk_angstrom(
        kA, z, k0=0.009, z0=3.0, A=0.066, n=-2.685, alpha=-0.22, k1=0.053, B=3.59, beta=-0.18
):
    c_kms = speed_of_light / 1000
    A2kms = 1215.67 * (1 + z) / c_kms  # A / (km/s)
    kv = kA * A2kms

    log_k_k0 = np.log(kv / k0 + 1e-10)
    pk = A * np.power(kv / k0, 2 + n + alpha * log_k_k0) / (1 + (kv / k1)**2)
    z_evol = np.power((1 + z) / (1 + z0), B + beta * log_k_k0)
    pk *= z_evol * np.pi / k0  # km / s

    return pk * A2kms


def window_squared_angstrom(k, delta_lambda=2.4, R=0.8):
    return np.exp(-k**2 * R**2) * np.sinc(k * delta_lambda / 2 / np.pi)**2


def build_xi1d(z1=1.8, z2=4., nz=200, lambda_max=2048):
    z = np.linspace(z1, z2, nz)

    r, dlambda = np.linspace(-lambda_max, lambda_max, int(10 * lambda_max) + 1, retstep=True)
    k = np.fft.rfftfreq(r.size, d=dlambda) * 2 * np.pi

    kk, zz = np.meshgrid(k, z)

    fid_pk_angstrom = fiducial_Pk_angstrom(kk, zz)

    xi_wwindow = np.fft.irfft(fid_pk_angstrom * window_squared_angstrom(k), n=r.size) / dlambda
    xi_wwindow = np.fft.fftshift(xi_wwindow, axes=1)

    return RGI((z, r), xi_wwindow, method='cubic', bounds_error=False)


def get_xi_bins(tracer1, tracer2, angle):
    rp = np.abs(np.subtract.outer(tracer1.dist_c, tracer2.dist_c) * np.cos(angle / 2))
    rt = np.add.outer(tracer1.dist_m, tracer2.dist_m) * np.sin(angle / 2)

    mask = (rp >= globals.rp_min) & (rp < globals.rp_max) & (rt < globals.rt_max)
    bins_rp = (
        (rp - globals.rp_min) / (globals.rp_max - globals.rp_min) * globals.num_bins_rp
    ).astype(int)
    bins_rt = (rt / globals.rt_max * globals.num_bins_rt).astype(int)
    bins = (bins_rt + globals.num_bins_rt * bins_rp).astype(int)
    bins[~mask] = -1

    return bins


def get_xi_bins_t(tracer1, tracer2, angle):
    rp = np.abs(np.subtract.outer(tracer1.dist_c, tracer2.dist_c) * np.cos(angle / 2))
    rt = np.add.outer(tracer1.dist_m, tracer2.dist_m) * np.sin(angle / 2)

    mask = (rp >= globals.rp_min) & (rp < globals.rp_max) & (rt < globals.rt_max)

    bins_rp = (
        (rp - globals.rp_min) / (globals.rp_max - globals.rp_min) * globals.num_bins_rp
    ).astype(int)
    bins_rt = (rt / globals.rt_max * globals.num_bins_rt).astype(int)
    bins = (bins_rp + globals.num_bins_rp * bins_rt).astype(int)
    bins[~mask] = -1

    return bins


def build_deriv(bins):
    unique_bins = np.unique(bins)
    if unique_bins[0] == -1:
        unique_bins = unique_bins[1:]

    idx_list = [np.nonzero(bins == bin_index) for bin_index in unique_bins]
    c_deriv_list = [
        coo_array((np.ones(idx[0].size), idx), shape=bins.shape).tocsr()
        for idx in idx_list
    ]

    idx_minmax_list = [
        (idx[0].min(), idx[0].max() + 1, idx[1].min(), idx[1].max() + 1) for idx in idx_list]

    return unique_bins, c_deriv_list, idx_minmax_list


def build_deriv_bysort(bins):
    """ Build list of derivative matrices by sorting. Performance depends on number of bins and
    the shape of bins.
    Return a list of tuples: (bin_index, C_deriv), sorted by bin_index
    """
    nrows, ncols = bins.shape
    bins_flat = bins.ravel()
    idx_sort = bins_flat.argsort()  # j + i * ncols
    bins_flat = bins_flat[idx_sort]

    unique_bins, unique_indices = np.unique(bins_flat, return_index=True)
    split_bins = np.split(idx_sort, unique_indices[1:])
    if unique_bins[0] == -1:
        unique_bins = unique_bins[1:]
        split_bins = split_bins[1:]

    c_deriv_list = [
        (bin_index, coo_array((np.ones(idx.size), divmod(idx, ncols)), shape=bins.shape).tocsr())
        for bin_index, idx in zip(unique_bins, split_bins)
    ]

    return c_deriv_list


def compute_xi_and_fisher_pair(
        tracer1, tracer2, angle,
        xi_est, fisher_est
):
    bins = get_xi_bins_t(tracer1, tracer2, angle)
    unique_bins, c_deriv_list, idx_minmax_list = build_deriv(bins)
    n_unique_bins = unique_bins.size

    # deltas are weighted before this function is called
    xi_est[unique_bins] += np.fromiter((
        2 * np.dot(tracer1.deltas, c_deriv.dot(tracer2.deltas)) for c_deriv in c_deriv_list
    ), float, n_unique_bins)

    invcov1_x_c_deriv_list = [c_deriv.T.dot(tracer1.invcov).T for c_deriv in c_deriv_list]
    row_slices = [np.s_[rmin:rmax] for (rmin, rmax, _, _) in idx_minmax_list]
    col_slices = [np.s_[cmin:cmax] for (_, _, cmin, cmax) in idx_minmax_list]

    for i, (bin1, c_deriv, rs) in enumerate(zip(unique_bins, c_deriv_list, row_slices)):
        c_deriv_x_invcov2 = c_deriv[rs].dot(tracer2.invcov)

        fisher_est[bin1, unique_bins[i:]] += np.fromiter((
            np.vdot(c_deriv_x_invcov2[:, cs], invcov1_x_c_deriv[rs, cs])
            for invcov1_x_c_deriv, cs in zip(invcov1_x_c_deriv_list[i:], col_slices[i:])
        ), float, n_unique_bins - i)

    # return xi_est, fisher_est


def compute_xi_and_fisher(healpix_id):
    """Compute correlation function

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
    (array, array, array, array, array, array)
        correlation function, sum of weights in each bin, line-of-sight separation grid,
        transverse separation grid, redshift grid, number of pixel pairs in each bin
    """
    hp_neighs = [other_hp for other_hp in globals.healpix_neighbours[healpix_id]
                 if other_hp in globals.tracers2]
    hp_neighs += [healpix_id]

    total_size = int(globals.num_bins_rp * globals.num_bins_rt)

    xi_est = np.zeros(total_size)
    fisher_est = np.zeros((total_size, total_size))
    xi1d_interp = build_xi1d()

    for tracer1 in globals.tracers1[healpix_id]:
        potential_neighbours = [tracer2 for hp in hp_neighs for tracer2 in globals.tracers2[hp]]

        neighbours, angles = tracer1.get_neighbours(
            potential_neighbours, globals.auto_flag,
            globals.z_min, globals.z_max,
            globals.rp_max, globals.rt_max
        )

        tracer1.set_inverse_covariance(xi1d_interp, globals.continuum_order)
        tracer1.apply_invcov_to_deltas()

        np.random.seed(globals.seed)
        w = np.random.rand(neighbours.size) > globals.rejection_fraction
        for tracer2, angle in zip(neighbours[w], angles[w]):
            tracer2.set_inverse_covariance(xi1d_interp, globals.continuum_order)
            tracer2.apply_invcov_to_deltas()
            compute_xi_and_fisher_pair(tracer1, tracer2, angle, xi_est, fisher_est)
            tracer2.release_inverse_covariance()

        tracer1.release_inverse_covariance()

    return healpix_id, (xi_est, fisher_est)


def compute_num_pairs(healpix_id):
    hp_neighs = [other_hp for other_hp in globals.healpix_neighbours[healpix_id]
                 if other_hp in globals.tracers2]
    hp_neighs += [healpix_id]

    num_pairs = int(0)
    for tracer1 in globals.tracers1[healpix_id]:
        potential_neighbours = [tracer2 for hp in hp_neighs for tracer2 in globals.tracers2[hp]]
        neighbours, _ = tracer1.get_neighbours(
                potential_neighbours, globals.auto_flag,
                globals.z_min, globals.z_max,
                globals.rp_max, globals.rt_max
            )

        num_pairs += len(neighbours)

    return num_pairs
