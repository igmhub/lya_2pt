import sys

import numpy as np
from numba import njit
from scipy.constants import speed_of_light
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.sparse import coo_array

import lya_2pt.global_data as globals
from lya_2pt.tracer_utils import get_angle


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


def build_xi1d(z1=1.8, z2=4., nz=50, lambda_max=2048):
    z = np.linspace(z1, z2, nz)

    r, dlambda = np.linspace(-lambda_max, lambda_max, int(10 * lambda_max) + 1, retstep=True)
    k = np.fft.rfftfreq(r.size, d=dlambda) * 2 * np.pi

    kk, zz = np.meshgrid(k, z)

    fid_pk_angstrom = fiducial_Pk_angstrom(kk, zz)

    xi_wwindow = np.fft.irfft(fid_pk_angstrom * window_squared_angstrom(k), n=r.size) / dlambda
    xi_wwindow = np.fft.fftshift(xi_wwindow, axes=1)

    return RGI((z, r), xi_wwindow, method='linear', bounds_error=False)


def build_inverse_covariance(tracer, xi1d_interp):
    z_ij = np.sqrt((1 + tracer.z[:, None]) * (1 + tracer.z[None, :])) - 1
    wavelength = 10**tracer.log_lambda

    delta_lambdas = wavelength[:, None] - wavelength[None, :]
    covariance = xi1d_interp((z_ij, delta_lambdas))
    covariance[np.diag_indices(tracer.z.size)] += 1 / tracer.ivar
    # covariance[np.diag_indices(tracer.z.size)] += 1 / tracer.weights

    return np.linalg.inv(covariance)


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

    def my_coo_array(idx):
        return coo_array((np.ones(idx[0].size, idx)), shape=bins.shape).tocsr()

    idx_list = [np.nonzero(bins == bin_index) for bin_index in unique_bins]
    c_deriv_list = [my_coo_array(idx) for idx in idx_list]
    rminmax_list = [(idx[0].min(), idx[0].max() + 1) for idx in idx_list]

    return unique_bins, c_deriv_list, rminmax_list


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


def compute_xi_and_fisher_pair(
        tracer1, tracer2, angle, xi1d_interp,
        xi_est, fisher_est
):
    invcov1 = build_inverse_covariance(tracer1, xi1d_interp)
    invcov2 = build_inverse_covariance(tracer2, xi1d_interp)

    bins = get_xi_bins_t(tracer1, tracer2, angle)
    unique_bins, c_deriv_list, rminmax_list = build_deriv(bins)

    y1 = invcov1 @ tracer1.deltas
    y2 = invcov2 @ tracer2.deltas
    xi = np.array([2 * np.dot(y1, c_deriv.dot(y2)) for c_deriv in c_deriv_list])
    xi_est[unique_bins] += xi

    invcov1_x_c_deriv_list = [c_deriv.T.dot(invcov1).T for c_deriv in c_deriv_list]
    c_deriv_x_invcov2_list = [c_deriv.dot(invcov2) for c_deriv in c_deriv_list]

    for i, (bin1, c_deriv_x_invcov2, (rmin1, rmax1)) in enumerate(
            zip(unique_bins, c_deriv_x_invcov2_list, rminmax_list)
    ):
        s_list = [
            np.s_[max(rmin1, rmin2):min(rmax1, rmin2)]
            for rmin2, rmax2 in rminmax_list[i:]
        ]

        fisher_est[bin1, unique_bins[i:]] += np.array([
            np.vdot(c_deriv_x_invcov2[s], invcov1_x_c_deriv[s])
            for s, invcov1_x_c_deriv in zip(s_list, invcov1_x_c_deriv_list[i:])
        ])

    return xi_est, fisher_est


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
        # with globals.lock:
        #     xicounter = round(globals.counter.value * 100. / globals.num_tracers, 2)
        #     if (globals.counter.value % 10 == 0):
        #         print(("computing xi: {}%").format(xicounter))
        #         sys.stdout.flush()
        #     globals.counter.value += 1

        potential_neighbours = [tracer2 for hp in hp_neighs for tracer2 in globals.tracers2[hp]]

        neighbours = tracer1.get_neighbours(
            potential_neighbours, globals.auto_flag,
            globals.z_min, globals.z_max,
            globals.rp_max, globals.rt_max
            )

        for tracer2 in neighbours:
            angle = get_angle(
                tracer1.x_cart, tracer1.y_cart, tracer1.z_cart, tracer1.ra, tracer1.dec,
                tracer2.x_cart, tracer2.y_cart, tracer2.z_cart, tracer2.ra, tracer2.dec
                )

            compute_xi_and_fisher_pair(tracer1, tracer2, angle, xi1d_interp, xi_est, fisher_est)

    return healpix_id, (xi_est, fisher_est)
