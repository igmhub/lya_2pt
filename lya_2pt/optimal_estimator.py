import numpy as np
from numba import njit
from scipy.constants import speed_of_light
from scipy.interpolate import RegularGridInterpolator as RGI

import lya_2pt.global_data as globals


def fiducial_Pk(kv, z, A=0.066, k0=0.009, z0=3.0, n=-2.685, alpha=-0.22, k1=0.053, B=3.59, beta=-0.18):
    log_k_k0 = np.log(kv / k0 + 1e-10)
    pk = A * np.power(kv / k0, 2 + n + alpha * log_k_k0) / (1 + (kv / k1)**2)
    z_evol = np.power((1 + z) / (1 + z0), B + beta * log_k_k0)
    return pk * z_evol * np.pi / k0


def fiducial_Pk_angstrom(kA, z, A=0.066, k0=0.009, z0=3.0, n=-2.685, alpha=-0.22, k1=0.053, B=3.59, beta=-0.18):
    kv = kA * 1215.67 * (1 + z) / (speed_of_light / 1000) 
    return fiducial_Pk(kv, z, A, k0, z0, n, alpha, k1, B, beta) * 1215.67 * (1 + z) / (speed_of_light / 1000)


def window_squared_angstrom(k, delta_lambda=2.4, R=0.8):
    return (np.exp(-k**2 * R**2 / 2) * np.sinc(k * delta_lambda / 2 / np.pi))**2


def build_xi1d():
    z = np.linspace(1.8, 4, 50)

    dlambda = 0.01
    r = dlambda * (np.arange(2**20) - 2**19)
    k = np.fft.rfftfreq(r.size, d=dlambda) * 2 * np.pi

    kk, zz = np.meshgrid(k, z)

    fid_pk_angstrom = fiducial_Pk_angstrom(kk, zz)

    xi_wwindow = np.fft.irfft(fid_pk_angstrom * window_squared_angstrom(k)) / dlambda
    xi_wwindow = np.fft.fftshift(xi_wwindow)

    xi1d_interp = RGI((r, z), xi_wwindow.T, method='linear', bounds_error=False)
    return xi1d_interp


def build_covariance(tracer, xi1d_interp):
    z_ij = np.sqrt((1 + tracer.z[:, None]) * (1 + tracer.z[None, :])) - 1
    indeces = np.arange(tracer.z.size)
    wave_bin_size = 10**tracer.log_lambda[1] - 10**tracer.log_lambda[0]

    delta_lambdas = np.abs(indeces[:, None] - indeces[None, :]) * wave_bin_size
    covariance = xi1d_interp((delta_lambdas, z_ij))

    np.fill_diagonal(covariance, covariance.diagonal() + 1 / tracer.ivar)

    return np.linalg.inv(covariance)


def get_xi_bins(tracer1, tracer2, angle):
    rp = np.abs(np.subtract.outer(tracer1.dist_c, tracer2.dist_c) * np.cos(angle/2))
    rt = np.add.outer(tracer1.dist_m, tracer2.dist_m) * np.sin(angle/2)

    mask = (rp >= globals.rp_min) & (rp < globals.rp_max) & (rt < globals.rt_max)
    bins_rp = ((rp - globals.rp_min) / (globals.rp_max - globals.rp_min) 
               * globals.num_bins_rp).astype(int)
    bins_rt = np.floor(rt / globals.rt_max * globals.num_bins_rt).astype(int)
    bins = (bins_rt + globals.num_bins_rt * bins_rp).astype(int)

    return bins


def build_deriv(xi_size, bins):
    c_deriv_dict = {}
    c_deriv_masks = {}
    for bin_index in np.unique(bins):
        if bin_index >= xi_size:
            continue

        mask2 = np.array(np.where(bins == bin_index)).T
        mask2 = mask2[mask2[:, 1].argsort()].T

        c_deriv_dict[bin_index] = np.zeros(xi.shape)
        c_deriv_dict[bin_index][mask2[0], mask2[1]] = 1
        c_deriv_masks[bin_index] = mask2

    return c_deriv_dict, c_deriv_masks


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
        elif mask[1][i+1] != col_idx:
            row_idx = mask[0][start:i+1]
            prod[:, col_idx] += np.sum(invcov[:, row_idx], axis=1)

            start = i + 1
            end = i + 1
        else:
            end += 1

    return prod


def compute_xi_est(tracer1, tracer2):
    pass


def compute_fisher(tracer1, tracer2):
    pass


def compute_xi_and_fisher_with_vectors(tracer1, tracer2, c_deriv_masks):
    c_deriv_dict = {}
    for key, mask1 in c_deriv_masks.items():
        c_deriv_dict[key] = np.zeros((tracer1.deltas.size, tracer2.deltas.size), dtype=np.float_)
        c_deriv_dict[key][mask1[0], mask1[1]] = 1

    xi_est = []
    fisher = []
    for key1, c_deriv1 in c_deriv_dict.items():
        xi = c_deriv1 @ (invcov2 @ tracer2.deltas)
        xi = 2 * (tracer1.deltas @ invcov1) @ xi

        xi_est[key1] += xi

        c1_inv_times_c_deriv = invcov1 @ c_deriv1

        for key2, c_deriv2 in c_deriv_dict.items():
            c2_inv_times_c_deriv = invcov2 @ c_deriv2.T

            fisher[key1, key2] += c1_inv_times_c_deriv.flatten().dot(c2_inv_times_c_deriv.flatten())
