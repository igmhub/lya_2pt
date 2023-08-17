import numpy as np
from numba import njit

SMALL_ANGLE_CUT_OFF = 2./3600.*np.pi/180.  # 2 arcsec


def get_angle_list(tracer1, tracers2):
    """Compute angle between two tracers"""
    x_cart = np.array([t2.x_cart for t2 in tracers2])
    y_cart = np.array([t2.y_cart for t2 in tracers2])
    z_cart = np.array([t2.z_cart for t2 in tracers2])
    ra = np.array([t2.ra for t2 in tracers2])
    dec = np.array([t2.dec for t2 in tracers2])

    cos = x_cart * tracer1.x_cart + y_cart * tracer1.y_cart + z_cart * tracer1.z_cart
    w = cos >= 1.
    if w.sum() != 0:
        cos[w] = 1.
    w = cos <= -1.
    if w.sum() != 0:
        cos[w] = -1.
    angles = np.arccos(cos)

    w = ((np.abs(ra - tracer1.ra) < SMALL_ANGLE_CUT_OFF) &
         (np.abs(dec - tracer1.dec) < SMALL_ANGLE_CUT_OFF))
    if w.sum() != 0:
        angles[w] = np.sqrt((dec[w] - tracer1.dec)**2 +
                            (np.cos(tracer1.dec) * (ra[w] - tracer1.ra))**2)

    return angles


@njit
def get_angle(x1, y1, z1, ra1, dec1, x2, y2, z2, ra2, dec2):
    """Compute angle between two tracers"""
    cos = x1 * x2 + y1 * y2 + z1 * z2
    if cos >= 1.:
        cos = 1.
    elif cos <= -1.:
        cos = -1.
    angle = np.arccos(cos)

    if ((np.abs(ra2 - ra1) < SMALL_ANGLE_CUT_OFF) & (np.abs(dec2 - dec1) < SMALL_ANGLE_CUT_OFF)):
        angle = np.sqrt((dec2 - dec1)**2 + (np.cos(dec1) * (ra2 - ra1))**2)

    return angle


# def get_angle(tracer1, tracer2):
#     """Compute angle between two tracers of Tracer type

#     Arguments
#     ---------
#     tracer1 : Tracer
#     First tracer

#     tracer2 : Tracer
#     Second tracer

#     Return
#     ------
#     angle: float
#     Angle between tracer1 and tracer2
#     """
#     cos = (tracer2.x_cart * tracer1.x_cart + tracer2.y_cart * tracer1.y_cart
#            + tracer2.z_cart * tracer1.z_cart)

#     if cos >= 1.:
#         cos = 1.
#     elif cos <= -1.:
#         cos = -1.
#     angle = np.arccos(cos)

#     if ((np.abs(tracer2.ra - tracer1.ra) < SMALL_ANGLE_CUT_OFF)
#       & (np.abs(tracer2.dec - tracer1.dec) < SMALL_ANGLE_CUT_OFF)):
#         angle = np.sqrt((tracer2.dec - tracer1.dec)**2
#                         + (np.cos(tracer1.dec) * (tracer2.ra - tracer1.ra))**2)

#     return angle


# @njit()
def rebin(log_lambda, deltas, weights, ivar, rebin_factor, dwave, use_ivar=False):
    """Rebin a Tracer by combining N pixels together

    Arguments
    ---------
    log_lambda: array of float
    An array with the logarithm of the wavelength

    deltas: array of float
    An array with the delta field

    weights: array of float
    An array with the weights associated with the delta field

    rebin_factor: int
    Number of pixels to merge together

    wave_solution: "lin" or "log"
    Specifies whether the underlying wavelength grid is evenly
    spaced on wavelength (lin) or on the logarithm of the wavelength (log)

    Return
    ------
    rebin_log_lambda: array of float
    The rebinned array for the logarithm of the wavelength

    rebin_deltas: array of float
    The rebinned array for the deltas

    rebin_weight: array of float
    The rebinned array for the weights
    """
    wave = 10**np.array(log_lambda)

    start = wave.min() - dwave / 2
    num_bins = np.ceil(((wave[-1] - wave[0]) / dwave + 1) / rebin_factor)

    edges = np.arange(num_bins) * dwave * rebin_factor + start

    new_indx = np.searchsorted(edges, wave)
    binned_weight = np.bincount(new_indx, weights=weights, minlength=edges.size+1)[1:-1]
    binned_ivar = np.bincount(new_indx, weights=ivar, minlength=edges.size+1)[1:-1]

    if use_ivar:
        binned_delta = np.bincount(new_indx, weights=deltas*ivar, minlength=edges.size+1)[1:-1]
        mask = binned_ivar != 0
        binned_delta[mask] /= binned_ivar[mask]
    else:
        binned_delta = np.bincount(new_indx, weights=deltas*weights, minlength=edges.size+1)[1:-1]
        mask = binned_weight != 0
        binned_delta[mask] /= binned_weight[mask]

    new_wave = (edges[1:] + edges[:-1]) / 2

    return np.log10(new_wave[mask]), binned_delta[mask], binned_weight[mask], binned_ivar[mask]


@njit()
def project_deltas(log_lambda, deltas, weights, order):
    """Project the delta field

    The projection gets rid of the distortion caused by the continuum
    fitiing. See equations 5 and 6 of du Mas des Bourboux et al. 2020

    Arguments
    ---------
    log_lambda: array of float
    An array with the logarithm of the wavelength

    deltas: array of float
    An array with the delta field

    weights: array of float
    An array with the weights associated with the delta field

    order: int
    Order of the polynomial used to fit the continuum

    Return
    ------
    projected_deltas: array of float
    The projected deltas. If the sum of weights is zero, then the function
    does nothing and returns the original deltas
    """
    # 2nd term in equation 6
    sum_weights = np.sum(weights)
    if sum_weights > 0.0:
        mean_delta = np.average(deltas, weights=weights)
    else:
        # TODO: write a warning
        return deltas

    projected_deltas = deltas - mean_delta

    # 3rd term in equation 6
    if order == 1:
        mean_log_lambda = np.average(log_lambda, weights=weights)
        meanless_log_lambda = log_lambda - mean_log_lambda
        mean_delta_log_lambda = (
            np.sum(weights * deltas * meanless_log_lambda) /
            np.sum(weights * meanless_log_lambda**2))
        projected_deltas -= mean_delta_log_lambda * meanless_log_lambda

    return projected_deltas


@njit()
def get_projection_matrix(log_lambda, weights, order):
    input_vectors_matrix = np.ones((order + 1, weights.size)) * weights
    for i in range(1, order + 1):
        input_vectors_matrix[i] *= log_lambda**i

    _, __, Vh = np.linalg.svd(input_vectors_matrix, full_matrices=False)
    return Vh.T, np.eye(weights.size) - Vh.T @ Vh


def gram_schmidt(log_lambda, weights, order):
    basis = []
    for n in range(order + 1):
        v = log_lambda**n
        for b in basis:
            v -= b * np.dot(v, b * weights)
        norm = np.dot(v, v * weights)
        if norm < 1e-14:
            continue
        v /= np.sqrt(norm)
        basis.append(v)
    return np.array(basis)
