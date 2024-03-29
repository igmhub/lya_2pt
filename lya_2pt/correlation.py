import sys
import numpy as np
from numba import njit

import lya_2pt.global_data as globals
from lya_2pt.tracer_utils import get_angle


def compute_xi(healpix_id):
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

    xi_grid = np.zeros(total_size)
    weights_grid = np.zeros(total_size)
    rp_grid = np.zeros(total_size)
    rt_grid = np.zeros(total_size)
    z_grid = np.zeros(total_size)
    num_pairs_grid = np.zeros(total_size, dtype=np.int32)

    for tracer1 in globals.tracers1[healpix_id]:
        with globals.lock:
            xicounter = round(globals.counter.value * 100. / globals.num_tracers, 2)
            if (globals.counter.value % 1000 == 0):
                print(("computing xi: {}%").format(xicounter))
                sys.stdout.flush()
            globals.counter.value += 1

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

            compute_xi_pair(
                tracer1.deltas, tracer1.weights, tracer1.z, tracer1.dist_c, tracer1.dist_m,
                tracer2.deltas, tracer2.weights, tracer2.z, tracer2.dist_c, tracer2.dist_m,
                angle, xi_grid, weights_grid, rp_grid, rt_grid, z_grid, num_pairs_grid
                )

    # Normalize correlation and average coordinate grids
    w = weights_grid > 0
    xi_grid[w] /= weights_grid[w]
    rp_grid[w] /= weights_grid[w]
    rt_grid[w] /= weights_grid[w]
    z_grid[w] /= weights_grid[w]

    return healpix_id, (xi_grid, weights_grid, rp_grid, rt_grid, z_grid, num_pairs_grid)


@njit
def compute_xi_pair(
        deltas1, weights1, z1, dist_c1, dist_m1,
        deltas2, weights2, z2, dist_c2, dist_m2, angle,
        xi_grid, weights_grid, rp_grid, rt_grid, z_grid, num_pairs_grid
):
    sin_angle = np.sin(angle / 2)
    cos_angle = np.cos(angle / 2)

    for i in range(deltas1.size):
        if weights1[i] == 0:
            continue
        for j in range(deltas2.size):
            if weights2[j] == 0:
                continue

            # Comoving separation between the two pixels
            rp = (dist_c1[i] - dist_c2[j]) * cos_angle
            rt = (dist_m1[i] + dist_m2[j]) * sin_angle
            if globals.auto_flag:
                rp = np.abs(rp)

            # Skip if pixel pair is too far apart
            if (rp < globals.rp_min) or (rp >= globals.rp_max) or (rt >= globals.rt_max):
                continue

            # Compute bin in the correlation function to asign the pixel pair to
            bins_rp = np.floor((rp - globals.rp_min) / (globals.rp_max - globals.rp_min)
                               * globals.num_bins_rp)
            bins_rt = np.floor(rt / globals.rt_max * globals.num_bins_rt)
            bins = int(bins_rt + globals.num_bins_rt * bins_rp)

            # Compute and write correlation and associated quantities
            weight12 = weights1[i] * weights2[j]
            xi_grid[bins] += deltas1[i] * deltas2[j] * weight12
            weights_grid[bins] += weight12
            rp_grid[bins] += rp * weight12
            rt_grid[bins] += rt * weight12
            z_grid[bins] += (z1[i] + z2[j]) / 2 * weight12
            num_pairs_grid[bins] += 1
