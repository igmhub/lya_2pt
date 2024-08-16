import sys
import numpy as np
from numba import njit

import lya_2pt.global_data as globals
from lya_2pt.tracer_utils import get_angle
from lya_2pt.utils import gen_gamma


import pdb
class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


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

    #init gamma auto and cross arrays
    gamma_grid = np.zeros(total_size)
    delta_gamma_grid = np.zeros(total_size)
    # delta_gamma_p_grid = np.zeros(total_size)

    sigma_v = globals.gamma_z_error
    
    if not globals.cont_polynomial is None:
        # #load polynomials 
        los_ids = globals.cont_polynomial['los_ids']
        zall = globals.cont_polynomial['ztrue']

    # p_ids = globals.cont_polynomial['los_ids']
    # bqs_1 = globals.cont_polynomial['bqs']
    # aqs_1 = globals.cont_polynomial['aqs']
    # bqs_2 = globals.cont_polynomial['bqs_500']
    # aqs_2 = globals.cont_polynomial['aqs_500']
    
    for tracer1 in globals.tracers1[healpix_id]:
        with globals.lock:
            xicounter = round(globals.counter.value * 100. / globals.num_tracers, 2)
            if (globals.counter.value % 1000 == 0):
                print(("computing xi: {}%").format(xicounter))
                sys.stdout.flush()
            globals.counter.value += 1
        


        #calculate rest frame waves and gamma function of tracer 1
        lambda_rest_1 = 1215.67 * (1 + tracer1.z) / (1 + tracer1.z_qso)
        gamma_1 = gen_gamma(lambda_rest_1,sigma_v)

        potential_neighbours = [tracer2 for hp in hp_neighs for tracer2 in globals.tracers2[hp]]
        # for hp in hp_neighs:
        #     if hp not in globals.tracers2:
        #         continue
        #     else:
        #         potential_neighbours.append(globals.tracers2[hp])
        # potential_neighbours = np.concatenate(potential_neighbours)
        #ForkedPdb().set_trace()

        wlam_1 = (lambda_rest_1 > 1000) & (lambda_rest_1 < 1300)

        if not globals.cont_polynomial is None:

            #where the forests are outside of the 1200A upper limit
            id_match_1 = los_ids == tracer1.los_id
            ztrue_1 = zall[id_match_1]
            if len(ztrue_1)==0:
                continue
            lrest_true_1 = 1215.67 * (1 + tracer1.z) / (1 + ztrue_1)
            wlam_1 = (lrest_true_1 > 1040) & (lrest_true_1 < 1200)
                

        neighbours = tracer1.get_neighbours(
            potential_neighbours, globals.auto_flag,
            globals.z_min, globals.z_max,
            globals.rp_max, globals.rt_max
            )

        # #get aq,bq value
        # id_match = p_ids == tracer1.los_id
        # bq_1 = bqs_1[id_match]

        # if bq_1.size==0:
        #     gamma_p1 = np.zeros_like(lambda_rest_1).astype(float)
        #     continue
        # else:
        #     aq_1 = aqs_1[id_match]
        #     aq_2 = aqs_2[id_match]
        #     bq_2 = bqs_2[id_match]
        #     L = (tracer1.log_lambda - tracer1.log_lambda.min()) / (tracer1.log_lambda.max() - tracer1.log_lambda.min())
        #     gamma_p1 = (aq_2 + bq_2 * L) / (aq_1 + ( bq_1 * L ))
                    #calculate rest frame waves and gamma function of tracer 2        
        
        for tracer2 in neighbours:
            
            lambda_rest_2 = 1215.67 * (1 + tracer2.z) / (1 + tracer2.z_qso)
            gamma_2 = gen_gamma(lambda_rest_2,sigma_v)
            
            wlam_2 = (lambda_rest_2> 1000) & (lambda_rest_2 < 1300)
            
            if not globals.cont_polynomial is None:
                #where the forests are outside of the 1200A upper limit
                id_match_2 = los_ids == tracer2.los_id
                ztrue_2 = zall[id_match_2]
                if len(ztrue_2)==0:
                    continue
                lrest_true_2 = 1215.67 * (1 + tracer2.z) / (1 + ztrue_2)
                wlam_2 = (lrest_true_2> 1040) & (lrest_true_2 < 1200)



            # #get aq,bq value
            # id_match = p_ids == tracer2.los_id
            # bq_1 = bqs_1[id_match]
            
            # if bq_1.size==0:
            #     gamma_p2 = np.zeros_like(lambda_rest_2).astype(float)
            #     continue
            # else:
            #     aq_1 = aqs_1[id_match]
            #     aq_2 = aqs_2[id_match]
            #     bq_2 = bqs_2[id_match]
            #     L = (tracer2.log_lambda - tracer2.log_lambda.min()) / (tracer2.log_lambda.max() - tracer2.log_lambda.min())
            #     gamma_p2 = (aq_2 + bq_2 * L) / (aq_1 + ( bq_1 * L ))
                

            angle = get_angle(
                tracer1.x_cart, tracer1.y_cart, tracer1.z_cart, tracer1.ra, tracer1.dec,
                tracer2.x_cart, tracer2.y_cart, tracer2.z_cart, tracer2.ra, tracer2.dec
                )

            compute_xi_pair(
                tracer1.deltas[wlam_1], tracer1.weights[wlam_1], tracer1.z[wlam_1], tracer1.dist_c[wlam_1], tracer1.dist_m[wlam_1],
                tracer2.deltas[wlam_2], tracer2.weights[wlam_2], tracer2.z[wlam_2], tracer2.dist_c[wlam_2], tracer2.dist_m[wlam_2],
                angle, xi_grid, weights_grid, rp_grid, rt_grid, z_grid, num_pairs_grid,
                gamma_grid, delta_gamma_grid, gamma_1[wlam_1], gamma_2[wlam_2]
                )

    # Normalize correlation and average coordinate grids
    w = weights_grid > 0
    xi_grid[w] /= weights_grid[w]
    rp_grid[w] /= weights_grid[w]
    rt_grid[w] /= weights_grid[w]
    z_grid[w] /= weights_grid[w]

    #normalise gamma cross and auto terms
    gamma_grid[w] /= weights_grid[w]
    delta_gamma_grid[w] /= weights_grid[w]
    #delta_gamma_p_grid[w] /= weights_grid[w]

    return healpix_id, (xi_grid, weights_grid, rp_grid, rt_grid, z_grid, num_pairs_grid, gamma_grid, delta_gamma_grid)


@njit
def compute_xi_pair(
        deltas1, weights1, z1, dist_c1, dist_m1,
        deltas2, weights2, z2, dist_c2, dist_m2, angle,
        xi_grid, weights_grid, rp_grid, rt_grid, z_grid, num_pairs_grid, 
        gamma_grid, delta_gamma_grid, gamma_1, gamma_2
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

            #<delta gamma>
            delta_gamma_grid[bins] += deltas1[i] * gamma_2[j] * weight12 
            #<delta gamma 2>
            #delta_gamma_p_grid[bins] += ( (((1+deltas1[i]-gamma_1[i])/gamma_p1[i]) - 1) * (((1+deltas2[j]-gamma_2[j])/gamma_p2#[j]) - 1) * weight12 )
            #<gamma gamma>
            gamma_grid[bins] += gamma_1[i] * gamma_2[j] * weight12 