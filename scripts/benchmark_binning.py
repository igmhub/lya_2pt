#!/usr/bin/env python3
"""Measure warmed-NUMBA correlation and distortion pair-binning kernels."""

import argparse
import statistics
import time

import numpy as np

import lya_2pt.global_data as globals
from lya_2pt.compute_utils import get_pixel_pairs_rmu_auto, get_pixel_pairs_rprt_auto
from lya_2pt.correlation import compute_xi_pair_rmu, compute_xi_pair_rprt


def time_calls(function, arguments, repeats):
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        function(*arguments)
        timings.append(time.perf_counter() - start)
    return statistics.median(timings)


def correlation_arguments(distances1, distances2, num_bins):
    size = num_bins * num_bins
    values1 = np.ones(distances1.shape[0])
    values2 = np.ones(distances2.shape[0])
    return (
        values1,
        values1,
        values1,
        distances1[:, 0],
        distances1[:, 1],
        values2,
        values2,
        values2,
        distances2[:, 0],
        distances2[:, 1],
        0.02,
        np.zeros(size),
        np.zeros(size),
        np.zeros(size),
        np.zeros(size),
        np.zeros(size),
        np.zeros(size, dtype=np.int32),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixels", type=int, default=512, help="Pixels in each synthetic forest")
    parser.add_argument("--repeats", type=int, default=5, help="Timed warmed-kernel repetitions")
    args = parser.parse_args()

    rng = np.random.default_rng(1234)
    distances1 = np.column_stack(
        (rng.uniform(1000, 5000, args.pixels), rng.uniform(1000, 5000, args.pixels))
    )
    distances2 = np.column_stack(
        (rng.uniform(1000, 5000, args.pixels), rng.uniform(1000, 5000, args.pixels))
    )
    num_bins = 50
    globals.auto_flag = True

    globals.rp_min = 0.0
    globals.rp_max = 200.0
    globals.rt_max = 200.0
    globals.num_bins_rp = num_bins
    globals.num_bins_rt = num_bins
    globals.num_bins_rp_model = num_bins
    globals.num_bins_rt_model = num_bins
    rprt_correlation = correlation_arguments(distances1, distances2, num_bins)
    get_pixel_pairs_rprt_auto(distances1, distances2, 0.02)
    compute_xi_pair_rprt(*rprt_correlation)

    globals.r_min = 0.0
    globals.r_max = 200.0
    globals.mu_min = 0.0
    globals.mu_max = 1.0
    globals.num_bins_r = num_bins
    globals.num_bins_mu = num_bins
    globals.num_bins_r_model = num_bins
    globals.num_bins_mu_model = num_bins
    rmu_correlation = correlation_arguments(distances1, distances2, num_bins)
    get_pixel_pairs_rmu_auto(distances1, distances2, 0.02)
    compute_xi_pair_rmu(*rmu_correlation)

    print(f"pixels per forest: {args.pixels}; timed repetitions: {args.repeats}")
    print(
        "rp/rt correlation: "
        f"{time_calls(compute_xi_pair_rprt, rprt_correlation, args.repeats):.6f} s"
    )
    print(
        "rp/rt distortion pairs: "
        f"{time_calls(get_pixel_pairs_rprt_auto, (distances1, distances2, 0.02), args.repeats):.6f} s"
    )
    print(
        f"r/mu correlation: {time_calls(compute_xi_pair_rmu, rmu_correlation, args.repeats):.6f} s"
    )
    print(
        "r/mu distortion pairs: "
        f"{time_calls(get_pixel_pairs_rmu_auto, (distances1, distances2, 0.02), args.repeats):.6f} s"
    )


if __name__ == "__main__":
    main()
