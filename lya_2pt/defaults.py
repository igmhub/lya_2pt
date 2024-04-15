
cosmo = {
    "use-picca-cosmo": False,
    "omega_m": 0.315,
    "omega_r": 7.963219132297603e-05,
    "hubble-constant": 67.36,
    "use-h-units": True,
    "omega_k": 0.0,
    "w0": -1,
}


settings = {
    "nside": 16,
    "num-cpu": 1,
    "z_min": 0,
    "z_max": 10,
    "rp_min": 0,
    "rp_max": 200,
    "rt_max": 200,
    "num_bins_rp": 50,
    "num_bins_rt": 50,
    "num_bins_rp_model": 50,
    "num_bins_rt_model": 50,
    "rejection_fraction": 0.99,
    "get-old-distortion": True
}

tracer_reader = {
    "tracer-type": "continuous",
    "absorption-line": "LYA",
    "project-deltas": True,
    "projection-order": 1,
    "use-old-projection": True,
    "rebin": 1,
    "redshift-evolution": 2.9,
    "reference-redshift": 2.25,
}

output = {
    "name": "lyaxlya",
}

export = {
    "export-correlation": False,
    "export-distortion": False,
    "smooth-covariance": True,
}
