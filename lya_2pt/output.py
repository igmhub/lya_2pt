import fitsio


def write_healpix_output(output, healpix_id, config, settings, blinding):
    """Write computation output for the main healpix

    Arguments
    ---------
    config: configparser.SectionProxy
    Configuration options

    file: str
    Name of the read file, used to construct the output file

    output: ?
    ?
    """
    output_directory = config["output"].get("output directory")
    filename = output_directory + f"/correlation-{healpix_id}.fits.gz"

    # save data
    results = fitsio.FITS(filename, 'rw', clobber=True)
    header = [{
        'name': 'R_PAR_MIN',
        'value': settings.getfloat('rp_min'),
        'comment': 'Minimum r-parallel [h^-1 Mpc]'
    }, {
        'name': 'R_PAR_MAX',
        'value': settings.getfloat('rp_max'),
        'comment': 'Maximum r-parallel [h^-1 Mpc]'
    }, {
        'name': 'R_TRANS_MAX',
        'value': settings.getfloat('rt_max'),
        'comment': 'Maximum r-transverse [h^-1 Mpc]'
    }, {
        'name': 'NUM_BINS_R_PAR',
        'value': settings.getint('num_bins_rp'),
        'comment': 'Number of bins in r-parallel'
    }, {
        'name': 'NUM_BINS_R_TRANS',
        'value': settings.getint('num_bins_rt'),
        'comment': 'Number of bins in r-transverse'
    }, {
        'name': 'Z_MIN',
        'value': settings.getfloat('z_min'),
        'comment': 'Minimum redshift of pairs'
    }, {
        'name': 'Z_MAX',
        'value': settings.getfloat('z_max'),
        'comment': 'Maximum redshift of pairs'
    }, {
        'name': 'NSIDE',
        'value': settings.getint('nside'),
        'comment': 'Healpix nside'
    }, {
        'name': 'OMEGA_M',
        'value': config['cosmology'].getfloat('Omega_m'),
        'comment': 'Omega_matter(z=0) of fiducial LambdaCDM cosmology'
    }, {
        'name': "BLINDING",
        'value': blinding,
        'comment': 'String specifying the blinding strategy'
    }
    ]
    results.write(
        [output[2], output[3], output[4], output[5]],
        names=['R_PAR', 'R_TRANS', 'Z', 'NUM_PAIRS'],
        comment=['R-parallel', 'R-transverse', 'Redshift', 'Number of pairs'],
        units=['h^-1 Mpc', 'h^-1 Mpc', '', ''],
        header=header,
        extname='ATTRIBUTES')

    if config["compute"].getboolean('compute correlation'):
        header2 = [{
            'name': 'HEALPIX_ID',
            'value': healpix_id,
            'comment': 'Healpix id'
        }]
        correlation_name = "CORRELATION"
        if blinding != "none":
            correlation_name += "_BLIND"
        results.write([output[0], output[1]],
                      names=[correlation_name, "WEIGHT_SUM"],
                      comment=['unnormalized correlation', 'Sum of weight'],
                      header=header2,
                      extname='CORRELATION')

    # TODO: add other modes
    results.close()
