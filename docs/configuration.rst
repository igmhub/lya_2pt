Configuration
=============

The INI configuration separates the computation into ``tracer1``, ``settings``,
``cosmology``, ``compute``, ``output``, and ``export`` sections. The example
configuration documents the available auto-correlation settings.

Key settings include the input directory, correlation bin limits, redshift
limits, CPU count, and the output name and directory. Preserve the section
names and option spelling used in ``examples/lyaxlya_cf.ini``; unknown options
are rejected during configuration parsing.

Coordinate grids
----------------

The default ``coordinate-system = rp-rt`` uses the existing ``rp_min``,
``rp_max``, ``rt_max``, ``num_bins_rp``, and ``num_bins_rt`` settings. To bin
in isotropic separation and line-of-sight cosine instead, use ``r-mu`` and
replace those settings with:

.. code-block:: ini

   [settings]
   coordinate-system = r-mu
   r_min = 10
   r_max = 200
   mu_min = 0
   mu_max = 1
   num_bins_r = 50
   num_bins_mu = 50
   num_bins_r_model = 50
   num_bins_mu_model = 50

Here ``r`` is in :math:`h^{-1}\,\mathrm{Mpc}` and ``mu = rp / r`` is
dimensionless. Bins use half-open intervals. The auto-correlation workflow
uses nonnegative ``rp``, so it populates ``0 <= mu < 1``; the kernel also
supports signed ``mu`` in ``[-1, 1]`` for future cross-correlation work.
The r/mu FITS products are marked ``COORDSYS = R_MU`` and contain native
``R``/``MU`` columns; legacy rp/rt products retain their existing schema.
See ``examples/lyaxlya_rmu_cf.ini`` for a complete configuration.

The package currently supports auto-correlations and their distortion matrices.
Cross-correlations and metal-matrix workflows remain under development.
