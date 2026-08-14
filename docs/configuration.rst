Configuration
=============

The INI configuration separates the computation into ``tracer1``, ``settings``,
``cosmology``, ``compute``, ``output``, and ``export`` sections. The example
configuration documents the available auto-correlation settings.

Key settings include the input directory, correlation bin limits, redshift
limits, CPU count, and the output name and directory. Preserve the section
names and option spelling used in ``examples/lyaxlya_cf.ini``; unknown options
are rejected during configuration parsing.

The package currently supports auto-correlations and their distortion matrices.
Cross-correlations and metal-matrix workflows remain under development.
