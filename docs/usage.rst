Usage
=====

The standard configuration-driven command is:

.. code-block:: bash

   lya-2pt -i path/to/config.ini

``examples/lyaxlya_cf.ini`` is the annotated starting point for an
auto-correlation run. The package also provides these commands:

* ``lya-2pt-cf`` for auto-correlation command-line arguments.
* ``lya-2pt-dmat`` for distortion-matrix command-line arguments.
* ``lya-2pt-export`` to export existing computation products.
* ``lya-2pt-mpi`` for MPI execution, normally through a scheduler such as
  ``srun lya-2pt-mpi -i path/to/config.ini``.

Use ``--help`` on any command for its accepted options.
