Installation
============

``lya_2pt`` supports Python 3.10 and later. An MPI implementation and a
compatible ``mpi4py`` installation are required, including for serial use.
Install MPICH or Open MPI with the platform package manager, clone the source
repository, and then run:

.. code-block:: bash

   git clone https://github.com/igmhub/lya_2pt.git
   cd lya_2pt
   python -m pip install --upgrade pip
   python -m pip install .

Developers should use ``python -m pip install -e '.[dev]'``. At NERSC, install
``mpi4py`` against the system MPI first:

.. code-block:: bash

   MPICC="cc -shared" python -m pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py

See the NERSC and mpi4py documentation for site-specific compiler settings.
