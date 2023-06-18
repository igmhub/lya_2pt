# lya_2pt

[![Pytest](https://github.com/igmhub/lya_2pt/actions/workflows/python_package.yml/badge.svg)](https://github.com/igmhub/lya_2pt/actions/workflows/python_package.yml)
[![codecov](https://codecov.io/gh/igmhub/lya_2pt/branch/main/graph/badge.svg?token=KNVLT9XERN)](https://codecov.io/gh/igmhub/lya_2pt)

Package for computing 3D correlation functions from the Lyman-alpha forest and associated tracers.
This package is still under development. 

Currently available functionality:
- Auto-correlation function
- Distortion matrix for auto-correlation

Remaining functionality to implement:
- Support for cross-correlations
- Metal matrices
- Wick covariance

## Installation
First, create a clean environment:
```
conda create -n my_env python==version gitpython
conda activate my_env
```

If you plan to use the MPI parallelized version, the next step is to install mpi4py. If you are at NERSC use this command (see [NERSC documentation](https://docs.nersc.gov/development/languages/python/parallel-python/#mpi4py-in-your-custom-conda-environment)):
```
MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
```
If not at NERSC, follow the instructions in the [mpi4py documentation](https://mpi4py.readthedocs.io/en/stable/install.html).

Finally, clone and install lya_2pt:
```
git clone https://github.com/igmhub/lya_2pt.git
cd lya_2pt
pip install -e .
```

## Usage
You can run lya_2pt using a input configuration file with:
```
lya-2pt -i path/to/config.ini
```
See [this example](https://github.com/igmhub/lya_2pt/blob/main/examples/lyaxlya_cf.ini) for how to setup the configuration file.

For running the MPI parralelized version at NERSC, use:
```
srun lya-2pt-mpi -i path/to/config.ini
```

If you want to export the computation products separately from computing them, use:
```
lya-2pt-export -i path/to/config.ini
```

## Credits
This package is based in part on [picca](https://github.com/igmhub/picca).
