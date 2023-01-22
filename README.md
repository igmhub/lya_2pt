# lya_2pt

Package for computing 3D correlation function from Lya forest and associated tracers.

Work in progress. To be incorporated with picca at a later date.

## Installation
First, create a clean environment:
```
conda create -n my_env python==version gitpython
conda activate my_env
```
The next step is to install mpi4py. If you are at NERSC use this command (see ![NERSC documentation](https://docs.nersc.gov/development/languages/python/parallel-python/#mpi4py-in-your-custom-conda-environment)):
```
MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
```
If not at NERSC, follow the instructions in the ![mpi4py documentation](https://mpi4py.readthedocs.io/en/stable/install.html)
Finally, clone and install lya_2pt:
```
git clone https://github.com/igmhub/lya_2pt.git
cd lya_2pt
pip install -e .
```

## Development plan
![Blue board plan](https://github.com/igmhub/lya_2pt/blob/main/blueboard_plan.jpeg?raw=true)
