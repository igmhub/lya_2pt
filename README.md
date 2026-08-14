# lya_2pt

[![Quality](https://github.com/igmhub/lya_2pt/actions/workflows/quality.yml/badge.svg)](https://github.com/igmhub/lya_2pt/actions/workflows/quality.yml)
[![Documentation](https://readthedocs.org/projects/lya-2pt/badge/?version=latest)](https://lya-2pt.readthedocs.io/)

`lya_2pt` computes three-dimensional correlation functions from the Lyman-alpha
forest and associated tracers. It currently supports auto-correlation functions
and their distortion matrices.

## Install

Python 3.10 or later and an MPI implementation are required. Install MPICH or
Open MPI using your platform package manager, then install the package:

```bash
python -m pip install --upgrade pip
python -m pip install .
```

On NERSC, build `mpi4py` against the system MPI before installing the package:

```bash
MPICC="cc -shared" python -m pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
python -m pip install .
```

For development, clone the repository and install the developer tools:

```bash
git clone https://github.com/igmhub/lya_2pt.git
cd lya_2pt
python -m pip install -e '.[dev]'
pre-commit install
```

## Run

Run the configuration-driven workflow with an INI file:

```bash
lya-2pt -i path/to/config.ini
```

The repository includes an annotated example at
[`examples/lyaxlya_cf.ini`](examples/lyaxlya_cf.ini). Other entry points are
`lya-2pt-cf`, `lya-2pt-dmat`, `lya-2pt-export`, and `lya-2pt-mpi`; use
`<command> --help` for their options. MPI jobs are normally launched through
the local scheduler, for example `srun lya-2pt-mpi -i path/to/config.ini`.

## Development

Run the local checks before opening a pull request:

```bash
ruff format --check .
ruff check .
pytest
```

See the [documentation](https://lya-2pt.readthedocs.io/) for configuration and
API guidance, and [CONTRIBUTING.md](CONTRIBUTING.md) for repository workflow
and release conventions.

## Credits

This package is based in part on [picca](https://github.com/igmhub/picca).
