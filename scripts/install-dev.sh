#!/usr/bin/env bash
# Install lya_2pt and all local development tools into the active environment.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python -m pip install --upgrade pip
MPICC="cc -shared" python -m pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
python -m pip install -e '.[dev,docs]'
pre-commit install
