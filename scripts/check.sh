#!/usr/bin/env bash
# Run the repository checks required before submitting a change.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

ruff format --check .
ruff check .
python -m pytest -q
python -m build
twine check dist/*
sphinx-build --fail-on-warning --keep-going -b html docs docs/_build/html
