# Contributing to lya_2pt

## Development setup

Use Python 3.10 or later and install an MPI implementation before installing
the project. Create an editable environment with:

```bash
python -m pip install -e '.[dev]'
pre-commit install
```

Run `pytest`, `ruff format --check .`, and `ruff check .` before submitting a
change. Use `pytest tests/test_cf.py -q` for the end-to-end FITS regression
test. Changes to numerical outputs require an explanation and updated reference
fixtures only when the changed result is scientifically intended.

## Pull requests and commits

Use Conventional Commit subjects, such as `fix: handle empty tracer files` or
`docs: clarify MPI installation`. Keep commits focused and do not mix unrelated
refactoring with scientific behavior changes. Pull requests must describe the
change, link relevant issues, list validation commands, and identify changes to
configuration formats or FITS products.

## Documentation and releases

Update Sphinx documentation for user-facing behavior. Build it locally with
`python -m pip install -e '.[docs]'` followed by
`sphinx-build --fail-on-warning -b html docs docs/_build/html`.

Releases use annotated semantic-version tags named `vMAJOR.MINOR.PATCH`. Pushing
a valid tag builds the source and wheel distributions and creates a GitHub
Release; it does not publish to PyPI.
