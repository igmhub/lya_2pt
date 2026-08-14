"""Sphinx configuration for lya_2pt."""

from importlib.metadata import PackageNotFoundError, version

project = "lya_2pt"
copyright = "2026, lya_2pt contributors"
author = "Andrei Cuceu and Ignasi Pérez-Ràfols"

try:
    release = version("lya_2pt")
except PackageNotFoundError:
    release = "unreleased"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
exclude_patterns = ["_build"]
html_theme = "furo"
