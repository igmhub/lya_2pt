#!/usr/bin/env python

import glob
import git

from setuptools import find_namespace_packages, setup
from pathlib import Path

scripts = sorted(glob.glob('bin/*'))

description = (f"Package for computing 3D correlations from Lya forest and associated tracers\n"
               f"commit hash: {git.Repo('.').head.object.hexsha}")
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

exec(open('lya_2pt/_version.py').read())

setup(name="lya_2pt",
      version=__version__,
      description=description,
      long_description=long_description,
      long_description_content_type='text/markdown',
      url="https://github.com/igmhub/lya_2pt",
      author=__author__,
      author_email=__email__,
      packages=find_namespace_packages(),
      install_requires=['numpy', 'scipy', 'astropy', 'healpy', 'fitsio', 'numba', 'setuptools',
                        'gitpython'],
      scripts=scripts
      )