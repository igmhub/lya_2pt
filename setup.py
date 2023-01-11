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
version = __version__

setup(name="lya_2pt",
      version=version,
      description=description,
      long_description=long_description,
      long_description_content_type='text/markdown',
      url="https://github.com/igmhub/lya_2pt",
      author="Ignasi Pérez-Ràfols, Andrei Cuceu et al",
      author_email="iprafols@gmail.com",
      packages=find_namespace_packages(),
      install_requires=['numpy', 'scipy', 'healpy', 'fitsio', 'numba', 'setuptools',
                        'mpi4py', 'gitpython'],
      scripts=scripts
      )