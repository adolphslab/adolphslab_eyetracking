#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
from setuptools import setup, find_packages


__version__ = '0.0.1'

path_root = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(path_root, "README.rst")) as f:
    long_description = f.read()
    
install_requires = [
    # "numpy",
    # "scipy",
    ]
    
#with open(os.path.join(path_root, "requirements.txt")) as f:
#    install_requires = f.read().strip().split("\n")
    
    
setup(
      name='alabeye',
      version=__version__,
      description='Adolphs Lab, eye tracking tools',
      maintainer='Umit Keles',
      maintainer_email='ukeles@caltech.edu',
      packages=find_packages(),
      install_requires=install_requires,
      python_requires='>=3.7',
      long_description=long_description,
      long_description_content_type='text/x-rst',
)
