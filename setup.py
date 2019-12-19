#! /usr/bin/env python

from setuptools import setup, Extension
import importlib
import os

# copied from kymatio's setup.py: https://github.com/kymatio/kymatio/blob/master/setup.py
sfm_version_spec = importlib.util.spec_from_file_location('sfm_version', 'sfm/version.py')
sfm_version_module = importlib.util.module_from_spec(sfm_version_spec)
sfm_version_spec.loader.exec_module(sfm_version_module)
VERSION = sfm_version_module.version

setup(
    name='sfm',
    version='0.1',
    description='Spatial frequency preferences model',
    license='MIT',
    url='https://github.com/billbrod/spatial-frequency-model',
    author='William F. Broderick',
    author_email='billbrod@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.7'],
    keywords='Visual Information Processing',
    packages=['sfm'],
    install_requires=['numpy>=1.1',
                      'torch>=1.1',
                      'pandas>=0.25'
                      'scipy>=1.0',
                      'matplotlib>=3.1',
                      'pytest',
                      'seaborn>=0.9.0'],
    tests='tests',
)
