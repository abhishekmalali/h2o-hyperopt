import os
import logging
from setuptools import setup

setup(name='h2ohyperopt',
      version='0.1',
      description='Library for Hyperopt based optmization in H2O Ecosystem',
      url='http://github.com/abhishekmalali/h2o-hyperopt/',
      author='Abhishek Malali',
      author_email='anon@anon.com',
      license='MIT',
      include_package_data=True,
      install_requires=['numpy', 'scipy', 'nose', 'hyperopt', 'h2o'])
