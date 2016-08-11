import os
import logging
from setuptools import setup
from setuptools import find_packages

setup(name='h2ohyperopt',
      version='0.3',
      description='Library for Hyperopt based optimization in H2O Ecosystem',
      url='http://github.com/abhishekmalali/h2o-hyperopt/',
      author='Abhishek Malali',
      author_email='anon@anon.com',
      license='MIT',
      include_package_data=True,
      packages=find_packages(),
      install_requires=['numpy', 'scipy', 'nose', 'hyperopt', 'h2o', 'networkx', 'pandas', 'pymongo==2.2'])
