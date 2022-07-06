from __future__ import absolute_import
from __future__ import print_function
from setuptools import find_packages, setup


import io
import re
import os
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup

### Add root folder 
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ) as fh:
        return fh.read()

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Predicts the Red Wine quality.',
    author='Om Prakash',
    license='MIT',
)
