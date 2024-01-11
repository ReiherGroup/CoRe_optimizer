#!/usr/bin/python3

'''
Setup of core_optimizer Module
'''

# import modules
import sys
from os import path
from setuptools import setup

# check Python version
min_version = (3, 6)
if sys.version_info < min_version:
    error = '''
core_optimizer does not support Python {0}.{1}.
Python {2}.{3} and above is required.
Check your Python version like so:

python3 --version

This may be due to an out-of-date pip.
Make sure you have pip >= 9.0.1.
Upgrade pip like so:

pip install --upgrade pip
'''.format(*(sys.version_info[:2] + min_version))
    sys.exit(error)

# read README.rst and requirements.txt ignoring any commented-out lines
here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open(path.join(here, 'requirements.txt'), encoding='utf-8') as requirements_file:
    requirements = [line for line in requirements_file.read().splitlines() if not line.startswith('#')]

# read version
with open('core_optimizer/_version_.py', encoding='utf-8') as f:
    exec(f.read())

# define setup
setup(
    name = 'core_optimizer',
    version = __version__,
    description = 'Continual Resilient (CoRe) optimizer for PyTorch',
    long_description = readme,
    author = 'Marco Eckhoff, Markus Reiher',
    author_email = 'lifelong_ml@phys.chem.ethz.ch',
    install_requires = requirements,
    license = 'BSD (3-clause)',
    classifiers = [
        'Programming Language :: Python',
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English']
)