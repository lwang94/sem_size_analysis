"""setup file"""
from distutils.core import setup
from .src import config as cf


def list_reqs(fname='requirements.txt'):
    with open(fname) as fd:
        return fd.read().splitlines()

setup(
    name='saemi',
    version=cf.VERSION,
    packages=['src', ],
    install_requires=list_reqs(),
    long_description=open('README.md').read()
)
