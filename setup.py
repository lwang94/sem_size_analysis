"""setup file"""
from distutils.core import setup
from .src import config as cf


def list_reqs(fname='requirements.txt'):
    with open(fname) as fd:
       return reqs = fd.read().splitlines()
    # deps = [i for i in reqs if i.startswith('--find-links')]
    # reqs = [i for i in reqs if i not in deps]
    # return reqs, deps



setup(
    name='saemi',
    version=cf.VERSION,
    packages=['src', ],
    # dependency_links=list_reqs()[1],
    install_requires=list_reqs(),
    long_description=open('README.md').read()
)
