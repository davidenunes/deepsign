#!/usr/bin/env python

from distutils.core import setup

requirements = [
    'h5py>=2.6.0',
    'numpy>=1.11.1',
    'scipy>=0.18.1',
    'scikit-learn>=0.18.1',
    'tqdm>=4.19.5',
    'bidict>=0.14.2',
    'marisa-trie>=0.7.4',
    'tensorflow>=1.5',
    'matplotlib'
]

links = [
    "git+git://github.com/davidenunes/tensorx@dev#egg=tensorx",
    "git+git://bitbucket.org:davex32/exp.git@master#egg=exp"
]

setup(name='deepsign',
      version='1.0',
      description='research utils',
      author='Davide Nunes',
      author_email='davidelnunes@gmail.com',
      packages=['deepsign'],
      install_requires=requirements,
      depedency_links=links)
