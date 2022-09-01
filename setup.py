#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup

setup(
   name='torchworks',
   version='0.1.0',
   author='Aykut G. Gelen',
   author_email='aykut.gelen@erzincan.edu.tr',
   packages=['torchworks'],
   scripts=['bin/init_torchworks_experiment'],
   # url='http://pypi.python.org/pypi/PackageName/',
   license='LICENSE.md',
   description='Torchworks is a framework for simplify training, validation, and testing processes with many networks and losses.',
   long_description=open('README.md').read(),
   install_requires=[
   "torch",
   "numpy",
   "tqdm"
   ],
)