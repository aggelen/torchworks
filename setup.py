#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup

setup(
   name='torchworks',
   version='0.1.0',
   author='Aykut G. Gelen',
   author_email='aykutggelen1@gmail.com',
   packages=['torchworks'],
   # scripts=['bin/script1','bin/script2'],
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