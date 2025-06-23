#!/usr/bin/env python

import os

from setuptools import setup


# get version info from __init__.py
def readfile(filename):
    with open(filename) as fp:
        filecontents = fp.read()
    return filecontents


setup(
    name="popstock",
    description="A lightweight package for fast evaluation of the stochastic GW energy density Omega_GW emitted by a population of binary black holes and/or neutron stars.",
    url="https://git.ligo.org/arianna.renzini/popstock",
    author="Arianna Renzini, Jacob Golomb",
    author_email="aria.renzini@gmail.com",
    license="MIT",
    packages=["popstock"],
    package_dir={"popstock": "popstock"},
    scripts=[
        "scripts/population_O3", "scripts/population_O3_fixed_samples", "scripts/population_O3_new_draws"
    ],
    install_requires=[
        "numpy",
        "scipy>=1.8.0",
        "bilby>=1.4",
        "gwpy>=3.0.4",
        "astropy>=5.2",
        "gwpopulation",
        "lalsuite>=7.3",
    ],
    extras_require={
        "dev": [
            "pytest",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
