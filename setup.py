#!/usr/bin/env python
from setuptools import find_packages, setup

requirements = []
with open("requirements.txt") as f:
    for line in f:
        stripped = line.split("#")[0].strip()
        if len(stripped) > 0:
            requirements.append(stripped)

setup(
    name="octopuslite",
    version="0.1",
    description="Reader for octopuslite/micromanager files.",
    author="Alan R. Lowe",
    author_email="a.lowe@ucl.ac.uk",
    url="https://github.com/lowe-lab-ucl/octopuslite-reader",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
)
