from setuptools import setup, find_packages

setup(
    author="Guilherme Zilli",
    description="A package for network planning.",
    name="py3gppchannels",
    version="0.1.0",
    packages=find_packages(include=["matplotlib","numpy", "itertools", "hexalattice"]),
)

