from setuptools import setup, find_packages

setup(
    author="Guilherme Zilli",
    description="The py3gppchannel library is a Python package that implements the 3GPP standardized radio channel "
                "models (3GPP TR 38.901). It provides a comprehensive set of functions for simulating wireless "
                "communication systems and evaluating their performance under realistic channel conditions.",
    name="py3gppchannels",
    version="0.0.1",
    packages=find_packages(include=["matplotlib","numpy", "itertools", "hexalattice"]),
    keywords=['wireless', 'wireless communication', 'wireless network', '3gpp', '5G', 'channel model', 'pathloss'],
)

