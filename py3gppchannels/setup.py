from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'The py3gppchannel library is a Python package that implements the 3GPP standardized radio channel ' \
              'models (3GPP TR 38.901). It provides a comprehensive set of functions for simulating wireless ' \
              'communication systems and evaluating their performance under realistic channel conditions.'
setup(
    author="Guilherme Zilli",
    description=DESCRIPTION,
    name="py3gppchannels",
    version=VERSION,
    url='https://github.com/zilligm/py3gppchannels',
    packages=find_packages(include=["matplotlib", "numpy", "itertools", "hexalattice"]),
    keywords=['wireless', 'wireless communication', 'wireless network', '3gpp', '5G', 'channel model', 'pathloss'],
    setup_requires=['wheel']
)

