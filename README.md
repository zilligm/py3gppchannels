# py3gppchannels

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

The `py3gppchannel` library is a Python package that implements the 3GPP standardized radio channel models (3GPP TR 38.901). 
It provides a comprehensive set of functions for simulating wireless communication systems and evaluating their performance under realistic channel conditions.

**Note that this is an ongoing (and unfinished) project and it is not fully tested or validated.**

## Features

- Supports RMa, UMa, and UMi scenarios (indoor scenarios TBD)
- Customization of model parameters such as frequency, distance, and antenna height
- Computation of LOS, pathloss, and shadow fading
- Calculation of correlated Large Scale Parameters (e.g., delay spread, angular spreads, Rician K factor, and shadow fading)
- Calculation of Small Scale Parameters (e.g., cluster delays, cluster powers, cluster AoAs and AoDs)

## Installation

You can install `py3gppchannel` using pip:
pip install py3gppchannel
