# py3gppchannels

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

The `py3gppchannels` library is a Python package that implements the 3GPP standardized radio channel models (3GPP TR 38.901). 
It provides a comprehensive set of functions for simulating wireless communication systems and evaluating their performance under realistic channel conditions.

**Note that this is an ongoing (and unfinished) project and it is not fully tested or validated.**

## Features

- Supports RMa, UMa, and UMi scenarios (indoor scenarios TBD)
- Customization of model parameters such as frequency, distance, and antenna height
- Computation of LOS, pathloss, and shadow fading
- Calculation of correlated Large Scale Parameters (e.g., delay spread, angular spreads, Rician K factor, and shadow fading)
- Calculation of Small Scale Parameters (e.g., cluster delays, cluster powers, cluster AoAs and AoDs)

## Installation

You can install `py3gppchannels` using pip:

`pip install py3gppchannels`

## Disclaimer
The authors and contributors of this software package (hereinafter referred to as "the Code") make no warranties or guarantees, expressed or implied, regarding the functionality, performance, or fitness for a particular purpose of the Code.

The Code is provided "as is" without any warranty, whether expressed or implied. The authors and contributors shall not be held liable for any damages, claims, or liabilities arising from the use, distribution, or modification of the Code or any derivative works thereof.

Users of the Code assume all risks associated with its use. The authors and contributors shall not be held responsible for any loss or damage incurred by users, including but not limited to data loss, system failures, or any other direct or indirect damages.

By using the Code, you agree to these terms and acknowledge that the authors and contributors shall not be held liable for any issues or consequences arising from its use. If you do not agree with these terms, you should refrain from using the Code.
