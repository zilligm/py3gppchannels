import numpy as np
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

sys.path.insert(1, '/hexalattice/hexalattice')
import hexalattice

class BaseStation():
    def __init__(self, pos_x: float = 0, pos_y: float = 0, height: float = 10, tx_power_dB: float = 20):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.height = height
        self.tx_power_dB = tx_power_dB


    def grid_signal_power(self, grid:int):
        pass

class Grid():
    def __init__(self, x_length: float = 100, y_length: float = 100, Nx: int = 100, Ny: int = 100 ):
        coord_x, coord_y = np.meshgrid(np.arange(Nx), np.arange(Nx), sparse=False, indexing='xy')
        coord_x, coord_y = x_length*coord_x, y_length*coord_y


if __name__ == "__main__":
    # Main function includes multiple examples

    BS = BaseStation()