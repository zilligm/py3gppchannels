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

    def grid_signal_power(self, grid):
        signal_power = np.zeros([grid.Nx,grid.Ny])
        for idx_x in range(grid.Nx):
            for idx_y in range(grid.Ny):
                dist = ((grid.coord_x[idx_x,idx_y] - self.pos_x)**2 + (grid.coord_y[idx_x,idx_y] - self.pos_y)**2 )**0.5
                if dist == 0:
                    sigpower = self.tx_power_dB
                else:
                    sigpower = 10*np.log10(10**(self.tx_power_dB/10) * 1/(dist**2))
                print(f'x:{idx_x}, y:{idx_x}, sig:{sigpower}')
                signal_power[idx_x][idx_y] = sigpower

        return signal_power

class Grid():
    def __init__(self, x_length: float = 100, y_length: float = 100, Nx: int = 100, Ny: int = 100,
                 center_at_origin: bool = True):
        self.Nx = Nx
        self.Ny = Ny
        self.coord_x, self.coord_y = np.meshgrid(np.arange(Nx+1), np.arange(Ny+1), sparse=False, indexing='xy')
        self.coord_x, self.coord_y = x_length*self.coord_x/Nx, y_length*self.coord_y/Ny
        # self.coord_x = self.coord_x.reshape(-1)
        # self.coord_y = self.coord_y.reshape(-1)
        # self.coord_x = self.coord_x.flatten()
        # self.coord_y = self.coord_y.flatten()

        if center_at_origin:
            self.coord_x = self.coord_x - x_length/2
            self.coord_y = self.coord_y - x_length/2

        # self.coordinates = [self.coord_x, self.coord_y]
    # TODO: Plotable grid (draw lines)


if __name__ == "__main__":
    # Main function includes multiple examples
    grid = Grid()
    # print(grid.coord_x.shape, grid.coord_y.shape)
    BS1 = BaseStation()
    BS2 = BaseStation(pos_x=20, pos_y=20)
    signal_power_1 = BS1.grid_signal_power(grid)
    signal_power_2 = BS2.grid_signal_power(grid)

    print((grid.coord_y))
    # print(shape(signal_power_2))
    #
    # print(len(signal_power_1))
    # print(len(signal_power_2))

    signal_power = signal_power_1 + signal_power_2

    # print(signal_power)
    fig, ax = plt.subplots()
    ax.pcolormesh(grid.coord_x, grid.coord_y, signal_power, shading='auto')
    plt.show()