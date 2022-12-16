import sys

import matplotlib.pyplot as plt
import numpy as np

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
                signal_power[idx_x][idx_y] = sigpower

        return signal_power

class Grid():
    def __init__(self, x_length: float = 100, y_length: float = 100, Nx: int = 100, Ny: int = 100,
                 center_at_origin: bool = True):
        self.Nx = Nx
        self.Ny = Ny
        self.coord_x, self.coord_y = np.meshgrid(np.arange(Nx+1), np.arange(Ny+1), sparse=False, indexing='xy')
        self.coord_x, self.coord_y = x_length*self.coord_x/Nx, y_length*self.coord_y/Ny

        if center_at_origin:
            self.coord_x = self.coord_x - x_length/2
            self.coord_y = self.coord_y - y_length/2

        # self.coordinates = [self.coord_x, self.coord_y]
    # TODO: Plotable grid (draw lines)


if __name__ == "__main__":
    plt.ion()
    Nx = 400
    Ny = 400

    N = 3
    d = 50
    hex_centers, h_ax = hexalattice.create_hex_grid(nx=N, ny=N, crop_circ=N * d/2, min_diam=d, do_plot=True, h_ax=None)
    tile_centers_x = hex_centers[:, 0]
    tile_centers_y = hex_centers[:, 1]
    x_lim = h_ax.get_xlim()
    y_lim = h_ax.get_ylim()

    grid = Grid(x_length=x_lim[1] - x_lim[0],y_length=y_lim[1] - y_lim[0], Nx=Nx, Ny=Ny)

    NBS = len(tile_centers_x)
    BSs = []
    signal_power = np.zeros([Nx, Ny, NBS])
    total_signal_power = np.zeros([Nx, Ny])
    for i in range(NBS):
        BS = BaseStation(pos_x=tile_centers_x[i], pos_y=tile_centers_y[i], tx_power_dB=36)
        signal_power[:, :, i] = BS.grid_signal_power(grid)
        total_signal_power = total_signal_power + signal_power[:, :, i]
        BSs.append(BS)

    sinr = np.zeros([Nx, Ny, NBS])
    noise_floor = -120
    for i in range(NBS):
        signal = 10**(signal_power[:, :, i]/10)
        interference = np.ones([Nx, Ny]) * 10**(noise_floor/10)
        for j in (np.delete(np.arange(NBS),i)):
            interference = interference + 10**(signal_power[:, :, j]/10)
        sinr[:, :, i] = 10*np.log10(signal / interference)
        sinr[:, :, i] = np.clip(sinr[:, :, i], -40, 30)


    chart = h_ax.pcolormesh(grid.coord_x, grid.coord_y, total_signal_power, shading='auto', alpha=.8, cmap='turbo')
    # chart = h_ax.pcolormesh(grid.coord_x, grid.coord_y, sinr[:, :, 3], shading='auto', alpha=.8, cmap='turbo')
    plt.colorbar(chart, ax=h_ax)

    plt.show(block=True)