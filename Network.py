import matplotlib.pyplot as plt
import numpy as np
import itertools
import hexalattice


class BaseStation:
    bs_id = itertools.count()
    sector_id = itertools.count()

    class Sector:
        sector_id = itertools.count()

        def __init__(self, center_orientation: float = 0, frequency: int = 3500):
            self.ID = next(BaseStation.sector_id)
            self.center_orientation = center_orientation
            self.frequency = frequency

    def __init__(self, pos_x: float = 0, pos_y: float = 0, height: float = 10, number_of_sectors: int = 1,
                 tx_power_dB: float = 20, rotation: float = 0):
        self.ID = next(BaseStation.bs_id)
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.height = height
        self.tx_power_dB = tx_power_dB
        self.sector = []
        for sec in range(number_of_sectors):
            center_orientation = rotation + sec * (number_of_sectors / 360) + (number_of_sectors / 360) / 2
            self.sector.append(self.Sector(center_orientation=center_orientation))

    def grid_signal_power(self, grid):
        signal_power = np.zeros([grid.Nx,grid.Ny])
        for idx_x in range(grid.Nx):
            for idx_y in range(grid.Ny):
                dist = ((grid.coord_x[idx_x,idx_y] - self.pos_x)**2 + (grid.coord_y[idx_x,idx_y] - self.pos_y)**2 )**0.5
                if dist == 0:
                    sigpower = self.tx_power_dB
                else:
                    sigpower = 10 * np.log10(10 ** (self.tx_power_dB / 10) * 1/(dist**2))   # Implement PATHLOSS formula
                signal_power[idx_x][idx_y] = sigpower

        return signal_power

class UserEquipment:
    pos_x = float
    pos_y = float
    height = float
    location = ''
    los = {}
    id_iter = itertools.count()

    def __init__(self, pos_x: float = 0, pos_y: float = 0, height: float = 1.5, location: str = 'Outdoor',
                 los: str = 'LOS', tx_power_dB: float = 20, noise_floor: float = -125):
        self.ID = next(UserEquipment.id_iter)
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.height = height
        self.location = location
        self.los = los
        self.tx_power_dB = tx_power_dB
        self.serving_cell = None
        self.neighbor_cells = []
        self.noise_floor = noise_floor # dB
        self.serving_cell_power = 0
        self.interference_power = 0
        self.sinr = 0

    def attach(self, base_stations: list[BaseStation]):
        rsrp = np.zeros(len(base_stations))
        for bs_idx, bs in enumerate(base_stations):
            dist = ((bs.pos_x - self.pos_x)**2 + (bs.pos_y - self.pos_y)**2)**0.5
            if dist == 0:
                rsrp[bs_idx] = bs.tx_power_dB
            else:
                rsrp[bs_idx] = 10 * np.log10(10 ** (bs.tx_power_dB / 10) * 1 / (dist ** 2)) # Implement PATHLOSS formula
        # Find serving cell
        serving_idx = np.argmax(rsrp)
        self.serving_cell = base_stations[serving_idx].ID

    def compute_sinr(self, base_stations: list[BaseStation]):
        self.serving_cell_power = 0
        self.interference_power = 10 ** (self.noise_floor / 10)
        for bs_idx, bs in enumerate(base_stations):
            dist = ((bs.pos_x - self.pos_x)**2 + (bs.pos_y - self.pos_y)**2)**0.5
            if dist == 0:
                power = 10 ** (bs.tx_power_dB / 10)
            else:
                power = 10 ** (bs.tx_power_dB / 10) * 1 / (dist ** 2)                       # Implement PATHLOSS formula

            if base_stations[bs_idx].ID == self.serving_cell:
                self.serving_cell_power = power
            else:
                self.interference_power = self.interference_power + power
        self.sinr = 10 * np.log10(self.serving_cell_power/self.interference_power)

    # TODO: Handover method: select new cell, detach from current cell, attach to new cell

class Network:
    def __init__(self, scenario: str = 'UMa'):
        self.scenario = scenario

        if self.scenario == 'UMi':
            self.number_of_BSs = 19
            self.number_of_sectors = 3
            self.ISD = 200                                      # meters
            self.BS_height = 10                                 # meters
            self.UE_location = ['Outdoor', 'Indoor']
            self.UE_los = ['LOS', 'NLOS']
            self.UE_height = 1.5                                # meters    # Todo: Check TR 36.873
            self.UE_outdoor_ratio = 0.80                        # 0..1
            self.UE_mobility = 3                                # km/h
            self.min_BS_UE_distance = 10                        # meters
            self.UE_distribution = 'Uniform'

        if self.scenario == 'UMa':
            self.number_of_BSs = 19
            self.number_of_sectors = 3
            self.ISD = 500                                      # meters
            self.BS_height = 35                                 # meters
            self.UE_location = ['Outdoor', 'Indoor']
            self.UE_los = ['LOS', 'NLOS']
            self.UE_height = 1.5                                # meters    # Todo: Check TR 36.873
            self.UE_outdoor_ratio = 0.80                        # 0..1
            self.UE_mobility = 3                                # km/h
            self.min_BS_UE_distance = 35                        # meters
            self.UE_distribution = 'Uniform'

        if self.scenario == 'RMa':
            self.number_of_BSs = 19
            self.number_of_sectors = 3
            self.ISD = 1732 # or 5000                           # meters
            self.BS_height = 35                                 # meters
            self.UE_location = ['Outdoor', 'Indoor']
            self.UE_los = ['LOS', 'NLOS']
            self.UE_height = 1.5                                # meters    # Todo: Check TR 36.873
            self.UE_outdoor_ratio = 0.80                        # 0..1
            self.UE_mobility = 3                                # km/h
            self.min_BS_UE_distance = 35                        # meters
            self.UE_distribution = 'Uniform'

            # Optional Parametes
            self.average_building_heigh = 5.0                   # meters [5..50]
            self.average_street_width = 20.0                    # meters [5..50]

        # TODO: Implement other scenarios:  Indoor Factory (InF) - (InF-SL, InF-DL, InF-SH, InF-DH, InF-HH)
        #                                   Indoor Office

    def PathLoss(self, bs: BaseStation, ue: UserEquipment):
        dist_2D = ((bs.pos_x - ue.pos_x) ** 2 + (bs.pos_y - ue.pos_y) ** 2) ** 0.5
        dist_3D = (dist_2D ** 2 + (bs.height - ue.height) ** 2) ** 0.5
        fc = 3.5            # in GHz            # TODO: get frequency from sector
        c = 300000000       # meters/s

        if self.scenario == 'UMi':
            if (1.5 <= ue.height) and (ue.height <= 22.5):
                raise ("UE height outside the pathloss fomula's applicability range")
            if not (bs.height == 10):
                raise ("BS is not the default value")  # Todo: need to check for correction formulas

            # Breakpoint Distance
            h_e = 1.0  # meter
            d_bp = 4 * (bs.height - h_e) * (ue.height - h_e) * fc / c

            # Pathloss computation for LOS
            if ue.los == 'LOS':
                # Compute PL_UMi-LOS
                if (10 <= dist_2D) and (dist_2D <= d_bp):
                    pathloss = 32.4 + 21 * np.log10(dist_3D) + 20 * np.log10(fc)
                    sigma_sf = 4.0
                elif (d_bp <= dist_2D) and (dist_2D <= 5000):
                    pathloss = 32.4 + 40 * np.log10(dist_3D) + 20 * np.log10(fc) \
                               - 9.5 * np.log10(d_bp ** 2 + (bs.height - ue.height) ** 2)
                    sigma_sf = 4.0
                else:
                    raise ('Invalid range for UE-BS distance')

            # Pathloss computation for NLOS
            if ue.los == 'NLOS':
                # Compute PL_UMi-LOS
                if (10 <= dist_2D) and (dist_2D <= d_bp):
                    PL_UMi_LOS = 32.4 + 21 * np.log10(dist_3D) + 20 * np.log10(fc)
                elif (d_bp <= dist_2D) and (dist_2D <= 5000):
                    PL_UMi_LOS = 32.4 + 40 * np.log10(dist_3D) + 20 * np.log10(fc) \
                               - 9.5 * np.log10(d_bp ** 2 + (bs.height - ue.height) ** 2)
                else:
                    raise ('Invalid range for UE-BS distance')

                # Compute PL_UMi-NLOS
                PL_UMi_NLOS = 35.3 * np.log10(dist_3D) + 22.4 + 21.3 * np.log10(fc) - 0.3 * np.log10(ue.height - 1.5)

                pathloss = max(PL_UMi_LOS, PL_UMi_NLOS)
                sigma_sf = 7.82

                # Optional
                # pathloss = 32.4 + 20 * np.log10(fc) + 31.9 * np.log10(dist_3D)
                # sigma_sf = 8.2

        if self.scenario == 'UMa':
            if (1.5 <= ue.height) and (ue.height <= 22.5):
                raise ("UE height outside the pathloss fomula's applicability range")
            if not (bs.height == 25):
                raise ("BS is not the default value")  # Todo: need to check for correction formulas

            # Breakpoint Distance
            if ue.height < 13:
                C = 0
            elif (13 <= ue.height) and (ue.height <= 23):
                if dist_2D <= 18:
                    g = 0
                else:
                    g = (5 / 4) * ((dist_2D / 100) ** 2) * np.exp(-dist_2D / 150)
                C = (((ue.height - 13) / 10) ** 1.5) * g

            if np.random.random() < 1/(1+C):
                h_e = 1
            else:
                h_e = np.random.choice(np.arange(12, ue.height - 1.5, 3))

            d_bp = 2 * np.pi * bs.height * ue.height * fc / c

            # Pathloss computation for LOS
            if ue.los == 'LOS':
                # Compute PL_UMa-LOS
                if (10 <= dist_2D) and (dist_2D <= d_bp):
                    pathloss = 28.0 + 22 * np.log10(dist_3D) + 20 * np.log10(fc)
                    sigma_sf = 4
                elif (d_bp <= dist_2D) and (dist_2D <= 5000):
                    pathloss = 28.0 + 40 * np.log10(dist_3D) + 20 * np.log10(fc) \
                               - 9 * np.log10(d_bp**2 + (bs.height - ue.height)**2)
                    sigma_sf = 4
                else:
                    raise ('Invalid range for UE-BS distance')

            # Pathloss computation for NLOS
            if ue.los == 'NLOS':
                # Compute PL_UMa-LOS
                if (10 <= dist_2D) and (dist_2D <= d_bp):
                    PL_UMa_LOS = 28.0 + 22 * np.log10(dist_3D) + 20 * np.log10(fc)
                elif (d_bp <= dist_2D) and (dist_2D <= 5000):
                    PL_UMa_LOS = 28.0 + 40 * np.log10(dist_3D) + 20 * np.log10(fc) \
                               - 9 * np.log10(d_bp**2 + (bs.height - ue.height)**2)
                else:
                    raise ('Invalid range for UE-BS distance')

                # Compute PL_UMa-NLOS
                PL_UMa_NLOS = 13.54 + 39.08 * np.log10(dist_3D) + 20 * np.log10(fc) - 0.6 * np.log10(ue.height - 1.5)

                pathloss = max(PL_UMa_LOS, PL_UMa_NLOS)
                sigma_sf = 6

                # Optional
                # pathloss = 32.4 + 20 * np.log10(fc) + 30 * np.log10(dist_3D)
                # sigma_sf = 7.8

        if self.scenario == 'RMa':
            if not(ue.height == 1.5):
                raise ("UE height is not the default value")    # Todo: need to check for correction formulas
            if not (bs.height == 35):
                raise ("BS height is not the default value")    # Todo: need to check for correction formulas

            d_bp = 2 * np.pi * bs.height * ue.height * fc / c

            if ue.los == 'LOS':
                # Compute PL_RMa-LOS
                if (10 <= dist_2D) and (dist_2D <= d_bp):
                    pathloss = 20 * np.log10(40 * np.pi * dist_3D * fc / 3) \
                               + min(0.03 * (self.average_building_heigh ** 1.72), 10) * np.log10(dist_3D) \
                               - min(0.044 * (self.average_building_heigh ** 1.72), 14.77) \
                               + 0.002 * np.log10(self.average_building_heigh) * dist_3D
                    sigma_sf = 4
                elif (d_bp <= dist_2D) and (dist_2D <= 5000):
                    pathloss = 20 * np.log10(40 * np.pi * dist_3D * fc / 3) \
                               + min(0.03 * (self.average_building_heigh ** 1.72), 10) * np.log10(dist_3D) \
                               - min(0.044 * (self.average_building_heigh ** 1.72), 14.77) \
                               + 0.002 * np.log10(self.average_building_heigh) * dist_3D \
                               + 40 * np.log10(dist_3D / d_bp)
                    sigma_sf = 6
                else:
                    raise('Invalid range for UE-BS distance')
            if ue.los == 'NLOS':
                # Compute PL_RMa-LOS
                if (10 <= dist_2D) and (dist_2D <= d_bp):
                    PL_RMa_LOS = 20 * np.log10(40 * np.pi * dist_3D * fc / 3) \
                               + min(0.03 * (self.average_building_heigh ** 1.72), 10) * np.log10(dist_3D) \
                               - min(0.044 * (self.average_building_heigh ** 1.72), 14.77) \
                               + 0.002 * np.log10(self.average_building_heigh) * dist_3D
                elif (d_bp <= dist_2D) and (dist_2D <= 5000):
                    PL_RMa_LOS = 20 * np.log10(40 * np.pi * dist_3D * fc / 3) \
                               + min(0.03 * (self.average_building_heigh ** 1.72), 10) * np.log10(dist_3D) \
                               - min(0.044 * (self.average_building_heigh ** 1.72), 14.77) \
                               + 0.002 * np.log10(self.average_building_heigh) * dist_3D \
                               + 40 * np.log10(dist_3D / d_bp)
                else:
                    raise ('Invalid range for UE-BS distance')

                # Compute PL_RMa-NLOS
                PL_RMa_NLOS = 161.04 - 7.1 * np.log10(self.average_street_width) \
                              + 7.5 * np.log10(self.average_building_heigh) \
                              - (24.37 - 3.7 * ((self.average_building_heigh/bs.height)**2)) * np.log10(bs.height) \
                              + (43.42 - 3.1 * np.log10(bs.height)) * (np.log10(dist_3D) - 3) \
                              + 20 * np.log10(fc) \
                              - (3.2 * (np.log10(11.75 * ue.height)) ** 2 - 4.97)

                pathloss = max(PL_RMa_LOS, PL_RMa_NLOS)
                sigma_sf = 8

        if self.scenario == 'FreeSpace':
            # TODO: Need to check PL and sigma
            pathloss = 20 * np.log10(4 * np.pi * fc * 10 ** 9 / c) + 20 * np.log10(dist_3D)
            sigma_sf = 1

        pathloss = pathloss + np.random.lognormal(sigma=sigma_sf)

        return pathloss


class Grid:
    def __init__(self, x_length: float = 100, y_length: float = 100, Nx: int = 100, Ny: int = 100,
                 center_at_origin: bool = True):
        self.Nx = Nx
        self.Ny = Ny
        self.coord_x, self.coord_y = np.meshgrid(np.arange(Nx+1), np.arange(Ny+1), sparse=False, indexing='xy')
        self.coord_x, self.coord_y = x_length*self.coord_x/Nx, y_length*self.coord_y/Ny

        if center_at_origin:
            self.coord_x = self.coord_x - x_length/2
            self.coord_y = self.coord_y - y_length/2

        self.coordinates = [self.coord_x, self.coord_y]
    # TODO: Plotable grid (draw lines)


if __name__ == "__main__":
    network = Network(scenario='UMa')

    plt.ion()
    fig, (h_ax, ax2) = plt.subplots(1, 2)
    Nx = 200
    Ny = 200

    N = 3
    d = 50
    hex_centers, h_ax = hexalattice.create_hex_grid(nx=N, ny=N, crop_circ=N * d/2, min_diam=d, do_plot=True, h_ax=h_ax)
    tile_centers_x = hex_centers[:, 0]
    tile_centers_y = hex_centers[:, 1]
    x_lim = h_ax.get_xlim()
    y_lim = h_ax.get_ylim()

    UE_x_lim = (min(tile_centers_x) - d / 2, max(tile_centers_x) + d / 2)
    UE_y_lim = (min(tile_centers_y) - d / 2, max(tile_centers_y) + d / 2)
    print(UE_x_lim)

    grid = Grid(x_length=x_lim[1] - x_lim[0], y_length=y_lim[1] - y_lim[0], Nx=Nx, Ny=Ny)


    NBS = len(tile_centers_x)
    BSs = []
    signal_power = np.zeros([Nx, Ny, NBS])
    total_signal_power = np.zeros([Nx, Ny])
    color = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'Greys']
    for i in range(NBS):
        BS = BaseStation(pos_x=tile_centers_x[i], pos_y=tile_centers_y[i], tx_power_dB=36)
        signal_power[:, :, i] = BS.grid_signal_power(grid)
        total_signal_power = total_signal_power + signal_power[:, :, i]
        BSs.append(BS)

    sinr_map = np.zeros([Nx, Ny, NBS])
    noise_floor_ = -120
    for i in range(NBS):
        signal = 10 ** (signal_power[:, :, i] / 10)
        interference = np.ones([Nx, Ny]) * 10 ** (noise_floor_ / 10)
        for j in (np.delete(np.arange(NBS), i)):
            interference = interference + 10 ** (signal_power[:, :, j] / 10)
        sinr_map[:, :, i] = 10 * np.log10(signal / interference)


    chart = h_ax.pcolormesh(grid.coord_x, grid.coord_y,
                            np.clip(sinr_map[:, :, 3], -30, 30),
                            shading='auto', alpha=.8, cmap='turbo')

    color = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']
    NUE = 100
    UEs = []
    sinr = []
    for i in range(NUE):
        pos_xx = np.random.uniform(low=UE_x_lim[0], high=UE_x_lim[1], size=None)
        pos_yy = np.random.uniform(low=UE_y_lim[0], high=UE_y_lim[1], size=None)
        UE = UserEquipment(pos_x=pos_xx, pos_y=pos_yy, tx_power_dB=18)
        UE.attach(BSs)
        UE.compute_sinr(BSs)
        sinr.append(UE.sinr)
        UEs.append(UE)
        h_ax.scatter(UE.pos_x, UE.pos_y, color=color[UE.serving_cell], marker='.')



    # chart = h_ax.pcolormesh(grid.coord_x, grid.coord_y, total_signal_power, shading='auto', alpha=.8, cmap='turbo')

    plt.colorbar(chart, ax=h_ax)

    ax2.hist(sinr)

    plt.show(block=True)