import py3gppchannels.py3gppchannels as nw
import numpy as np
import hexalattice
import matplotlib.pyplot as plt


if __name__ == "__main__":

    network = nw.Network(scenario='UMa', free_space=False)

    # plt.ion()
    fig, (h_ax, ax2) = plt.subplots(1, 2)
    Nx = 200
    Ny = 200

    # N = 5  # result in 19 cells
    N = 3 # result in 7 cells
    d = network.ISD
    hex_centers, h_ax = hexalattice.create_hex_grid(nx=N, ny=N, crop_circ=N * d / 2, min_diam=d, do_plot=True,
                                                    h_ax=h_ax, rotate_deg=30)
    # hex_centers = hexalattice.create_hex_grid(nx=N, ny=N, crop_circ=N * d / 2, min_diam=d, do_plot=True, rotate_deg=30)
    tile_centers_x = hex_centers[:, 0]
    tile_centers_y = hex_centers[:, 1]
    # x_lim = h_ax.get_xlim()
    # y_lim = h_ax.get_ylim()

    UE_x_lim = (min(tile_centers_x) - d / 2, max(tile_centers_x) + d / 2)
    UE_y_lim = (min(tile_centers_y) - d / 2, max(tile_centers_y) + d / 2)
    print(UE_x_lim, UE_y_lim)

    # Initialize Base Stations
    NBS = len(tile_centers_x)
    for i in range(NBS):
        network.add_bs(pos_x=tile_centers_x[i], pos_y=tile_centers_y[i])

    # Initialize User Equipment
    NUE = 200
    for i in range(NUE):
        pos_xx = np.random.uniform(low=UE_x_lim[0], high=UE_x_lim[1], size=None)
        pos_yy = np.random.uniform(low=UE_y_lim[0], high=UE_y_lim[1], size=None)
        network.add_ue(pos_x=pos_xx, pos_y=pos_yy)

    # Compute LOS and Pathloss for all UEs/BS/Sector
    network.computeLOS()
    network.computeGeometry()
    network.NetworkPathloss()

    # network.large_scale_parameter_correlation_method_two()

    #
    network.cell_sector_mapping(network.BSs)
    # network.computeLOS()
    # network.computeGeometry()
    # network.NetworkPathloss()
    # # network.computeSmallScaleParameters(network.BSs, network.UEs)
    network.computeRSRP()
    network.UE_attach()
    network.computeSINR()

    color = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']
    marker = ['.', 'x', '+']

    for bs in network.BSs:
        h_ax.scatter(bs.pos_x, bs.pos_y,
                     color=color[int(bs.ID % 7)],
                     marker='^', label=str(bs.ID))
        h_ax.annotate(str(bs.ID), (bs.pos_x + 0.4, bs.pos_y + 3))

    for ue in network.UEs:
        h_ax.scatter(ue.pos_x, ue.pos_y,
                     color=color[int(ue.serving_base_station % 7)],
                     marker=marker[int(ue.serving_sector % 3)], alpha=0.5)

    SinrMatrix = np.clip(network.SINR_Matrix, -60, 60)
    ax2.hist(SinrMatrix, range=[np.floor(min(SinrMatrix)) - 1, np.ceil(max(SinrMatrix)) + 1])
    plt.show(block=True)
