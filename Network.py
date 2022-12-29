import matplotlib.pyplot as plt
import numpy as np
import itertools
import hexalattice


class BaseStation:
    # Auto ID generation
    bs_id = itertools.count()
    sector_id = itertools.count()

    class Sector:
        def __init__(self, bs_id: int, center_orientation: float = 0, sector_width: float = 120,
                     frequency: int = 3.5, tx_power_dB: float = 20):
            """
            Create sectors within a Base Station
            :param bs_id: Base Station ID
            :param center_orientation: orientation of the sector center [degrees] > 0 points toward North
            :param sector_width: width of the sector [degrees]
            :param frequency: operation frequency of the sector [GHz]
            :param tx_power_dB: transmission power [dBm]
            """
            self.ID = next(BaseStation.sector_id)
            self.BS_ID = bs_id
            self.center_orientation = center_orientation
            self.sector_width = sector_width
            self.frequency = frequency
            self.tx_power_dB = tx_power_dB
            self.connected_UEs = []

    class AntennaPanel:
        def __init__(self, sector_id: int,
                     n_panel_col: int = 1, n_panel_row: int = 1,
                     panel_v_spacing: float = 0, panel_h_spacing: float = 0,
                     n_antenna_col: int = 1, n_antenna_row: int = 1,
                     antenna_v_spacing: float = 0, antenna_h_spacing: float = 0,
                     polarization: str = 'single'
                     ):
            """
            Create the antenna panel object. Each antenna panel contains an antenna array.

            :param sector_id: sector ID associated with the antenna panel
            :param n_panel_col: number of columns in the panel array
            :param n_panel_row: number of rows in the panel array
            :param panel_v_spacing: vertical spacing between panels (i.e., space between panels of two consecutive rows
                                    in the same column) measured from the center of the panel
            :param panel_h_spacing: horizontal spacing between panels (i.e., space between panels of two consecutive
                                    columns in the same row) measured from the center of the panel
            :param n_antenna_col: number of antennas columns in the antenna array (i.e., within the panel)
            :param n_antenna_row: number of antennas rows in the antenna array (i.e., within the panel)
            :param antenna_v_spacing: vertical spacing between antennas
            :param antenna_h_spacing: horizontal spacing between antennas
            :param polarization: polarization, either 'single' or 'dual'
            """
            self.sector_id = sector_id
            self.n_panel_col = n_panel_col
            self.n_panel_row = n_panel_row
            self.panel_v_spacing = panel_v_spacing
            self.panel_h_spacing = panel_h_spacing
            self.n_antenna_col = n_antenna_col
            self.n_antenna_row = n_antenna_row
            self.antenna_v_spacing = antenna_v_spacing
            self.antenna_h_spacing = antenna_h_spacing
            self.polarization = polarization

    def __init__(self, pos_x: float = 0, pos_y: float = 0, height: float = 10, number_of_sectors: int = 3,
                 tx_power_dB: float = 20, rotation: float = 0):
        """
        Create a Base Station
        :param pos_x: position of the Base Station in the x axis [meters]
        :param pos_y: position of the Base Station in the y axis [meters]
        :param height: height of the Base Station [meters]
        :param number_of_sectors: number of sectors in the Base Station
        :param tx_power_dB: transmission power [dBm]
        :param rotation: offset to the Base Station' sector orientation
        """
        self.network = network
        self.ID = next(BaseStation.bs_id)
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.height = height
        self.tx_power_dB = tx_power_dB
        self.sector = []
        for sec in range(number_of_sectors):
            center_orientation = rotation + sec * 360 / number_of_sectors
            sector_width = 360 / number_of_sectors
            sector = self.Sector(bs_id=self.ID, center_orientation=center_orientation, sector_width=sector_width,
                                 tx_power_dB=self.tx_power_dB)
            print(f'Base Station {self.ID} - Sector {sector.ID}')
            self.sector.append(sector)


class UserEquipment:
    pos_x = float
    pos_y = float
    height = float
    location = ''
    los = ''
    id_iter = itertools.count()

    def __init__(self, pos_x: float = 0, pos_y: float = 0, height: float = 1.5,
                 location: str = 'Outdoor', los: str = 'LOS', tx_power_dB: float = 20, noise_floor: float = -125):
        """
        Create User Equipment (UE)
        :param pos_x: position of the UE in the x axis [meters]
        :param pos_y: position of the UEt in the y axis [meters]
        :param height: height of the UE [meters]
        :param location: location of the UE ['Indoor','Outdoor']
        :param los: line-of-sight ['LOS','NLOS']
        :param tx_power_dB: transmission power [dBm]
        :param noise_floor: noise floor [dBm]
        """
        self.network = network
        self.ID = next(UserEquipment.id_iter)
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.height = height
        self.location = location
        self.los = los
        self.tx_power_dB = tx_power_dB
        self.serving_sector = None
        self.serving_base_station = None
        self.neighbor_cells = []
        self.noise_floor = noise_floor  # dB


class Network:
    def __init__(self, scenario: str = 'UMa', free_space: bool = False):
        """
        Create the Network
        :param scenario: indicates the scenario:    'UMa' -> Urban Macro
                                                    'UMi' -> Urban Micro
                                                    'RMa' -> Rural Macro
        :param free_space: flag for pathloss model; if True, override the scenario pathloss model (this may be useful 
                           for debugging 
        """
        self.free_space = free_space
        self.scenario = scenario

        self.pathlossMatrix = [[]]
        self.shadowFadingMatrix = [[]]
        self.losMatrix = [[]]
        self.RsrpMatrix = np.array([])
        self.SinrMatrix = np.array([])

        if self.scenario == 'UMi':
            self.layout = 'Hexagonal'
            self.number_of_BSs = 19
            self.number_of_sectors = 3
            self.ISD = 200  # meters
            self.BS_height = 10  # meters
            self.UE_location = ['Outdoor', 'Indoor']
            self.UE_los = ['LOS', 'NLOS']
            self.UE_height = 1.5  # meters    # Todo: Check TR 36.873
            self.UE_indoor_ratio = 0.80  # 0..1
            self.UE_mobility = 3  # km/h
            self.min_BS_UE_distance = 10  # meters
            self.UE_distribution = 'Uniform'

        if self.scenario == 'UMa':
            self.layout = 'Hexagonal'
            self.number_of_BSs = 19
            self.number_of_sectors = 3
            self.ISD = 500  # meters
            self.BS_height = 25  # meters
            self.UE_location = ['Outdoor', 'Indoor']
            self.UE_los = ['LOS', 'NLOS']
            self.UE_height = 1.5  # meters    # Todo: Check TR 36.873
            self.UE_indoor_ratio = 0.80  # 0..1
            self.UE_mobility = 3  # km/h
            self.min_BS_UE_distance = 35  # meters
            self.UE_distribution = 'Uniform'

        if self.scenario == 'RMa':
            self.layout = 'Hexagonal'
            self.number_of_BSs = 19
            self.number_of_sectors = 3
            self.ISD = 1732  # 1732 or 5000                           # meters
            self.BS_height = 35  # meters
            self.UE_location = ['Indoor', 'Car']
            self.UE_los = ['LOS', 'NLOS']
            self.UE_height = 1.5  # meters    # Todo: Check TR 36.873
            self.UE_indoor_ratio = 0.50  # 0..1
            self.UE_mobility = 3  # km/h
            self.min_BS_UE_distance = 35  # meters
            self.UE_distribution = 'Uniform'

            # Optional Parametes
            self.average_building_heigh = 5.0  # meters [5..50]
            self.average_street_width = 20.0  # meters [5..50]

        # TODO: Implement other scenarios:  Indoor Factory (InF) - (InF-SL, InF-DL, InF-SH, InF-DH, InF-HH)
        #                                   Indoor Office/Hotspot (InH)

    def LineOfSight(self, bs: BaseStation, ue: UserEquipment):
        """
        Determine if a given BS and UE pair is in 'LOS' or 'NLOS'
        :param bs: BaseStation object
        :type bs: object BaseStation
        :param ue: UserEquipment object
        :type bs: object UserEquipment
        :return: the line-of-sight condition, i.e.: 'LOS' or 'NLOS'
        """
        dist_2D = ((bs.pos_x - ue.pos_x) ** 2 + (bs.pos_y - ue.pos_y) ** 2) ** 0.5

        # Rural Macro scenario
        Pr_LOS = 1
        if self.scenario == 'RMa':
            if dist_2D <= 10:
                Pr_LOS = 1
            else:
                Pr_LOS = np.exp(-(dist_2D - 10) / 1000)

        # Urban Macro scenario
        if self.scenario == 'UMa':
            if dist_2D <= 18:
                Pr_LOS = 1
            else:
                C = 0
                if ue.height < 13:
                    C = 0
                elif (13 <= ue.height) and (ue.height <= 23):
                    C = (((ue.height - 13) / 10) ** 1.5)

                Pr_LOS = (18 / dist_2D) + np.exp(-dist_2D / 63) * (1 - 18 / dist_2D) \
                         * (1 + C * (5 / 4) * ((dist_2D / 100) ** 3) * np.exp(-dist_2D / 150))

        # Urban Micro scenario
        if self.scenario == 'UMi':
            if dist_2D <= 18:
                Pr_LOS = 1
            else:
                Pr_LOS = (18 / dist_2D) + np.exp(-dist_2D / 36) * (1 - 18 / dist_2D)

        if np.random.random() < Pr_LOS:
            return 'LOS'
        else:
            return 'NLOS'

    def UELocation(self):
        """
        Generates the location of an UE based on the scenario Indoor Ratio parameter
        :return: an UE location, i.e., 'Indoor', 'Outdoor', or 'Car'
        """
        if (self.scenario == 'UMi') or (self.scenario == 'UMa'):
            if np.random.random() < self.UE_indoor_ratio:
                return 'Indoor'
            else:
                return 'Outdoor'
        if self.scenario == 'RMa':
            # Todo: Need to check this
            if np.random.random() < self.UE_indoor_ratio:
                return 'Indoor'
            else:
                return 'Car'

    def Pathloss(self, bs: BaseStation, sec: BaseStation.Sector, ue: UserEquipment):
        """
        Computes the Pathloss between a BS sector and UE pair based on the Network scenario
        :param bs: BaseStation object
        :type bs: object BaseStation
        :param sec: Sector object
        :type sec: object BaseStation.Sector
        :param ue: UserEquipment object
        :type ue: object UserEquipment
        :return: Pathloss [dB]
        """
        dist_2D = ((bs.pos_x - ue.pos_x) ** 2 + (bs.pos_y - ue.pos_y) ** 2) ** 0.5
        dist_3D = (dist_2D ** 2 + (bs.height - ue.height) ** 2) ** 0.5
        fc = sec.frequency
        c = 300000000  # meters/s
        pathloss = 0
        sigma_sf = 0

        # Basic Pathloss
        # If free_space flag is True, compute pathloss according to the Free Space model
        # If free_space flag is False, compute pathloss according to scenario
        if self.free_space:
            pathloss = 20 * np.log10(4 * np.pi * fc * 10 ** 9 / c) + 20 * np.log10(dist_3D)

        else:
            # Pathloss for Rural Macro scenario
            if self.scenario == 'RMa':
                if not (ue.height == 1.5):
                    raise "UE height is not the default value"  # Todo: need to check for correction formulas
                if not (bs.height == 35):
                    raise "BS height is not the default value"  # Todo: need to check for correction formulas

                d_bp = 2 * np.pi * bs.height * ue.height * fc / c

                if ue.los == 'LOS':
                    # Compute PL_RMa-LOS
                    if (10 <= dist_2D) and (dist_2D <= d_bp):
                        pathloss = 20 * np.log10(40 * np.pi * dist_3D * fc / 3) \
                                   + min(0.03 * (self.average_building_heigh ** 1.72), 10) * np.log10(dist_3D) \
                                   - min(0.044 * (self.average_building_heigh ** 1.72), 14.77) \
                                   + 0.002 * np.log10(self.average_building_heigh) * dist_3D
                        sigma_sf = 4
                    elif (d_bp <= dist_2D) and (dist_2D <= 10000):
                        pathloss = 20 * np.log10(40 * np.pi * dist_3D * fc / 3) \
                                   + min(0.03 * (self.average_building_heigh ** 1.72), 10) * np.log10(dist_3D) \
                                   - min(0.044 * (self.average_building_heigh ** 1.72), 14.77) \
                                   + 0.002 * np.log10(self.average_building_heigh) * dist_3D \
                                   + 40 * np.log10(dist_3D / d_bp)
                        sigma_sf = 6
                    else:
                        # # TODO: remove
                        # pathloss = np.inf
                        # sigma_sf = 6
                        raise 'Invalid range for UE-BS distance'

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
                        # TODO: remove
                        PL_RMa_LOS = np.inf
                        sigma_sf = 8
                        # raise ('Invalid range for UE-BS distance')

                    # Compute PL_RMa-NLOS
                    PL_RMa_NLOS = 161.04 - 7.1 * np.log10(self.average_street_width) \
                                  + 7.5 * np.log10(self.average_building_heigh) \
                                  - (24.37 - 3.7 * ((self.average_building_heigh / bs.height) ** 2)) * np.log10(
                        bs.height) \
                                  + (43.42 - 3.1 * np.log10(bs.height)) * (np.log10(dist_3D) - 3) \
                                  + 20 * np.log10(fc) \
                                  - (3.2 * (np.log10(11.75 * ue.height)) ** 2 - 4.97)

                    pathloss = max(PL_RMa_LOS, PL_RMa_NLOS)
                    sigma_sf = 8

            # Pathloss for Urban Macro scenario
            if self.scenario == 'UMa':
                if not ((1.5 <= ue.height) and (ue.height <= 22.5)):
                    raise "UE height outside the pathloss formula's applicability range"
                if not (bs.height == 25):
                    raise "BS is not the default value"  # Todo: need to check for correction formulas

                # Breakpoint Distance
                C = 0
                if ue.height < 13:
                    C = 0
                elif (13 <= ue.height) and (ue.height <= 23):
                    if dist_2D <= 18:
                        g = 0
                    else:
                        g = (5 / 4) * ((dist_2D / 100) ** 3) * np.exp(-dist_2D / 150)
                    C = (((ue.height - 13) / 10) ** 1.5) * g

                if np.random.random() < 1 / (1 + C):
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
                                   - 9 * np.log10(d_bp ** 2 + (bs.height - ue.height) ** 2)
                        sigma_sf = 4
                    else:
                        raise 'Invalid range for UE-BS distance'

                # Pathloss computation for NLOS
                if ue.los == 'NLOS':
                    # Compute PL_UMa-LOS
                    if (10 <= dist_2D) and (dist_2D <= d_bp):
                        PL_UMa_LOS = 28.0 + 22 * np.log10(dist_3D) + 20 * np.log10(fc)
                    elif (d_bp <= dist_2D) and (dist_2D <= 5000):
                        PL_UMa_LOS = 28.0 + 40 * np.log10(dist_3D) + 20 * np.log10(fc) \
                                     - 9 * np.log10(d_bp ** 2 + (bs.height - ue.height) ** 2)
                    else:
                        raise 'Invalid range for UE-BS distance'

                    # Compute PL_UMa-NLOS
                    if (dist_3D > 0) and ((ue.height - 1.5) > 0):
                        PL_UMa_NLOS = 13.54 + 39.08 * np.log10(dist_3D) + 20 * np.log10(fc) - 0.6 * np.log10(
                            ue.height - 1.5)
                    else:
                        PL_UMa_NLOS = -np.inf

                    pathloss = max(PL_UMa_LOS, PL_UMa_NLOS)
                    sigma_sf = 6

                    # Optional
                    # pathloss = 32.4 + 20 * np.log10(fc) + 30 * np.log10(dist_3D)
                    # sigma_sf = 7.8

            # Pathloss for Urban Micro scenario
            if self.scenario == 'UMi':
                if not ((1.5 <= ue.height) and (ue.height <= 22.5)):
                    raise "UE height outside the pathloss formula's applicability range"
                if not (bs.height == 10):
                    raise "BS is not the default value"  # Todo: need to check for correction formulas

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
                        raise 'Invalid range for UE-BS distance'

                # Pathloss computation for NLOS
                if ue.los == 'NLOS':

                    # Compute PL_UMi-LOS
                    if (10 <= dist_2D) and (dist_2D <= d_bp):
                        PL_UMi_LOS = 32.4 + 21 * np.log10(dist_3D) + 20 * np.log10(fc)
                    elif (d_bp <= dist_2D) and (dist_2D <= 5000):
                        PL_UMi_LOS = 32.4 + 40 * np.log10(dist_3D) + 20 * np.log10(fc) \
                                     - 9.5 * np.log10(d_bp ** 2 + (bs.height - ue.height) ** 2)
                    else:
                        raise 'Invalid range for UE-BS distance'

                    # Compute PL_UMi-NLOS
                    try:
                        PL_UMi_NLOS = 35.3 * np.log10(dist_3D) \
                                      + 22.4 + 21.3 * np.log10(fc) \
                                      - 0.3 * np.log10(ue.height - 1.5)
                        pathloss = max(PL_UMi_LOS, PL_UMi_NLOS)
                        sigma_sf = 7.82
                    except:
                        # Optional
                        pathloss = 32.4 + 20 * np.log10(fc) + 31.9 * np.log10(dist_3D)
                        sigma_sf = 8.2

            # Additional Pathloss terms for Indoor UEs
            if ue.location == 'Indoor':
                # Todo: implement the penetration loss model - See TR 38.901 - 7.4.3
                PL_tw = 0
                PL_in = 0
                sigma_p_sq = 0
                pathloss = pathloss + PL_tw + PL_in + np.random.normal(scale=np.sqrt(sigma_p_sq))

            # Additional Pathloss terms for in Car UEs
            if ue.location == 'Car':
                mu = 9.0  # 20 for metalized window
                sigma_p = 5.0
                pathloss = pathloss + np.random.normal(loc=mu, scale=sigma_p)

            # Final Pathloss with shadow fading
            pathloss = pathloss + np.random.lognormal(sigma=sigma_sf)

        # Sectorization
        # Todo: incorporate AoA/AoD
        # For now, I'm doing simple sectorization
        ue_centered_position = np.array([(ue.pos_x - bs.pos_x) / dist_2D, (ue.pos_y - bs.pos_y) / dist_2D])
        sector_center = np.array(
            [np.cos(np.deg2rad(sec.center_orientation)), np.sin(np.deg2rad(sec.center_orientation))])
        angle = np.dot(ue_centered_position, sector_center) / (
                np.linalg.norm(ue_centered_position) * np.linalg.norm(sector_center))
        angle = np.arccos(angle)

        if angle <= (np.deg2rad(sec.sector_width / 2)):
            pathloss = pathloss
        else:
            pathloss = np.inf

        return pathloss, sigma_sf

    def NetworkPathlossAndLos(self, BS_list: list[BaseStation], UE_list: list[UserEquipment]):
        """
        Computes the Pathloss and Line of Sight parameters for combinations of BSs and UEs
        :param BS_list: list of BSs
        :param UE_list: lsit of UEs
        :return: update Network attributes losMatrix and pathlossMatrix
        """
        nBS = len(BS_list)
        nUE = len(UE_list)
        nSectors = nBS * self.number_of_sectors
        self.pathlossMatrix = np.zeros((nUE, nSectors))
        self.shadowFadingMatrix = np.zeros((nUE, nSectors))
        self.losMatrix = [[None] * nSectors] * nUE
        for eu_ind, ue in enumerate(UE_list):
            for bs_ind, bs in enumerate(BS_list):
                # LOS/NLOS is determined for each BS and UE pair
                ue.los = self.LineOfSight(bs=bs, ue=ue)
                self.losMatrix[ue.ID][bs.ID] = ue.los
                for sec_ind, sec in enumerate(bs.sector):
                    # Pathloss is determined for each Sector and UE pair
                    pathloss, sigma_sf = self.Pathloss(bs=bs, sec=sec, ue=ue)
                    self.pathlossMatrix[ue.ID][sec.ID] = pathloss
                    self.shadowFadingMatrix[ue.ID][sec.ID] = sigma_sf

        # with np.printoptions(precision=1, suppress=True):
        #     print(self.pathlossMatrix)
        # print(self.losMatrix)

    def generateLargeScaleParams_link(self, bs: BaseStation, sec: BaseStation.Sector, ue: UserEquipment):

        # Large Scale Parameters (LSP) for different BS-UE links are uncorrelated, but the LSPs for links from co-sited
        # sectors to a UE are the same. In addition, LSPs for the links of UEs on different floors are uncorrelated.

        fc = 3.5    # GHz  # Todo: figure if get from sector since LSPs should be the same for all sectors within a BS

        if self.scenario == 'UMi':
            # Frequency correction - see NOTE 7 from Table 7.5-6
            if fc < 2.0:
                fc = 2.0

            if self.losMatrix[ue.ID][bs.ID] == 'LOS':
                # Delay Spread (DS)
                mu_lg_DS = -0.24 * np.log10(1+fc) - 7.14
                sigma_lg_DS = 0.38

                # AOD Spread (ASD)
                mu_lg_ASD = -0.05 * np.log10(1 + fc) + 1.21
                sigma_lg_ASD = 0.41

                # AOA Spread (ASA)
                mu_lg_ASA = -0.08 * np.log10(1 + fc) + 1.73
                sigma_lg_ASA = 0.014 * np.log10(1 + fc) + 0.28

                # ZOA Spread (ZSA)
                mu_lg_ZSA = -0.1 * np.log10(1 + fc) + 0.73
                sigma_lg_ZSA = -0.04 * np.log10(1 + fc) + 0.34

                # Shadow Fading (SF) [dB]
                # Todo: See Table 7.4.1-1

                # K Factor (K) [dB]
                mu_K = 9
                sigma_K = 5

                # Cross-Correlations
                ASD_vs_DS = 0.5
                ASA_vs_DS = 0.8
                ASA_vs_SF = -0.4
                ASD_vs_SF = -0.5
                DS_vs_FS = -0.4
                ASD_vs_ASA = 0.4
                ASD_vs_K = -0.2
                ASA_vs_K = -0.3
                DS_vs_K = -0.7
                SF_vs_K = 0.5

                ZSD_vs_SF = 0.0
                ZSA_vs_SF = 0.0
                ZSD_vs_K = 0.0
                ZSA_vs_K = 0.0
                ZSD_vs_DS = 0.0
                ZSA_vs_DS = 0.2
                ZSD_vs_ASD = 0.5
                ZSA_vs_ASD = 0.3
                ZSD_vs_ASA = 0.0
                ZSA_vs_ASA = 0.0
                ZSD_vs_ZSA = 0.0

                # Delay Scaling Parameter
                r_tau = 3

                # XPR [dB]
                mu_XPR = 9
                sigma_xpr = 3

                # Number of clusters
                N = 12

                # Number of rays per cluster
                M = 20

                # Cluster DS [ns]
                c_DS = 5

                # Cluster ASD [deg]
                c_ASD = 3

                # Cluster ASA [deg]
                c_ASA = 17

                # Cluster ZSA [deg]
                c_ZSA = 7

                # Per cluster shadowing std [dB]
                xi = 7

                # Correlation distance in the horizontal plane [m]
                corr_dist_h_plane_DS = 7
                corr_dist_h_plane_ASD = 8
                corr_dist_h_plane_ASA = 8
                corr_dist_h_plane_SF = 10
                corr_dist_h_plane_K = 15
                corr_dist_h_plane_ZSA = 12
                corr_dist_h_plane_ZSD = 12

            if self.losMatrix[ue.ID][bs.ID] == 'NLOS':
                # Delay Spread (DS)
                mu_lg_DS = -0.24 * np.log10(1 + fc) - 6.83
                sigma_lg_DS = 0.16 * np.log10(1 + fc) + 0.28

                # AOD Spread (ASD)
                mu_lg_ASD = -0.23 * np.log10(1 + fc) + 1.53
                sigma_lg_ASD = 0.11 * np.log10(1 + fc) + 0.33

                # AOA Spread (ASA)
                mu_lg_ASA = -0.08 * np.log10(1 + fc) + 1.81
                sigma_lg_ASA = 0.05 * np.log10(1 + fc) + 0.3

                # ZOA Spread (ZSA)
                mu_lg_ZSA = -0.04 * np.log10(1 + fc) + 0.92
                sigma_lg_ZSA = -0.07 * np.log10(1 + fc) + 0.41

                # Shadow Fading (SF) [dB]
                # Todo: See Table 7.4.1-1

                # K Factor (K) [dB]
                mu_K = None
                sigma_K = None

                # Cross-Correlations
                ASD_vs_DS = 0.0
                ASA_vs_DS = 0.4
                ASA_vs_SF = -0.4
                ASD_vs_SF = 0.0
                DS_vs_SF = -0.7
                ASD_vs_ASA = 0.0
                ASD_vs_K = None
                ASA_vs_K = None
                DS_vs_K = None
                SF_vs_K = None

                ZSD_vs_SF = 0.0
                ZSA_vs_SF = 0.0
                ZSD_vs_K = None
                ZSA_vs_K = None
                ZSD_vs_DS = -0.5
                ZSA_vs_DS = 0.0
                ZSD_vs_ASD = 0.5
                ZSA_vs_ASD = 0.5
                ZSD_vs_ASA = 0.0
                ZSA_vs_ASA = 0.2
                ZSD_vs_ZSA = 0.0

                # Delay Scaling Parameter
                r_tau = 2.1

                # XPR [dB]
                mu_XPR = 8.0
                sigma_xpr = 3

                # Number of clusters
                N = 19

                # Number of rays per cluster
                M = 20

                # Cluster DS [ns]
                c_DS = 11

                # Cluster ASD [deg]
                c_ASD = 10

                # Cluster ASA [deg]
                c_ASA = 22

                # Cluster ZSA [deg]
                c_ZSA = 7

                # Per cluster shadowing std [dB]
                xi = 7

                # Correlation distance in the horizontal plane [m]
                corr_dist_h_plane_DS = 10
                corr_dist_h_plane_ASD = 10
                corr_dist_h_plane_ASA = 9
                corr_dist_h_plane_SF = 13
                corr_dist_h_plane_K = None
                corr_dist_h_plane_ZSA = 10
                corr_dist_h_plane_ZSD = 10

            if self.losMatrix[ue.ID][bs.ID] == 'O2I':
                # Delay Spread (DS)
                mu_lg_DS = -6.62
                sigma_lg_DS = 0.32

                # AOD Spread (ASD)
                mu_lg_ASD = 1.25
                sigma_lg_ASD = 0.42

                # AOA Spread (ASA)
                mu_lg_ASA = 1.76
                sigma_lg_ASA = 0.16

                # ZOA Spread (ZSA)
                mu_lg_ZSA = 1.01
                sigma_lg_ZSA = 0.43

                # Shadow Fading (SF) [dB]
                sigma_SF = 7

                # K Factor (K) [dB]
                mu_K = None
                sigma_K = None

                # Cross-Correlations
                ASD_vs_DS = 0.4
                ASA_vs_DS = 0.4
                ASA_vs_SF = 0
                ASD_vs_SF = 0.2
                DS_vs_SF = -0.5
                ASD_vs_ASA = 0.0
                ASD_vs_K = None
                ASA_vs_K = None
                DS_vs_K = None
                SF_vs_K = None

                ZSD_vs_SF = 0.0
                ZSA_vs_SF = 0.0
                ZSD_vs_K = None
                ZSA_vs_K = None
                ZSD_vs_DS = -0.6
                ZSA_vs_DS = -0.2
                ZSD_vs_ASD = -0.2
                ZSA_vs_ASD = 0.0
                ZSD_vs_ASA = 0.0
                ZSA_vs_ASA = 0.5
                ZSD_vs_ZSA = 0.5

                # Delay Scaling Parameter
                r_tau = 2.2

                # XPR [dB]
                mu_XPR = 9.0
                sigma_xpr = 5

                # Number of clusters
                N = 12

                # Number of rays per cluster
                M = 20

                # Cluster DS [ns]
                c_DS = 11

                # Cluster ASD [deg]
                c_ASD = 5

                # Cluster ASA [deg]
                c_ASA = 8

                # Cluster ZSA [deg]
                c_ZSA = 3

                # Per cluster shadowing std [dB]
                xi = 4

                # Correlation distance in the horizontal plane [m]
                corr_dist_h_plane_DS = 10
                corr_dist_h_plane_ASD = 11
                corr_dist_h_plane_ASA = 17
                corr_dist_h_plane_SF = 7
                corr_dist_h_plane_K = None
                corr_dist_h_plane_ZSA = 25
                corr_dist_h_plane_ZSD = 25

        if self.scenario == 'UMa':
            # Frequency correction - see NOTE 6 from Table 7.5-6 Part-1
            if fc < 6.0:
                fc = 6.0

            if self.losMatrix[ue.ID][bs.ID] == 'LOS':
                # Delay Spread (DS)
                mu_lg_DS = -6.955 - 0.0963 * np.log10(fc)
                sigma_lg_DS = 0.66

                # AOD Spread (ASD)
                mu_lg_ASD = 1.06 + 0.1114 * np.log10(fc)
                sigma_lg_ASD = 0.28

                # AOA Spread (ASA)
                mu_lg_ASA = 1.81
                sigma_lg_ASA = 0.20

                # ZOA Spread (ZSA)
                mu_lg_ZSA = 0.95
                sigma_lg_ZSA = 0.16

                # Shadow Fading (SF) [dB]
                # Todo: See Table 7.4.1-1

                # K Factor (K) [dB]
                mu_K = 9
                sigma_K = 3.5

                # Cross-Correlations
                ASD_vs_DS = 0.4
                ASA_vs_DS = 0.8
                ASA_vs_SF = -0.5
                ASD_vs_SF = -0.5
                DS_vs_FS = -0.4
                ASD_vs_ASA = 0.0
                ASD_vs_K = 0.0
                ASA_vs_K = -0.2
                DS_vs_K = -0.4
                SF_vs_K = 0.0

                ZSD_vs_SF = 0.0
                ZSA_vs_SF = -0.8
                ZSD_vs_K = 0.0
                ZSA_vs_K = 0.0
                ZSD_vs_DS = -0.2
                ZSA_vs_DS = 0.0
                ZSD_vs_ASD = 0.5
                ZSA_vs_ASD = 0.0
                ZSD_vs_ASA = -0.3
                ZSA_vs_ASA = 0.4
                ZSD_vs_ZSA = 0.0

                # Delay Scaling Parameter
                r_tau = 2.5

                # XPR [dB]
                mu_XPR = 8
                sigma_xpr = 4

                # Number of clusters
                N = 12

                # Number of rays per cluster
                M = 20

                # Cluster DS [ns]
                c_DS = max(0.25, 6.5622 - 3.4084 * np.log10(fc))

                # Cluster ASD [deg]
                c_ASD = 5

                # Cluster ASA [deg]
                c_ASA = 11

                # Cluster ZSA [deg]
                c_ZSA = 7

                # Per cluster shadowing std [dB]
                xi = 3

                # Correlation distance in the horizontal plane [m]
                corr_dist_h_plane_DS = 30
                corr_dist_h_plane_ASD = 18
                corr_dist_h_plane_ASA = 15
                corr_dist_h_plane_SF = 37
                corr_dist_h_plane_K = 12
                corr_dist_h_plane_ZSA = 15
                corr_dist_h_plane_ZSD = 15

            if self.losMatrix[ue.ID][bs.ID] == 'NLOS':
                # Delay Spread (DS)
                mu_lg_DS = -6.28 - 0.204 * np.log10(fc)
                sigma_lg_DS = 0.39

                # AOD Spread (ASD)
                mu_lg_ASD = 1.5 - 0.1144 * np.log10(fc)
                sigma_lg_ASD = 0.28

                # AOA Spread (ASA)
                mu_lg_ASA = 1.5 - 0.1144 * np.log10(fc)
                sigma_lg_ASA = 0.20

                # ZOA Spread (ZSA)
                mu_lg_ZSA = 0.95
                sigma_lg_ZSA = 0.16

                # Shadow Fading (SF) [dB]
                # Todo: See Table 7.4.1-1

                # K Factor (K) [dB]
                mu_K = 9
                sigma_K = 3.5

                # Cross-Correlations
                ASD_vs_DS = 0.4
                ASA_vs_DS = 0.8
                ASA_vs_SF = -0.5
                ASD_vs_SF = -0.5
                DS_vs_FS = -0.4
                ASD_vs_ASA = 0.0
                ASD_vs_K = 0.0
                ASA_vs_K = -0.2
                DS_vs_K = -0.4
                SF_vs_K = 0.0

                ZSD_vs_SF = 0.0
                ZSA_vs_SF = -0.8
                ZSD_vs_K = 0.0
                ZSA_vs_K = 0.0
                ZSD_vs_DS = -0.2
                ZSA_vs_DS = 0.0
                ZSD_vs_ASD = 0.5
                ZSA_vs_ASD = 0.0
                ZSD_vs_ASA = -0.3
                ZSA_vs_ASA = 0.4
                ZSD_vs_ZSA = 0.0

                # Delay Scaling Parameter
                r_tau = 2.5

                # XPR [dB]
                mu_XPR = 8
                sigma_xpr = 4

                # Number of clusters
                N = 12

                # Number of rays per cluster
                M = 20

                # Cluster DS [ns]
                c_DS = max(0.25, 6.5622 - 3.4084 * np.log10(fc))

                # Cluster ASD [deg]
                c_ASD = 5

                # Cluster ASA [deg]
                c_ASA = 11

                # Cluster ZSA [deg]
                c_ZSA = 7

                # Per cluster shadowing std [dB]
                xi = 3

                # Correlation distance in the horizontal plane [m]
                corr_dist_h_plane_DS = 30
                corr_dist_h_plane_ASD = 18
                corr_dist_h_plane_ASA = 15
                corr_dist_h_plane_SF = 37
                corr_dist_h_plane_K = 12
                corr_dist_h_plane_ZSA = 15
                corr_dist_h_plane_ZSD = 15

            if self.losMatrix[ue.ID][bs.ID] == 'O2I':
                # Delay Spread (DS)
                mu_lg_DS = -6.62
                sigma_lg_DS = 0.32

                # AOD Spread (ASD)
                mu_lg_ASD = 1.25
                sigma_lg_ASD = 0.42

                # AOA Spread (ASA)
                mu_lg_ASA = 1.76
                sigma_lg_ASA = 0.16

                # ZOA Spread (ZSA)
                mu_lg_ZSA = 1.01
                sigma_lg_ZSA = 0.43

                # Shadow Fading (SF) [dB]
                sigma_SF = 7

                # K Factor (K) [dB]
                mu_K = None
                sigma_K = None

                # Cross-Correlations
                ASD_vs_DS = 0.4
                ASA_vs_DS = 0.4
                ASA_vs_SF = 0
                ASD_vs_SF = 0.2
                DS_vs_SF = -0.5
                ASD_vs_ASA = 0.0
                ASD_vs_K = None
                ASA_vs_K = None
                DS_vs_K = None
                SF_vs_K = None

                ZSD_vs_SF = 0.0
                ZSA_vs_SF = 0.0
                ZSD_vs_K = None
                ZSA_vs_K = None
                ZSD_vs_DS = -0.6
                ZSA_vs_DS = -0.2
                ZSD_vs_ASD = -0.2
                ZSA_vs_ASD = 0.0
                ZSD_vs_ASA = 0.0
                ZSA_vs_ASA = 0.5
                ZSD_vs_ZSA = 0.5

                # Delay Scaling Parameter
                r_tau = 2.2

                # XPR [dB]
                mu_XPR = 9.0
                sigma_xpr = 5

                # Number of clusters
                N = 12

                # Number of rays per cluster
                M = 20

                # Cluster DS [ns]
                c_DS = 11

                # Cluster ASD [deg]
                c_ASD = 5

                # Cluster ASA [deg]
                c_ASA = 8

                # Cluster ZSA [deg]
                c_ZSA = 3

                # Per cluster shadowing std [dB]
                xi = 4

                # Correlation distance in the horizontal plane [m]
                corr_dist_h_plane_DS = 10
                corr_dist_h_plane_ASD = 11
                corr_dist_h_plane_ASA = 17
                corr_dist_h_plane_SF = 7
                corr_dist_h_plane_K = None
                corr_dist_h_plane_ZSA = 25
                corr_dist_h_plane_ZSD = 25

        if self.scenario == 'RMa':
            if self.losMatrix[ue.ID][bs.ID] == 'LOS':
                # Delay Spread (DS)
                mu_lg_DS = -7.49
                sigma_lg_DS = 0.55

                # AOD Spread (ASD)
                mu_lg_ASD = 0.90
                sigma_lg_ASD = 0.38

                # AOA Spread (ASA)
                mu_lg_ASA = 1.52
                sigma_lg_ASA = 0.24

                # ZOA Spread (ZSA)
                mu_lg_ZSA = 0.47
                sigma_lg_ZSA = 0.40

                # Shadow Fading (SF) [dB]
                # Todo: See Table 7.4.1-1

                # K Factor (K) [dB]
                mu_K = 7
                sigma_K = 4

                # Cross-Correlations
                ASD_vs_DS = 0.0
                ASA_vs_DS = 0.0
                ASA_vs_SF = 0.0
                ASD_vs_SF = 0.0
                DS_vs_FS = -0.5
                ASD_vs_ASA = 0.0
                ASD_vs_K = 0.0
                ASA_vs_K = 0.0
                DS_vs_K = 0.0
                SF_vs_K = 0.0

                ZSD_vs_SF = 0.01
                ZSA_vs_SF = -0.17
                ZSD_vs_K = 0.0
                ZSA_vs_K = -0.02
                ZSD_vs_DS = -0.05
                ZSA_vs_DS = 0.27
                ZSD_vs_ASD = 0.73
                ZSA_vs_ASD = -0.14
                ZSD_vs_ASA = -0.20
                ZSA_vs_ASA = 0.24
                ZSD_vs_ZSA = -0.07

                # Delay Scaling Parameter
                r_tau = 3.8

                # XPR [dB]
                mu_XPR = 12
                sigma_xpr = 4

                # Number of clusters
                N = 11

                # Number of rays per cluster
                M = 20

                # Cluster DS [ns]
                c_DS = None

                # Cluster ASD [deg]
                c_ASD = 2

                # Cluster ASA [deg]
                c_ASA = 3

                # Cluster ZSA [deg]
                c_ZSA = 3

                # Per cluster shadowing std [dB]
                xi = 3

                # Correlation distance in the horizontal plane [m]
                corr_dist_h_plane_DS = 50
                corr_dist_h_plane_ASD = 25
                corr_dist_h_plane_ASA = 35
                corr_dist_h_plane_SF = 37
                corr_dist_h_plane_K = 40
                corr_dist_h_plane_ZSA = 15
                corr_dist_h_plane_ZSD = 15

            if self.losMatrix[ue.ID][bs.ID] == 'NLOS':
                # Delay Spread (DS)
                mu_lg_DS = -7.43
                sigma_lg_DS = 0.48

                # AOD Spread (ASD)
                mu_lg_ASD = 0.95
                sigma_lg_ASD = 0.45

                # AOA Spread (ASA)
                mu_lg_ASA = 1.52
                sigma_lg_ASA = 0.13

                # ZOA Spread (ZSA)
                mu_lg_ZSA = 0.58
                sigma_lg_ZSA = 0.37

                # Shadow Fading (SF) [dB]
                # Todo: See Table 7.4.1-1

                # K Factor (K) [dB]
                mu_K = None
                sigma_K = None

                # Cross-Correlations
                ASD_vs_DS = -0.4
                ASA_vs_DS = 0.0
                ASA_vs_SF = 0.0
                ASD_vs_SF = 0.6
                DS_vs_FS = -0.5
                ASD_vs_ASA = 0.0
                ASD_vs_K = None
                ASA_vs_K = None
                DS_vs_K = None
                SF_vs_K = None

                ZSD_vs_SF = -0.04
                ZSA_vs_SF = -0.25
                ZSD_vs_K = None
                ZSA_vs_K = None
                ZSD_vs_DS = -0.10
                ZSA_vs_DS = -0.40
                ZSD_vs_ASD = 0.42
                ZSA_vs_ASD = -0.27
                ZSD_vs_ASA = -0.18
                ZSA_vs_ASA = 0.26
                ZSD_vs_ZSA = -0.27

                # Delay Scaling Parameter
                r_tau = 1.7

                # XPR [dB]
                mu_XPR = 7
                sigma_xpr = 3

                # Number of clusters
                N = 10

                # Number of rays per cluster
                M = 20

                # Cluster DS [ns]
                c_DS = None

                # Cluster ASD [deg]
                c_ASD = 2

                # Cluster ASA [deg]
                c_ASA = 3

                # Cluster ZSA [deg]
                c_ZSA = 3

                # Per cluster shadowing std [dB]
                xi = 3

                # Correlation distance in the horizontal plane [m]
                corr_dist_h_plane_DS = 36
                corr_dist_h_plane_ASD = 30
                corr_dist_h_plane_ASA = 40
                corr_dist_h_plane_SF = 120
                corr_dist_h_plane_K = None
                corr_dist_h_plane_ZSA = 50
                corr_dist_h_plane_ZSD = 50

            if self.losMatrix[ue.ID][bs.ID] == 'O2I':
                # Delay Spread (DS)
                mu_lg_DS = -7.47
                sigma_lg_DS = 0.24

                # AOD Spread (ASD)
                mu_lg_ASD = 0.67
                sigma_lg_ASD = 0.18

                # AOA Spread (ASA)
                mu_lg_ASA = 1.66
                sigma_lg_ASA = 0.21

                # ZOA Spread (ZSA)
                mu_lg_ZSA = 0.93
                sigma_lg_ZSA = 0.22

                # Shadow Fading (SF) [dB]
                sigma_SF = 8

                # K Factor (K) [dB]
                mu_K = None
                sigma_K = None

                # Cross-Correlations
                ASD_vs_DS = 0.0
                ASA_vs_DS = 0.0
                ASA_vs_SF = 0.0
                ASD_vs_SF = 0.0
                DS_vs_FS = 0.0
                ASD_vs_ASA = -0.7
                ASD_vs_K = None
                ASA_vs_K = None
                DS_vs_K = None
                SF_vs_K = None

                ZSD_vs_SF = 0.0
                ZSA_vs_SF = 0.0
                ZSD_vs_K = None
                ZSA_vs_K = None
                ZSD_vs_DS = 0.0
                ZSA_vs_DS = 0.0
                ZSD_vs_ASD = 0.66
                ZSA_vs_ASD = 0.47
                ZSD_vs_ASA = -0.55
                ZSA_vs_ASA = -0.22
                ZSD_vs_ZSA = 0.0

                # Delay Scaling Parameter
                r_tau = 1.7

                # XPR [dB]
                mu_XPR = 7
                sigma_xpr = 3

                # Number of clusters
                N = 10

                # Number of rays per cluster
                M = 20

                # Cluster DS [ns]
                c_DS = None

                # Cluster ASD [deg]
                c_ASD = 2

                # Cluster ASA [deg]
                c_ASA = 3

                # Cluster ZSA [deg]
                c_ZSA = 3

                # Per cluster shadowing std [dB]
                xi = 3

                # Correlation distance in the horizontal plane [m]
                corr_dist_h_plane_DS = 36
                corr_dist_h_plane_ASD = 30
                corr_dist_h_plane_ASA = 40
                corr_dist_h_plane_SF = 120
                corr_dist_h_plane_K = None
                corr_dist_h_plane_ZSA = 50
                corr_dist_h_plane_ZSD = 50

        if self.scenario == 'InH':
            # Frequency correction - see NOTE 6 from Table 7.5-6 Part-2
            if fc < 6.0:
                fc = 6.0

            if self.losMatrix[ue.ID][bs.ID] == 'LOS':
                # Delay Spread (DS)
                mu_lg_DS = -0.01 * np.log10(1+fc) - 7.692
                sigma_lg_DS = 0.18

                # AOD Spread (ASD)
                mu_lg_ASD = 1.60
                sigma_lg_ASD = 0.18

                # AOA Spread (ASA)
                mu_lg_ASA = -0.19 * np.log10(1+fc) + 1.781
                sigma_lg_ASA = 0.12 * np.log10(1+fc) + 0.119

                # ZOA Spread (ZSA)
                mu_lg_ZSA = -0.26 * np.log10(1+fc) + 1.44
                sigma_lg_ZSA = -0.04 * np.log10(1+fc) + 0.264

                # Shadow Fading (SF) [dB]
                # Todo: See Table 7.4.1-1

                # K Factor (K) [dB]
                mu_K = 7
                sigma_K = 4

                # Cross-Correlations
                ASD_vs_DS = 0.6
                ASA_vs_DS = 0.8
                ASA_vs_SF = -0.5
                ASD_vs_SF = -0.4
                DS_vs_FS = -0.8
                ASD_vs_ASA = 0.4
                ASD_vs_K = 0.0
                ASA_vs_K = 0.0
                DS_vs_K = -0.5
                SF_vs_K = 0.5

                ZSD_vs_SF = 0.2
                ZSA_vs_SF = 0.3
                ZSD_vs_K = 0.0
                ZSA_vs_K = 0.1
                ZSD_vs_DS = 0.1
                ZSA_vs_DS = 0.2
                ZSD_vs_ASD = 0.5
                ZSA_vs_ASD = 0.0
                ZSD_vs_ASA = 0.0
                ZSA_vs_ASA = 0.5
                ZSD_vs_ZSA = 0.0

                # Delay Scaling Parameter
                r_tau = 3.6

                # XPR [dB]
                mu_XPR = 11
                sigma_xpr = 4

                # Number of clusters
                N = 15

                # Number of rays per cluster
                M = 20

                # Cluster DS [ns]
                c_DS = None

                # Cluster ASD [deg]
                c_ASD = 5

                # Cluster ASA [deg]
                c_ASA = 8

                # Cluster ZSA [deg]
                c_ZSA = 9

                # Per cluster shadowing std [dB]
                xi = 6

                # Correlation distance in the horizontal plane [m]
                corr_dist_h_plane_DS = 8
                corr_dist_h_plane_ASD = 7
                corr_dist_h_plane_ASA = 5
                corr_dist_h_plane_SF = 10
                corr_dist_h_plane_K = 4
                corr_dist_h_plane_ZSA = 4
                corr_dist_h_plane_ZSD = 4

            if self.losMatrix[ue.ID][bs.ID] == 'NLOS':
                # Delay Spread (DS)
                mu_lg_DS = -0.28 * np.log10(1 + fc) - 7.173
                sigma_lg_DS = 0.10 * np.log10(1 + fc) + 0.055

                # AOD Spread (ASD)
                mu_lg_ASD = 1.62
                sigma_lg_ASD = 0.25

                # AOA Spread (ASA)
                mu_lg_ASA = -0.11 * np.log10(1 + fc) + 1.863
                sigma_lg_ASA = 0.12 * np.log10(1 + fc) + 0.059

                # ZOA Spread (ZSA)
                mu_lg_ZSA = -0.15 * np.log10(1 + fc) + 1.287
                sigma_lg_ZSA = -0.09 * np.log10(1 + fc) + 0.746

                # Shadow Fading (SF) [dB]
                # Todo: See Table 7.4.1-1

                # K Factor (K) [dB]
                mu_K = None
                sigma_K = None

                # Cross-Correlations
                ASD_vs_DS = 0.4
                ASA_vs_DS = 0.0
                ASA_vs_SF = -0.4
                ASD_vs_SF = 0.0
                DS_vs_FS = -0.5
                ASD_vs_ASA = 0.0
                ASD_vs_K = None
                ASA_vs_K = None
                DS_vs_K = None
                SF_vs_K = None

                ZSD_vs_SF = 0.0
                ZSA_vs_SF = 0.0
                ZSD_vs_K = None
                ZSA_vs_K = None
                ZSD_vs_DS = -0.27
                ZSA_vs_DS = -0.06
                ZSD_vs_ASD = 0.35
                ZSA_vs_ASD = 0.23
                ZSD_vs_ASA = -0.08
                ZSA_vs_ASA = 0.43
                ZSD_vs_ZSA = 0.42

                # Delay Scaling Parameter
                r_tau = 3.0

                # XPR [dB]
                mu_XPR = 10
                sigma_xpr = 4

                # Number of clusters
                N = 19

                # Number of rays per cluster
                M = 20

                # Cluster DS [ns]
                c_DS = None

                # Cluster ASD [deg]
                c_ASD = 5

                # Cluster ASA [deg]
                c_ASA = 11

                # Cluster ZSA [deg]
                c_ZSA = 9

                # Per cluster shadowing std [dB]
                xi = 3

                # Correlation distance in the horizontal plane [m]
                corr_dist_h_plane_DS = 5
                corr_dist_h_plane_ASD = 3
                corr_dist_h_plane_ASA = 3
                corr_dist_h_plane_SF = 6
                corr_dist_h_plane_K = None
                corr_dist_h_plane_ZSA = 4
                corr_dist_h_plane_ZSD = 4


        # TODO: Generate parametes using procedure described in clause 3.3.1 of the Winner Channel Model document
        # Generate DS
        DS = 0.0

        # Generate ASA
        ASA = 0.0
        ASA = min(ASA, 104.0)

        # Generate ASD
        ASD = 0.0
        ASD = min(ASD, 104.0)

        # Generate ZSA
        ZSA = 0.0
        ZSA = min(ZSA, 52.0)

        # Generate ZSD
        ZSD = 0.0
        ZSD = min(ZSD, 52.0)

        # Generate K
        K = 0.0

        # Generate SF
        SF = 0.0


        LSP = {'mu_lg_DS': mu_lg_DS, 'sigma_lg_DS': sigma_lg_DS, 'mu_lg_ASD': mu_lg_ASD, 'sigma_lg_ASD': sigma_lg_ASD,
               'mu_lg_ASA': mu_lg_ASA, 'sigma_lg_ASA': sigma_lg_ASA, 'mu_lg_ZSA': mu_lg_ZSA,
               'sigma_lg_ZSA': sigma_lg_ZSA, 'sigma_SF': sigma_SF, 'mu_K': mu_K, 'sigma_K': sigma_K,
               'ASD_vs_DS': ASD_vs_DS, 'ASA_vs_DS': ASA_vs_DS, 'ASA_vs_SF': ASA_vs_SF, 'ASD_vs_SF': ASD_vs_SF,
               'DS_vs_FS': DS_vs_FS, 'ASD_vs_ASA': ASD_vs_ASA, 'ASD_vs_K': ASD_vs_K, 'ASA_vs_K': ASA_vs_K,
               'DS_vs_K': DS_vs_K, 'SF_vs_K': SF_vs_K, 'ZSD_vs_SF': ZSD_vs_SF, 'ZSA_vs_SF': ZSA_vs_SF,
               'ZSD_vs_K': ZSD_vs_K, 'ZSA_vs_K': ZSA_vs_K, 'ZSD_vs_DS': ZSD_vs_DS, 'ZSA_vs_DS': ZSA_vs_DS,
               'ZSD_vs_ASD': ZSD_vs_ASD, 'ZSA_vs_ASD': ZSA_vs_ASD, 'ZSD_vs_ASA': ZSD_vs_ASA, 'ZSA_vs_ASA': ZSA_vs_ASA,
               'ZSD_vs_ZSA': ZSD_vs_ZSA, 'r_tau': r_tau, 'mu_XPR': mu_XPR, 'sigma_xpr': sigma_xpr, 'N': N, 'M': M,
               'c_DS': c_DS, 'c_ASD': c_ASD, 'c_ASA': c_ASA, 'c_ZSA': c_ZSA, 'xi': xi,
               'corr_dist_h_plane_DS': corr_dist_h_plane_DS, 'corr_dist_h_plane_ASD': corr_dist_h_plane_ASD,
               'corr_dist_h_plane_ASA': corr_dist_h_plane_ASA, 'corr_dist_h_plane_SF': corr_dist_h_plane_SF,
               'corr_dist_h_plane_K': corr_dist_h_plane_K, 'corr_dist_h_plane_ZSA': corr_dist_h_plane_ZSA,
               'corr_dist_h_plane_ZSD': corr_dist_h_plane_ZSD,
               'DS': DS, 'ASA': ASA, 'ASD': ASD, 'ZSA': ZSA, 'ZSD': ZSD, 'K': K, 'SF': SF
               }
        return LSP

    def generateSmallScaleParams_link(self, bs: BaseStation, sec: BaseStation.Sector, ue: UserEquipment, lsp: dict):
        # Small Scale Parameter generation

        ################################################################################################################
        # Step 5: Generate cluster delays Tau_n:
        Xn = np.random.uniform(size=lsp['N'])
        cluster_delay = - lsp['r_tau'] * lsp['DS'] * np.log(Xn)
        cluster_delay = np.sort(cluster_delay - min(cluster_delay))

        if self.losMatrix[ue.ID][bs.ID] == 'LOS':
            C_tau = 0.7705 - 0.0433 * lsp['K'] + 0.0002 * (lsp['K'] ** 2) + 0.000017 * (lsp['K'] ** 3)
            cluster_delay_LOS = cluster_delay/C_tau

        ################################################################################################################
        # Step 6: Generate cluster powers P_n:
        P_n_notch = np.exp(-cluster_delay * ((lsp['r_tau'] - 1)/(lsp['r_tau'] * lsp['DS'])))
        P_n_notch = P_n_notch * (10 ** (- np.random.normal(loc=0.0, scale=lsp['xi']) / 10))
        P_n = P_n_notch/sum(P_n_notch)

        if self.losMatrix[ue.ID][bs.ID] == 'LOS':
            K_R_linear = 10 ** (lsp['K']/10)
            P_n = (1 / (K_R_linear + 1)) * P_n
            P_n[0] = P_n[0] + (K_R_linear / (K_R_linear + 1))

        # Discard clusters with power less than -25 dB compared to the maximum cluster power
        P_n_dB = 10 * np.log10(P_n)
        P_n_dB = P_n_dB[P_n_dB >= max(P_n_dB) - 25]
        P_n = P_n[P_n_dB >= max(P_n_dB) - 25]

        # Power per ray
        P_n_ray = P_n/lsp['M']

        updated_N_cluster = len(P_n)

        ################################################################################################################
        # Step 7: Generate arrival angles and departure angles for both azimuth and elevation
        if updated_N_cluster == 4:
            C_phi_NLOS = 0.779
        elif updated_N_cluster == 5:
            C_phi_NLOS = 0.860
        elif updated_N_cluster == 6:    # Not in the standard - obtained from linear interpolation
            C_phi_NLOS = 0.913
        elif updated_N_cluster == 7:    # Not in the standard - obtained from linear interpolation
            C_phi_NLOS = 0.965
        elif updated_N_cluster == 8:
            C_phi_NLOS = 1.018
        elif updated_N_cluster == 9:    # Not in the standard - obtained from linear interpolation
            C_phi_NLOS = 1.054
        elif updated_N_cluster == 10:
            C_phi_NLOS = 1.090
        elif updated_N_cluster == 11:
            C_phi_NLOS = 1.123
        elif updated_N_cluster == 12:
            C_phi_NLOS = 1.146
        elif updated_N_cluster == 13:  # Not in the standard - obtained from linear interpolation
            C_phi_NLOS = 1.168
        elif updated_N_cluster == 14:
            C_phi_NLOS = 1.190
        elif updated_N_cluster == 15:
            C_phi_NLOS = 1.211
        elif updated_N_cluster == 16:
            C_phi_NLOS = 1.226
        elif updated_N_cluster == 17:  # Not in the standard - obtained from linear interpolation
            C_phi_NLOS = 1.242
        elif updated_N_cluster == 18:  # Not in the standard - obtained from linear interpolation
            C_phi_NLOS = 1.257
        elif updated_N_cluster == 19:
            C_phi_NLOS = 1.273
        elif updated_N_cluster == 20:
            C_phi_NLOS = 1.289
        elif updated_N_cluster == 21:  # Not in the standard - obtained from linear interpolation
            C_phi_NLOS = 1.303
        elif updated_N_cluster == 22:  # Not in the standard - obtained from linear interpolation
            C_phi_NLOS = 1.317
        elif updated_N_cluster == 23:  # Not in the standard - obtained from linear interpolation
            C_phi_NLOS = 1.330
        elif updated_N_cluster == 24:  # Not in the standard - obtained from linear interpolation
            C_phi_NLOS = 1.344
        elif updated_N_cluster == 25:
            C_phi_NLOS = 1.358
        else:
            raise "Invalid number of clusters"

        if self.losMatrix[ue.ID][bs.ID] == 'LOS':
            C_phi = C_phi_NLOS * (1.1035 - 0.028 * lsp['K'] - 0.002 * (lsp['K'] ** 2) + 0.0001 * (lsp['K'] ** 3))
        else:
            C_phi = C_phi_NLOS



    def computeRSRP(self, BS_list: list[BaseStation], UE_list: list[UserEquipment]):
        """
        Compute the RSRP
        :param BS_list: list of BaseStation
        :param UE_list: list of UserEquipment
        :return: update Network attribute RsrpMatrix
        """
        nBS = len(BS_list)
        nUE = len(UE_list)
        nSectors = nBS * self.number_of_sectors
        self.RsrpMatrix = np.zeros((nUE, nSectors))
        for bs_ind, bs in enumerate(BS_list):
            for sec_ind, sec in enumerate(bs.sector):
                for eu_ind, ue in enumerate(UE_list):
                    self.RsrpMatrix[ue.ID][sec.ID] = sec.tx_power_dB - self.pathlossMatrix[ue.ID][sec.ID]
        # with np.printoptions(precision=1, suppress=True):
        #     print(self.RsrpMatrix)

    def UE_attach(self, BS_list: list[BaseStation], UE_list: list[UserEquipment]):
        """
        Performs UE attachment -> finds the sector for which UE senses the highest RSRP and 'connect' to them
        :param BS_list:
        :param UE_list:
        :return: updates attributes 'serving_sector' and 'serving_base_station' of the UserEquipment object
        """
        self.computeRSRP(BS_list=BS_list, UE_list=UE_list)
        for eu_ind, ue in enumerate(UE_list):
            highestRSRP_sectorIndex = np.argmax(self.RsrpMatrix[eu_ind][:])
            ue.serving_sector = highestRSRP_sectorIndex
            ue.serving_base_station = self.cellSectorMap[highestRSRP_sectorIndex]

            # with np.printoptions(precision=1, suppress=True):
            #     print(f'UE:{eu_ind} - RSRP:{self.RsrpMatrix[eu_ind][:]}')
            #     print(f'UE:{eu_ind} - RSRP:{self.RsrpMatrix[eu_ind][highestRSRP_sectorIndex]}')

    def computeSINR(self, BS_list: list[BaseStation], UE_list: list[UserEquipment]):
        """
        Computes the SINR for the UEs in the UE list; it assumes that the desired signal is received from the serving
        sector and the interference is received for all other sectors.
        :param BS_list: list of BaseStation
        :param UE_list: list of UserEquipment
        :return:
        """
        nBS = len(BS_list)
        nUE = len(UE_list)
        nSectors = nBS * self.number_of_sectors
        self.SinrMatrix = np.zeros(nUE)
        for eu_ind, ue in enumerate(UE_list):
            signal_power_dB = self.RsrpMatrix[ue.ID][ue.serving_sector]
            interference_plus_noise = 10 ** (ue.noise_floor / 10)
            for sec_idx in (np.delete(np.arange(nSectors), ue.serving_sector)):
                interference_plus_noise = interference_plus_noise + 10 ** (self.RsrpMatrix[ue.ID][sec_idx] / 10)
            interference_plus_noise_dB = 10 * np.log10(interference_plus_noise)
            self.SinrMatrix[ue.ID] = signal_power_dB - interference_plus_noise_dB

            # with np.printoptions(precision=1, suppress=True):
            #     print(f'UE:{eu_ind} - SINR:{self.SinrMatrix[ue.ID]}')

    def cell_sector_mapping(self, BS_list: list[BaseStation]):
        nBS = len(BS_list)
        nSectors = nBS * self.number_of_sectors

        self.cellSectorMap = [None] * nSectors
        for bs in BS_list:
            for sec in bs.sector:
                self.cellSectorMap[sec.ID] = bs.ID


class Grid:
    def __init__(self, x_length: float = 100, y_length: float = 100, Nx: int = 100, Ny: int = 100,
                 center_at_origin: bool = True):
        self.Nx = Nx
        self.Ny = Ny
        self.coord_x, self.coord_y = np.meshgrid(np.arange(Nx + 1), np.arange(Ny + 1), sparse=False, indexing='xy')
        self.coord_x, self.coord_y = x_length * self.coord_x / Nx, y_length * self.coord_y / Ny

        if center_at_origin:
            self.coord_x = self.coord_x - x_length / 2
            self.coord_y = self.coord_y - y_length / 2

        self.coordinates = [self.coord_x, self.coord_y]
    # TODO: Plotable grid (draw lines)


if __name__ == "__main__":
    network = Network(scenario='UMa', free_space=True)

    plt.ion()
    fig, (h_ax, ax2) = plt.subplots(1, 2)
    Nx = 200
    Ny = 200

    N = 5
    d = network.ISD
    hex_centers, h_ax = hexalattice.create_hex_grid(nx=N, ny=N, crop_circ=N * d / 2, min_diam=d, do_plot=True,
                                                    h_ax=h_ax,rotate_deg=30)
    tile_centers_x = hex_centers[:, 0]
    tile_centers_y = hex_centers[:, 1]
    x_lim = h_ax.get_xlim()
    y_lim = h_ax.get_ylim()

    UE_x_lim = (min(tile_centers_x) - d / 2, max(tile_centers_x) + d / 2)
    UE_y_lim = (min(tile_centers_y) - d / 2, max(tile_centers_y) + d / 2)
    print(UE_x_lim, UE_y_lim)

    grid = Grid(x_length=x_lim[1] - x_lim[0], y_length=y_lim[1] - y_lim[0], Nx=Nx, Ny=Ny)

    # Initialize Base Stations
    NBS = len(tile_centers_x)
    BSs = []
    for i in range(NBS):
        BS = BaseStation(pos_x=tile_centers_x[i], pos_y=tile_centers_y[i], tx_power_dB=23, height=network.BS_height,
                         number_of_sectors=network.number_of_sectors)
        BSs.append(BS)

    # Initialize User Equipment
    NUE = 1000
    UEs = []

    for i in range(NUE):
        pos_xx = np.random.uniform(low=UE_x_lim[0], high=UE_x_lim[1], size=None)
        pos_yy = np.random.uniform(low=UE_y_lim[0], high=UE_y_lim[1], size=None)
        UE = UserEquipment(pos_x=pos_xx, pos_y=pos_yy, height=network.UE_height, tx_power_dB=18, noise_floor=-126)
        UE.location = network.UELocation()  # Indoor/Outdoor
        UEs.append(UE)

    # Compute LOS and Pathloss for all UEs/BS/Sector
    network.cell_sector_mapping(BSs)
    network.NetworkPathlossAndLos(BSs, UEs)
    network.computeRSRP(BSs, UEs)
    network.UE_attach(BSs, UEs)
    network.computeSINR(BSs, UEs)

    color = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']
    marker = ['.', 'x', '+']
    for bs in BSs:
        h_ax.scatter(bs.pos_x, bs.pos_y,
                     color=color[int(bs.ID % 7)],
                     marker='^')
    for ue in UEs:
        h_ax.scatter(ue.pos_x, ue.pos_y,
                     color=color[int(ue.serving_base_station % 7)],
                     marker=marker[int(ue.serving_sector % 3)], alpha=0.5)

    # ax2.hist(network.SinrMatrix, range=[np.floor(min(network.SinrMatrix)) - 1, np.ceil(max(network.SinrMatrix)) + 1],
    #          bins=np.arange(int(np.floor(min(network.SinrMatrix))), int(np.ceil(max(network.SinrMatrix))), 1))

    SinrMatrix = np.clip(network.SinrMatrix, -60, 60)
    ax2.hist(SinrMatrix, range=[np.floor(min(SinrMatrix)) - 1, np.ceil(max(SinrMatrix)) + 1])

    # chart = h_ax.pcolormesh(grid.coord_x, grid.coord_y, total_signal_power, shading='auto', alpha=.8, cmap='turbo')
    # plt.colorbar(chart, ax=h_ax)

    plt.show(block=True)
