import numpy as np
import itertools
from typing import List, Union
from scipy.signal import convolve2d
from scipy.linalg import sqrtm
from scipy.interpolate import RegularGridInterpolator
from enum import Enum


class LOS(Enum):
    UNDETERMINED = 0
    LOS = 1
    NLOS = 2
    O2I = 3


class Location(Enum):
    UNDETERMINED = 0
    Indoor = 1
    Outdoor = 2
    Car = 3


class AntennaPanel:
    def __init__(self, sector_id: int,
                 n_panel_col: int = 1, n_panel_row: int = 1,
                 panel_v_spacing: float = 0.5, panel_h_spacing: float = 0.5,
                 n_antenna_col: int = 8, n_antenna_row: int = 8,
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


class Sector:
    # Auto ID generation
    sector_id = itertools.count()

    def __init__(self, bs_id: int, orientation: float = 0, sector_width: float = 120,
                 frequency: int = 3.5, tx_power_dB: float = 20):
        """
        Create sectors within a Base Station
        :param bs_id: Base Station ID
        :param orientation: orientation of the sector center [degrees] > 0 points toward North
        :param sector_width: width of the sector [degrees]
        :param frequency: operation frequency of the sector [GHz]
        :param tx_power_dB: transmission power [dBm]
        """
        self.ID = next(Sector.sector_id)
        self.BS_ID = bs_id
        self.orientation = orientation
        self.sector_width = sector_width
        self.frequency = frequency
        self.tx_power_dB = tx_power_dB
        self.number_of_PRBs = 100  # 20 MHz channel with SCS 15 KHz (LTE)
        self.connected_UEs = []


class BaseStation:
    # Auto ID generation
    bs_id = itertools.count()

    def __init__(self, pos_x: float = 0, pos_y: float = 0, height: float = 10, number_of_sectors: int = 3,
                 tx_power_dB: float = 23, rotation: float = 0):
        """
        Create a Base Station
        :param pos_x: position of the Base Station in the x axis [meters]
        :param pos_y: position of the Base Station in the y axis [meters]
        :param height: height of the Base Station [meters]
        :param number_of_sectors: number of sectors in the Base Station
        :param tx_power_dB: transmission power [dBm]
        :param rotation: offset to the Base Station' sector orientation [degrees]
        """
        self.ID = next(BaseStation.bs_id)
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.height = height
        self.tx_power_dB = tx_power_dB
        self.sector = []
        self.number_of_sectors = number_of_sectors
        for sec in range(number_of_sectors):
            orientation = rotation + sec * 360 / number_of_sectors
            sector_width = 360 / number_of_sectors
            sector = Sector(bs_id=self.ID, orientation=orientation, sector_width=sector_width,
                            tx_power_dB=self.tx_power_dB)
            print(f'Base Station {self.ID} - Sector {sector.ID}')
            self.sector.append(sector)

    def get_sector_by_ID(self, ID):
        for sec_idx, sec in enumerate(self.sector):
            if sec.ID == ID:
                return sec_idx

        else:
            return None


class UserEquipment:
    # pos_x = float
    # pos_y = float
    # height = float
    # location = ''
    # los = ''
    id_iter = itertools.count()

    def __init__(self, pos_x: float = 0, pos_y: float = 0, height: float = 1.5,
                 location: Location = Location.Outdoor, tx_power_dB: float = 0, noise_floor: float = -125):
        """
        Create User Equipment (UE)
        :param pos_x: position of the UE in the x axis [meters]
        :param pos_y: position of the UEt in the y axis [meters]
        :param height: height of the UE [meters]
        :param location: location of the UE ['Indoor','Outdoor']
        :param tx_power_dB: transmission power [dBm]
        :param noise_floor: noise floor [dBm]
        """
        self.ID = next(UserEquipment.id_iter)
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.height = height
        self.location = location
        self.orientation = np.random.uniform(0, 2 * np.pi)
        self.tx_power_dB = tx_power_dB
        self.serving_sector = None
        self.serving_base_station = None
        self.neighbor_cells = []
        self.noise_floor = noise_floor  # dB

        self.dist2D = np.array([])  # 2D Distance to each BS
        self.dist3D = np.array([])  # 3D Distance to each BS
        self.los_azi_angle_rad = np.array([])  # LOS Azimuth from each BS to the UE
        self.los_zen_angle_rad = np.array([])  # LOS Zenith from each BS to the UE

        self.LSP = []


class Network:
    def __init__(self, scenario: str = 'UMa', free_space: bool = False,
                 # BSs: Union[List[BaseStation], BaseStation] = list,
                 # UEs: Union[List[UserEquipment], UserEquipment] = None,
                 seed: int = None):
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

        # self.BSs = BSs
        self.BSs = []
        self.BS_tx_power_dB = 23
        self.BS_noise_floor_dB = -125

        # self.UEs = UEs
        self.UEs = []
        self.UE_tx_power_dB = 18
        self.UE_noise_floor_dB = -125

        self.pathlossMatrix = [[]]
        self.shadowFadingMatrix = [[]]
        self.RSRP_Matrix = np.array([])
        self.SINR_Matrix = np.array([])
        self.DL_SINR_Matrix = np.array([])
        self.UL_SINR_Matrix = np.array([])

        self.los_Matrix = []
        self.los_azi_angle_rad_Matrix = []
        self.los_zen_angle_rad_Matrix = []
        self.dist2D_Matrix = []
        self.dist3D_Matrix = []

        self.UE_attach_threshold = -np.inf  # Todo: Use reasonable value

        # Random Generators
        self.random_seed = seed
        self.random_generator_LOS = np.random.default_rng(seed=self.random_seed)
        self.random_generator_Location = np.random.default_rng(seed=self.random_seed)
        self.random_generator_PL = np.random.default_rng(seed=self.random_seed)

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
            self.average_street_width = 5.0  # meters [5..50]

        # TODO: Implement other scenarios:  Indoor Factory (InF) - (InF-SL, InF-DL, InF-SH, InF-DH, InF-HH)
        #                                   Indoor Office/Hotspot (InH)

    def add_ue(self, pos_x: float = 0, pos_y: float = 0, height: float = None, location: Location = Location.UNDETERMINED,
               tx_power_dB: float = None, noise_floor: float = None):

        if height is None:
            height = self.UE_height
        if location is None:
            location = self.UELocation()
        if tx_power_dB is None:
            tx_power_dB = self.UE_tx_power_dB
        if noise_floor is None:
            noise_floor = self.UE_noise_floor_dB

        self.UEs.append(UserEquipment(pos_x=pos_x,
                                      pos_y=pos_y,
                                      tx_power_dB=tx_power_dB,
                                      height=height,
                                      location=location,
                                      noise_floor=noise_floor))

    # def remove_ue(self, ue_id: Union[List[int], int] = None, verbose=False):
    #     if type(ue_id) is list:
    #         for ue_idx, ue in enumerate(self.UEs):
    #             if ue.ID in ue_id:
    #                 self.UEs.pop(ue_idx)
    #                 ue_id.remove(ue_idx)
    #         if verbose:
    #             if len(ue_id) == 0:
    #                 print(f'All UEs have been removed successfully.')
    #             elif len(ue_id) > 0:
    #                 print(f'UEs {ue_id} have not been removed.')
    #     elif type(ue_id) is int:
    #         for ue_idx, ue in enumerate(self.UEs):
    #             if ue.ID == ue_id:
    #                 self.UEs.pop(ue_idx)
    #                 if verbose:
    #                     print(f'UE {ue_id} has been removed successfully.')
    #                     break
    #     else:
    #         raise 'Invalid input'
    #     # Todo: may need to recompute many variables using the new list of UEs

    def add_bs(self, pos_x: float = 0, pos_y: float = 0, height: float = None, number_of_sectors: int = None,
               tx_power_dB: float = None, rotation: float = None):
        if height is None:
            height = self.BS_height
        if number_of_sectors is None:
            number_of_sectors = self.number_of_sectors
        if tx_power_dB is None:
            tx_power_dB = self.BS_tx_power_dB
        if rotation is None:
            rotation = 0

        self.BSs.append(BaseStation(pos_x=pos_x,
                                    pos_y=pos_y,
                                    tx_power_dB=tx_power_dB,
                                    height=height,
                                    number_of_sectors=number_of_sectors,
                                    rotation=rotation))

    # def remove_bs(self, bs_id: Union[List[int], int] = None, verbose=False):
    #     if type(bs_id) is list:
    #         for bs_idx, bs in enumerate(self.BSs):
    #             if bs.ID in bs_id:
    #                 self.BSs.pop(bs_idx)
    #                 bs_id.remove(bs_idx)
    #         if verbose:
    #             if len(bs_id) == 0:
    #                 print(f'All BSs have been removed successfully.')
    #             elif len(bs_id) > 0:
    #                 print(f'BSs {bs_id} have not been removed.')
    #     elif type(bs_id) is int:
    #         for bs_idx, bs in enumerate(self.BSs):
    #             if bs.ID == bs_id:
    #                 self.BSs.pop(bs_idx)
    #                 if verbose:
    #                     print(f'BS {bs_id} has been removed successfully.')
    #                     break
    #         print(f'BS {bs_id} has not been removed.')
    #     else:
    #         raise 'Invalid input'
    #     # Todo: may need to recompute many variables using the new list of UEs

    def LineOfSight(self, bs: BaseStation, ue: UserEquipment):
        """
        Determine if a given BS and UE pair is in 'LOS' or 'NLOS'
        0: 'LOS'        1: 'NLOS'       2: 'IOF'

        :param bs: BaseStation object
        :type bs: object BaseStation
        :param ue: UserEquipment object
        :type bs: object UserEquipment
        :return: the line-of-sight condition, i.e.: 'LOS' or 'NLOS'
        """

        dist_2D = dist2d(bs, ue)

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

                Pr_LOS = (18 / dist_2D) + np.exp(-dist_2D / 63) * (1 - 18 / dist_2D) * (
                            1 + C * (5 / 4) * ((dist_2D / 100) ** 3) * np.exp(-dist_2D / 150))

        # Urban Micro scenario
        if self.scenario == 'UMi':
            if dist_2D <= 18:
                Pr_LOS = 1
            else:
                Pr_LOS = (18 / dist_2D) + np.exp(-dist_2D / 36) * (1 - 18 / dist_2D)

        if self.random_generator_LOS.random() < Pr_LOS:
            # return 'LOS'
            # return 0
            return LOS.LOS
        else:
            # return 'NLOS'
            # return 1
            return LOS.NLOS

    def UELocation(self):
        """
        Generates the location of an UE based on the scenario Indoor Ratio parameter
        :return: an UE location, i.e., 'Indoor', 'Outdoor', or 'Car'
        """
        if (self.scenario == 'UMi') or (self.scenario == 'UMa'):
            if self.random_generator_Location.random() < self.UE_indoor_ratio:
                return Location.Indoor
            else:
                return Location.Outdoor

        if self.scenario == 'RMa':
            # Todo: Need to check this
            if self.random_generator_Location.random() < self.UE_indoor_ratio:
                return Location.Indoor
            else:
                return Location.Car

    def computeGeometry(self):
        """
        Computes the 2D and 3D distances and the line-of-sight azimuth and zenith angles between all BSs and UEs.
        Results are stored in the class attributes 'dist2D_Matrix', 'dist3D_Matrix', 'los_azi_angle_rad_Matrix', and
        'los_zen_angle_rad_Matrix'.
        """
        nUE = self.number_of_ue
        nBS = self.number_of_bs
        self.los_azi_angle_rad_Matrix = np.zeros((nUE, nBS), dtype=float)
        self.los_zen_angle_rad_Matrix = np.zeros((nUE, nBS), dtype=float)
        self.dist2D_Matrix = np.zeros((nUE, nBS), dtype=float)
        self.dist3D_Matrix = np.zeros((nUE, nBS), dtype=float)

        for ue in self.UEs:
            for bs in self.BSs:
                # Compute 2D distance between UE and BS
                dist_2D = dist2d(bs, ue)
                dist_3D = dist3d(bs, ue)

                # Compute relative UE position (assuming BS is the origin)
                xy_position = np.array([(ue.pos_x - bs.pos_x), (ue.pos_y - bs.pos_y)])

                # Compute the azimuth angle
                if xy_position[1] >= 0:
                    az_angle_rad_gcs = np.arccos(xy_position[0] / dist_2D)
                else:
                    az_angle_rad_gcs = 2 * np.pi - np.arccos(xy_position[0] / dist_2D)

                # Compute relative BS height
                h_e = bs.height - ue.height
                ze_angle_rad_gcs = np.pi / 2  # If h_e == 0
                if h_e > 0:
                    ze_angle_rad_gcs = np.pi - np.arctan(dist_2D / h_e)
                elif h_e < 0:
                    ze_angle_rad_gcs = - np.arctan(dist_2D / h_e)

                self.dist2D_Matrix[ue.ID][bs.ID] = dist_2D
                self.dist3D_Matrix[ue.ID][bs.ID] = dist_3D
                self.los_azi_angle_rad_Matrix[ue.ID][bs.ID] = az_angle_rad_gcs
                self.los_zen_angle_rad_Matrix[ue.ID][bs.ID] = ze_angle_rad_gcs

    def computeLOS(self):
        """
        Computes line-of-sight conditions between BSs and UEs.
        Results are stored in the class attribute 'los_Matrix'.
        """
        self.los_Matrix = np.full((self.number_of_ue, self.number_of_bs), LOS(0))
        for bs in self.BSs:
            for ue in self.UEs:
                self.los_Matrix[ue.ID][bs.ID] = self.LineOfSight(bs, ue)

    def Pathloss(self, bs: BaseStation, sec: Sector, ue: UserEquipment):
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
        dist_2D = self.dist2D_Matrix[ue.ID][bs.ID]
        dist_3D = self.dist3D_Matrix[ue.ID][bs.ID]
        los = self.los_Matrix[ue.ID][bs.ID]
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

                # Break point distance (Table 7.4.1.1, Note 5)
                d_bp = 2 * np.pi * bs.height * ue.height * fc * 1_000_000_000 / c

                if los == LOS.LOS:
                    # Compute PL_RMa-LOS
                    if (10 <= dist_2D) and (dist_2D <= d_bp):
                        pathloss = 20 * np.log10(40 * np.pi * dist_3D * fc / 3) \
                                   + min(0.03 * (self.average_building_heigh ** 1.72), 10) * np.log10(dist_3D) \
                                   - min(0.044 * (self.average_building_heigh ** 1.72), 14.77) \
                                   + 0.002 * np.log10(self.average_building_heigh) * dist_3D
                        sigma_sf = 4
                    elif (d_bp <= dist_2D) and (dist_2D <= 10000):
                        pathloss = 20 * np.log10(40 * np.pi * d_bp * fc / 3) \
                                   + min(0.03 * (self.average_building_heigh ** 1.72), 10) * np.log10(d_bp) \
                                   - min(0.044 * (self.average_building_heigh ** 1.72), 14.77) \
                                   + 0.002 * np.log10(self.average_building_heigh) * d_bp \
                                   + 40 * np.log10(dist_3D / d_bp)
                        sigma_sf = 6
                    else:
                        # TODO: remove
                        pathloss = np.inf
                        sigma_sf = 0
                        # print(f'UE-BS distance: {dist_2D}')
                        # raise 'Invalid range for UE-BS distance'

                if LOS.NLOS:
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
                        sigma_sf = 0
                        # print(f'UE-BS distance: {dist_2D}')
                        # raise ('Invalid range for UE-BS distance')

                    # Compute PL_RMa-NLOS
                    PL_RMa_NLOS = 161.04 - 7.1 * np.log10(self.average_street_width) \
                                  + 7.5 * np.log10(self.average_building_heigh) \
                                  - (24.37 - 3.7 * ((self.average_building_heigh / bs.height) ** 2)) * np.log10(
                        bs.height) \
                                  + (43.42 - 3.1 * np.log10(bs.height)) * (np.log10(dist_3D) - 3) \
                                  + 20 * np.log10(fc) \
                                  - (3.2 * (np.log10(11.75 * ue.height) ** 2) - 4.97)

                    pathloss = max(PL_RMa_LOS, PL_RMa_NLOS)
                    sigma_sf = 8

            # Pathloss for Urban Macro scenario
            if self.scenario == 'UMa':
                if not ((1.5 <= ue.height) and (ue.height <= 22.5)):
                    raise "UE height outside the pathloss formula's applicability range"
                if not (bs.height == 25):
                    raise "BS height is not the default value"  # Todo: need to check for correction formulas

                # Breakpoint Distance (Table 7.4.1.1, Note 1)
                C = 0
                if ue.height < 13:
                    C = 0
                elif (13 <= ue.height) and (ue.height <= 23):
                    if dist_2D <= 18:
                        g = 0
                    else:
                        g = (5 / 4) * ((dist_2D / 100) ** 3) * np.exp(-dist_2D / 150)
                    C = (((ue.height - 13) / 10) ** 1.5) * g

                if self.random_generator_PL.random() < 1 / (1 + C):
                    h_e = 1
                else:
                    h_e = self.random_generator_PL.choice(np.arange(12, ue.height - 1.5, 3))

                d_bp = 4 * (bs.height - h_e) * (ue.height - h_e) * fc * 1_000_000_000 / c

                # Pathloss computation for LOS
                if los == LOS.LOS:
                    # Compute PL_UMa-LOS
                    if (10 <= dist_2D) and (dist_2D <= d_bp):
                        pathloss = 28.0 + 22 * np.log10(dist_3D) + 20 * np.log10(fc)
                        sigma_sf = 4.0
                    elif (d_bp <= dist_2D) and (dist_2D <= 5000):
                        pathloss = 28.0 + 40 * np.log10(dist_3D) + 20 * np.log10(fc) \
                                   - 9 * np.log10(d_bp ** 2 + (bs.height - ue.height) ** 2)
                        sigma_sf = 4.0
                    elif dist_2D > 5000:  # Todo: Is this valid?
                        pathloss = np.inf
                        sigma_sf = 4.0
                    else:
                        # TODO: remove
                        pathloss = np.inf
                        sigma_sf = 0
                        # print(f'UE-BS distance: {dist_2D}')
                        # raise 'Invalid range for UE-BS distance'

                # Pathloss computation for NLOS
                if los == LOS.NLOS:
                    # Compute PL_UMa-LOS
                    if (10 <= dist_2D) and (dist_2D <= d_bp):
                        PL_UMa_LOS = 28.0 + 22 * np.log10(dist_3D) + 20 * np.log10(fc)
                    elif (d_bp <= dist_2D) and (dist_2D <= 5000):
                        PL_UMa_LOS = 28.0 + 40 * np.log10(dist_3D) + 20 * np.log10(fc) \
                                     - 9 * np.log10(d_bp ** 2 + (bs.height - ue.height) ** 2)
                    elif dist_2D > 5000:  # Todo: Is this valid?
                        # TODO: remove
                        PL_UMa_LOS = np.inf
                        sigma_sf = 0
                        # print(f'UE-BS distance: {dist_2D}')
                        # raise 'Invalid range for UE-BS distance'

                    # Compute PL_UMa-NLOS
                    PL_UMa_NLOS = 13.54 + 39.08 * np.log10(dist_3D) + 20 * np.log10(fc) - 0.6 * (ue.height - 1.5)
                    # if (dist_3D > 0) and ((ue.height - 1.5) > 0):
                    #     PL_UMa_NLOS = 13.54 + 39.08 * np.log10(dist_3D) + 20 * np.log10(fc) - 0.6 * np.log10(
                    #         ue.height - 1.5)
                    # else:
                    #     PL_UMa_NLOS = np.inf

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

                # Breakpoint Distance (Table 7.4.1.1, Note 1)
                h_e = 1.0  # meter
                d_bp = 4 * (bs.height - h_e) * (ue.height - h_e) * fc * 1_000_000_000 / c

                # Pathloss computation for LOS
                if los == LOS.LOS:
                    # Compute PL_UMi-LOS
                    if (10 <= dist_2D) and (dist_2D <= d_bp):
                        pathloss = 32.4 + 21 * np.log10(dist_3D) + 20 * np.log10(fc)
                        sigma_sf = 4.0
                    elif (d_bp <= dist_2D) and (dist_2D <= 5000):
                        pathloss = 32.4 + 40 * np.log10(dist_3D) + 20 * np.log10(fc) \
                                   - 9.5 * np.log10(d_bp ** 2 + (bs.height - ue.height) ** 2)
                        sigma_sf = 4.0
                    elif dist_2D > 5000:  # Todo: Is this valid?
                        pathloss = np.inf
                        sigma_sf = 4.0
                    else:
                        # TODO: remove
                        pathloss = np.inf
                        sigma_sf = 0
                        # print(f'UE-BS distance: {dist_2D}')
                        # raise 'Invalid range for UE-BS distance'

                # Pathloss computation for NLOS
                if los == LOS.NLOS:

                    # Compute PL_UMi-LOS
                    if (10 <= dist_2D) and (dist_2D <= d_bp):
                        PL_UMi_LOS = 32.4 + 21 * np.log10(dist_3D) + 20 * np.log10(fc)
                    elif (d_bp <= dist_2D) and (dist_2D <= 5000):
                        PL_UMi_LOS = 32.4 + 40 * np.log10(dist_3D) + 20 * np.log10(fc) \
                                     - 9.5 * np.log10(d_bp ** 2 + (bs.height - ue.height) ** 2)
                    elif dist_2D > 5000:  # Todo: Is this valid?
                        PL_UMi_LOS = np.inf
                    else:
                        raise 'Invalid range for UE-BS distance'

                    # Compute PL_UMi-NLOS
                    try:
                        PL_UMi_NLOS = 35.3 * np.log10(dist_3D) \
                                      + 22.4 + 21.3 * np.log10(fc) \
                                      - 0.3 * (ue.height - 1.5)
                        pathloss = max(PL_UMi_LOS, PL_UMi_NLOS)
                        sigma_sf = 7.82
                    except:
                        # Optional
                        pathloss = 32.4 + 20 * np.log10(fc) + 31.9 * np.log10(dist_3D)
                        sigma_sf = 8.2

            # Additional Pathloss terms for Indoor UEs
            if ue.location == Location.Indoor:
                # Todo: implement the penetration loss model - See TR 38.901 - 7.4.3
                PL_tw = 0
                PL_in = 0
                sigma_p_sq = 0
                pathloss = pathloss + PL_tw + PL_in + self.random_generator_PL.normal(scale=np.sqrt(sigma_p_sq))

            # Additional Pathloss terms for in Car UEs
            if ue.location == Location.Car:
                mu = 9.0  # 20 for metalized window
                sigma_p = 5.0
                pathloss = pathloss + self.random_generator_PL.normal(loc=mu, scale=sigma_p)

            # Final Pathloss with shadow fading
            pathloss = pathloss + self.random_generator_PL.lognormal(sigma=sigma_sf)

        # Sectorization
        # Todo: incorporate ZoA/ZoD
        # Todo: this should be removed once the antenna patterns are defined.
        # For now, this is doing simple sectorization
        angle_diff = self.los_azi_angle_rad_Matrix[ue.ID][bs.ID] - np.deg2rad(sec.orientation)
        if angle_diff > np.pi:
            angle_diff = angle_diff - 2 * np.pi
        if angle_diff < - np.pi:
            angle_diff = angle_diff + 2 * np.pi
        if np.abs(angle_diff) <= (np.deg2rad(sec.sector_width / 2)):
            pathloss = pathloss
        else:
            # pathloss = np.inf
            pathloss = pathloss + 30    # Add 30 dB (which is the minimum antenna gain as per the standard)

        return pathloss, sigma_sf

    def NetworkPathloss(self, BS_list: list[BaseStation] = None, UE_list: list[UserEquipment] = None):
        """
        Computes the Pathloss and Line of Sight parameters for combinations of BSs and UEs
        :param BS_list: list of BSs
        :param UE_list: list of UEs
        :return: update Network attributes pathlossMatrix and shadowFadingMatrix
        """
        if BS_list is None:
            BS_list = self.BSs

        if UE_list is None:
            UE_list = self.UEs

        nBS = len(BS_list)
        nUE = len(UE_list)
        nSectors = sum([bs.number_of_sectors for bs in BS_list])
        self.pathlossMatrix = np.zeros((nUE, nSectors))
        self.shadowFadingMatrix = np.zeros((nUE, nSectors))
        for bs_ind, bs in enumerate(BS_list):
            for sec_ind, sec in enumerate(bs.sector):
                for eu_ind, ue in enumerate(UE_list):
                    # Pathloss is determined for each Sector and UE pair
                    pathloss, sigma_sf = self.Pathloss(bs=bs, sec=sec, ue=ue)
                    self.pathlossMatrix[ue.ID][sec.ID] = pathloss
                    self.shadowFadingMatrix[ue.ID][sec.ID] = sigma_sf

    def large_scale_parameter_correlation_method_two(self):
        # Method 2: Create grid; 2D-filter normal iid points in the grid; use filtered values to compute LSP

        ################################################################################################################
        # Get geometry and design the grid
        ################################################################################################################
        # Get UE positions:
        ue_position = np.zeros((len(self.UEs), 2))
        for ue_idx, ue in enumerate(self.UEs):
            ue_position[ue_idx][:] = [ue.pos_x, ue.pos_y]
            ue.LSP = [None] * len(self.BSs)

        x_min, y_min = np.min(ue_position, axis=0)
        x_max, y_max = np.max(ue_position, axis=0)
        delta_d = .5

        x = np.arange(x_min - delta_d, x_max + delta_d, delta_d)
        y = np.arange(y_min - delta_d, y_max + delta_d, delta_d)
        Nx = len(x)
        Ny = len(y)

        # Filter grid
        D = 5  # Filter Length
        if D % 2 == 0:
            D = D + 1
        xf = np.linspace(-delta_d * (D - 1) / 2, delta_d * (D - 1) / 2, num=D)
        yf = np.linspace(-delta_d * (D - 1) / 2, delta_d * (D - 1) / 2, num=D)
        xv, yv = np.meshgrid(xf, yf)
        d = np.sqrt(xv ** 2 + yv ** 2)
        # Todo: for efficiency, this should select D according to delta_m (when delta_m is small eg. 3, D has to be
        #  larger eg. 11; when when delta_m is small eg. 50, D can be smaller eg. 5

        for los in [LOS.LOS, LOS.NLOS]:
            C, Delta_m, LSP = self.generateLargeScaleParamsCorrelation(los=los)
            Q = sqrtm(C)

            for bs in self.BSs:
                print(f'LOS:{los.value} - BS:{bs.ID}')
                normal_epsilon = np.random.normal(0, 1, (Nx, Ny))
                correlated_epsilon = np.zeros((len(LSP), len(self.UEs)))

                for m, lsp in enumerate(LSP):
                    alpha = 1 / Delta_m[lsp]
                    filter_coeff = (alpha ** d) * np.exp(-alpha * d)
                    filter_coeff = filter_coeff / sum(sum(filter_coeff))
                    # filter_coeff = filter_coeff / D

                    # filtered_grid = convolve2d(np.random.normal(0, 1, (Nx + 2*D, Ny + 2*D)), filter_coeff, mode='same',
                    #                            boundary='wrap', fillvalue=0)
                    filtered_grid = convolve2d(normal_epsilon, filter_coeff, mode='same', boundary='wrap',
                                               fillvalue=0)
                    # plt.subplots()
                    # # plt.pcolormesh(x, y, filtered_grid.T)
                    # plt.pcolormesh(xf, yf, filter_coeff.T)
                    # plt.colorbar()
                    # plt.show()
                    interp = RegularGridInterpolator((x, y), filtered_grid, method='linear')
                    correlated_epsilon[m, :] = interp(ue_position)

                for ue_idx, ue in enumerate(self.UEs):
                    if self.getLOS(self.los_Matrix[ue.ID][bs.ID]) == los:
                        s_tilde = np.dot(Q, correlated_epsilon[:, ue_idx])
                        correlated_TLSP = dict(zip(LSP, s_tilde))
                        ue.LSP[bs.ID] = self.generateLinkLPS(bs, ue, correlated_TLSP)
                        # ue.SSP[bs.ID] = self.generateSmallScaleParams_link(bs=bs, sec=[], ue=ue)
                        self.generateSmallScaleParams_link(bs=bs, ue=ue)

    def generateLargeScaleParamsCorrelation(self, los):
        """"
        Generates the Large Scale Parameter correlation matrices and the correlation distance.
        """
        if self.scenario == 'UMi':
            if los == LOS.LOS:
                # Cross-Correlations
                ASD_vs_DS = 0.5
                ASA_vs_DS = 0.8
                ASA_vs_SF = -0.4
                ASD_vs_SF = -0.5
                DS_vs_SF = -0.4
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

                # Correlation distance in the horizontal plane [m]
                corr_dist_h_plane_DS = 7
                corr_dist_h_plane_ASD = 8
                corr_dist_h_plane_ASA = 8
                corr_dist_h_plane_SF = 10
                corr_dist_h_plane_K = 15
                corr_dist_h_plane_ZSA = 12
                corr_dist_h_plane_ZSD = 12

            if los == LOS.NLOS:
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

                # Correlation distance in the horizontal plane [m]
                corr_dist_h_plane_DS = 10
                corr_dist_h_plane_ASD = 10
                corr_dist_h_plane_ASA = 9
                corr_dist_h_plane_SF = 13
                corr_dist_h_plane_K = None
                corr_dist_h_plane_ZSA = 10
                corr_dist_h_plane_ZSD = 10

            if los == LOS.O2I:
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

                # Correlation distance in the horizontal plane [m]
                corr_dist_h_plane_DS = 10
                corr_dist_h_plane_ASD = 11
                corr_dist_h_plane_ASA = 17
                corr_dist_h_plane_SF = 7
                corr_dist_h_plane_K = None
                corr_dist_h_plane_ZSA = 25
                corr_dist_h_plane_ZSD = 25

        if self.scenario == 'UMa':
            if los == LOS.LOS:
                # Cross-Correlations
                ASD_vs_DS = 0.4
                ASA_vs_DS = 0.8
                ASA_vs_SF = -0.5
                ASD_vs_SF = -0.5
                DS_vs_SF = -0.4
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

                # Correlation distance in the horizontal plane [m]
                corr_dist_h_plane_DS = 30
                corr_dist_h_plane_ASD = 18
                corr_dist_h_plane_ASA = 15
                corr_dist_h_plane_SF = 37
                corr_dist_h_plane_K = 12
                corr_dist_h_plane_ZSA = 15
                corr_dist_h_plane_ZSD = 15

            if los == LOS.NLOS:
                # Cross-Correlations
                ASD_vs_DS = 0.4
                ASA_vs_DS = 0.8
                ASA_vs_SF = -0.5
                ASD_vs_SF = -0.5
                DS_vs_SF = -0.4
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

                # Correlation distance in the horizontal plane [m]
                corr_dist_h_plane_DS = 30
                corr_dist_h_plane_ASD = 18
                corr_dist_h_plane_ASA = 15
                corr_dist_h_plane_SF = 37
                corr_dist_h_plane_K = 12
                corr_dist_h_plane_ZSA = 15
                corr_dist_h_plane_ZSD = 15

            if los == LOS.O2I:
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

                # Correlation distance in the horizontal plane [m]
                corr_dist_h_plane_DS = 10
                corr_dist_h_plane_ASD = 11
                corr_dist_h_plane_ASA = 17
                corr_dist_h_plane_SF = 7
                corr_dist_h_plane_K = None
                corr_dist_h_plane_ZSA = 25
                corr_dist_h_plane_ZSD = 25

        if self.scenario == 'RMa':
            if los == LOS.LOS:
                # Cross-Correlations
                ASD_vs_DS = 0.0
                ASA_vs_DS = 0.0
                ASA_vs_SF = 0.0
                ASD_vs_SF = 0.0
                DS_vs_SF = -0.5
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

                # Correlation distance in the horizontal plane [m]
                corr_dist_h_plane_DS = 50
                corr_dist_h_plane_ASD = 25
                corr_dist_h_plane_ASA = 35
                corr_dist_h_plane_SF = 37
                corr_dist_h_plane_K = 40
                corr_dist_h_plane_ZSA = 15
                corr_dist_h_plane_ZSD = 15

            if los == LOS.NLOS:
                # Cross-Correlations
                ASD_vs_DS = -0.4
                ASA_vs_DS = 0.0
                ASA_vs_SF = 0.0
                ASD_vs_SF = 0.6
                DS_vs_SF = -0.5
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

                # Correlation distance in the horizontal plane [m]
                corr_dist_h_plane_DS = 36
                corr_dist_h_plane_ASD = 30
                corr_dist_h_plane_ASA = 40
                corr_dist_h_plane_SF = 120
                corr_dist_h_plane_K = None
                corr_dist_h_plane_ZSA = 50
                corr_dist_h_plane_ZSD = 50

            if los == LOS.O2I:
                # Cross-Correlations
                ASD_vs_DS = 0.0
                ASA_vs_DS = 0.0
                ASA_vs_SF = 0.0
                ASD_vs_SF = 0.0
                DS_vs_SF = 0.0
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

                # Correlation distance in the horizontal plane [m]
                corr_dist_h_plane_DS = 36
                corr_dist_h_plane_ASD = 30
                corr_dist_h_plane_ASA = 40
                corr_dist_h_plane_SF = 120
                corr_dist_h_plane_K = None
                corr_dist_h_plane_ZSA = 50
                corr_dist_h_plane_ZSD = 50

        if self.scenario == 'InH':
            if los == LOS.LOS:
                # Cross-Correlations
                ASD_vs_DS = 0.6
                ASA_vs_DS = 0.8
                ASA_vs_SF = -0.5
                ASD_vs_SF = -0.4
                DS_vs_SF = -0.8
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

                # Correlation distance in the horizontal plane [m]
                corr_dist_h_plane_DS = 8
                corr_dist_h_plane_ASD = 7
                corr_dist_h_plane_ASA = 5
                corr_dist_h_plane_SF = 10
                corr_dist_h_plane_K = 4
                corr_dist_h_plane_ZSA = 4
                corr_dist_h_plane_ZSD = 4

            if los == LOS.NLOS:
                # Cross-Correlations
                ASD_vs_DS = 0.4
                ASA_vs_DS = 0.0
                ASA_vs_SF = -0.4
                ASD_vs_SF = 0.0
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
                ZSD_vs_DS = -0.27
                ZSA_vs_DS = -0.06
                ZSD_vs_ASD = 0.35
                ZSA_vs_ASD = 0.23
                ZSD_vs_ASA = -0.08
                ZSA_vs_ASA = 0.43
                ZSD_vs_ZSA = 0.42

                # Correlation distance in the horizontal plane [m]
                corr_dist_h_plane_DS = 5
                corr_dist_h_plane_ASD = 3
                corr_dist_h_plane_ASA = 3
                corr_dist_h_plane_SF = 6
                corr_dist_h_plane_K = None
                corr_dist_h_plane_ZSA = 4
                corr_dist_h_plane_ZSD = 4

        if los == LOS.LOS:
            LSP = ['SF', 'K', 'DS', 'ASD', 'ASA', 'ZSD', 'ZSA']
            C = np.array([[0, SF_vs_K, DS_vs_SF, ASD_vs_SF, ASA_vs_SF, ZSD_vs_SF, ZSA_vs_SF],
                          [0, 0, DS_vs_K, ASD_vs_K, ASA_vs_K, ZSD_vs_K, ZSA_vs_K],
                          [0, 0, 0, ASD_vs_DS, ASA_vs_DS, ZSD_vs_DS, ZSA_vs_DS],
                          [0, 0, 0, 0, ASD_vs_ASA, ZSD_vs_ASD, ZSA_vs_ASD],
                          [0, 0, 0, 0, 0, ZSD_vs_ASA, ZSA_vs_ASA],
                          [0, 0, 0, 0, 0, 0, ZSD_vs_ZSA],
                          [0, 0, 0, 0, 0, 0, 0]])
            delta_m = {'SF': corr_dist_h_plane_SF,
                       'K': corr_dist_h_plane_K,
                       'DS': corr_dist_h_plane_DS,
                       'ASD': corr_dist_h_plane_ASD,
                       'ASA': corr_dist_h_plane_ASA,
                       'ZSD': corr_dist_h_plane_ZSD,
                       'ZSA': corr_dist_h_plane_ZSA
                       }
        else:
            LSP = ['SF', 'DS', 'ASD', 'ASA', 'ZSD', 'ZSA']
            C = np.array([[0, DS_vs_SF, ASD_vs_SF, ASA_vs_SF, ZSD_vs_SF, ZSA_vs_SF],
                          [0, 0, ASD_vs_DS, ASA_vs_DS, ZSD_vs_DS, ZSA_vs_DS],
                          [0, 0, 0, ASD_vs_ASA, ZSD_vs_ASD, ZSA_vs_ASD],
                          [0, 0, 0, 0, ZSD_vs_ASA, ZSA_vs_ASA],
                          [0, 0, 0, 0, 0, ZSD_vs_ZSA],
                          [0, 0, 0, 0, 0, 0]])
            delta_m = {'SF': corr_dist_h_plane_SF,
                       'DS': corr_dist_h_plane_DS,
                       'ASD': corr_dist_h_plane_ASD,
                       'ASA': corr_dist_h_plane_ASA,
                       'ZSD': corr_dist_h_plane_ZSD,
                       'ZSA': corr_dist_h_plane_ZSA
                       }

        C = C + C.T + np.eye(len(LSP))

        return C, delta_m, LSP

    def generateLinkLPS(self, bs: BaseStation, ue: UserEquipment, correlated_TLSP):

        # Large Scale Parameters (LSP) for different BS-UE links are uncorrelated, but the LSPs for links from co-sited
        # sectors to a UE are the same. In addition, LSPs for the links of UEs on different floors are uncorrelated.

        fc = 3.5  # GHz  # Todo: figure if get from sector since LSPs should be the same for all sectors within a BS
        los = self.getLOS(self.los_Matrix[ue.ID][bs.ID])

        if self.scenario == 'UMi':
            # Frequency correction - see NOTE 7 from Table 7.5-6
            if fc < 2.0:
                fc = 2.0

            if los == LOS.LOS:
                # Delay Spread (DS)
                mu_lg_DS = -0.24 * np.log10(1 + fc) - 7.14
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

                # ZOD Spread (ZSD)
                mu_lg_ZSD = np.maximum(-0.21, -14.8 * (self.dist2D_Matrix[ue.ID][bs.ID] / 1000) + 0.01 * np.abs(
                    ue.height - 1.5) + 0.83)
                sigma_lg_ZSD = 0.35
                mu_offset_ZOD = 0

                # Shadow Fading (SF) [dB]
                sigma_SF = 4

                # K Factor (K) [dB]
                mu_K = 9
                sigma_K = 5

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

            if los == LOS.NLOS:
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

                # ZOD Spread (ZSD)
                mu_lg_ZSD = np.maximum(-0.5, -3.1 * (self.dist2D_Matrix[ue.ID][bs.ID] / 1000) + 0.01 * np.maximum(
                    ue.height - bs.height, 0) + 0.2)
                sigma_lg_ZSD = 0.35
                mu_offset_ZOD = -10 ** (-1.5 * np.log10(np.maximum(10, self.dist2D_Matrix[ue.ID][bs.ID])) + 3.3)

                # Shadow Fading (SF) [dB]
                sigma_SF = 7.84

                # K Factor (K) [dB]
                mu_K = None
                sigma_K = None

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

            if los == LOS.O2I:
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

        if self.scenario == 'UMa':
            # Frequency correction - see NOTE 6 from Table 7.5-6 Part-1
            if fc < 6.0:
                fc = 6.0

            if los == 'LOS':
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

                # ZOD Spread (ZSD)
                mu_lg_ZSD = np.maximum(-0.5,
                                       -2.1 * (self.dist2D_Matrix[ue.ID][bs.ID] / 1000) - 0.01 * (
                                               ue.height - 1.5) + 0.75)
                sigma_lg_ZSD = 0.4
                mu_offset_ZOD = 0

                # Shadow Fading (SF) [dB]
                sigma_SF = 4

                # K Factor (K) [dB]
                mu_K = 9
                sigma_K = 3.5

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

            if los == 'NLOS':
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

                # ZOD Spread (ZSD)
                mu_lg_ZSD = np.maximum(-0.5,
                                       -2.1 * (self.dist2D_Matrix[ue.ID][bs.ID] / 1000) - 0.01 * (
                                               ue.height - 1.5) + 0.9)
                sigma_lg_ZSD = 0.49
                mu_offset_ZOD = (7.66 * np.log10(fc) - 5.96) - 10 ** (
                        (0.208 * np.log10(fc) - 0.782) * np.log10(np.maximum(25, self.dist2D_Matrix[ue.ID][bs.ID]))
                        - 0.13 * np.log10(fc) + 2.03 - 0.07 * (ue.height - 1.5))

                # Shadow Fading (SF) [dB]
                sigma_SF = 6

                # K Factor (K) [dB]
                mu_K = 9
                sigma_K = 3.5

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

            if los == 'O2I':
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

        if self.scenario == 'RMa':
            if los == 'LOS':
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

                # ZOD Spread (ZSD)
                mu_lg_ZSD = np.maximum(-1, -0.17 * (self.dist2D_Matrix[ue.ID][bs.ID] / 1000) - 0.01 * (
                        ue.height - bs.height) + 0.22)
                sigma_lg_ZSD = 0.34
                mu_offset_ZOD = 0

                # Shadow Fading (SF) [dB]
                sigma_SF = 4  # or 6
                # Todo: do checks (may vary)

                # K Factor (K) [dB]
                mu_K = 7
                sigma_K = 4

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

            if los == 'NLOS':
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

                # ZOD Spread (ZSD)
                mu_lg_ZSD = np.maximum(-1, -0.19 * (self.dist2D_Matrix[ue.ID][bs.ID] / 1000) - 0.01 * (
                        ue.height - bs.height) + 0.28)
                sigma_lg_ZSD = 0.30
                mu_offset_ZOD = np.arctan((35 - 3.5) / self.dist2D_Matrix[ue.ID][bs.ID]) - np.arctan(
                    (35 - 1.5) / self.dist2D_Matrix[ue.ID][bs.ID])

                # Shadow Fading (SF) [dB]
                sigma_SF = 8

                # K Factor (K) [dB]
                mu_K = None
                sigma_K = None

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

            if los == 'O2I':
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

                # ZOD Spread (ZSD)
                mu_lg_ZSD = np.maximum(-1, -0.19 * (self.dist2D_Matrix[ue.ID][bs.ID] / 1000) - 0.01 * (
                        ue.height - bs.height) + 0.28)
                sigma_lg_ZSD = 0.30
                mu_offset_ZOD = np.arctan((35 - 3.5) / self.dist2D_Matrix[ue.ID][bs.ID]) - np.arctan(
                    (35 - 1.5) / self.dist2D_Matrix[ue.ID][bs.ID])

                # Shadow Fading (SF) [dB]
                sigma_SF = 8

                # K Factor (K) [dB]
                mu_K = None
                sigma_K = None

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

        if self.scenario == 'InH':
            # Frequency correction - see NOTE 6 from Table 7.5-6 Part-2
            if fc < 6.0:
                fc = 6.0

            if los == 'LOS':
                # Delay Spread (DS)
                mu_lg_DS = -0.01 * np.log10(1 + fc) - 7.692
                sigma_lg_DS = 0.18

                # AOD Spread (ASD)
                mu_lg_ASD = 1.60
                sigma_lg_ASD = 0.18

                # AOA Spread (ASA)
                mu_lg_ASA = -0.19 * np.log10(1 + fc) + 1.781
                sigma_lg_ASA = 0.12 * np.log10(1 + fc) + 0.119

                # ZOA Spread (ZSA)
                mu_lg_ZSA = -0.26 * np.log10(1 + fc) + 1.44
                sigma_lg_ZSA = -0.04 * np.log10(1 + fc) + 0.264

                # ZOD Spread (ZSD)
                mu_lg_ZSD = -1.43 * np.log10(1 + fc) + 2.228
                sigma_lg_ZSD = 0.13 * np.log10(1 + fc) + 0.30
                mu_offset_ZOD = 0

                # Shadow Fading (SF) [dB]
                sigma_SF = 3

                # K Factor (K) [dB]
                mu_K = 7
                sigma_K = 4

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

            if los == 'NLOS':
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

                # ZOD Spread (ZSD)
                mu_lg_ZSD = 1.08
                sigma_lg_ZSD = 0.36
                mu_offset_ZOD = 0

                # Shadow Fading (SF) [dB]
                sigma_SF = 8.03

                # K Factor (K) [dB]
                mu_K = None
                sigma_K = None

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

        # Generate DS
        DS = 10 ** (mu_lg_DS + sigma_lg_DS * correlated_TLSP['DS'])

        # Generate ASA
        ASA = 10 ** (mu_lg_ASA + sigma_lg_ASA * correlated_TLSP['ASA'])
        ASA = min(ASA, 104.0)

        # Generate ASD
        ASD = 10 ** (mu_lg_ASD + sigma_lg_ASD * correlated_TLSP['ASD'])
        ASD = min(ASD, 104.0)

        # Generate ZSA
        ZSA = 10 ** (mu_lg_ZSA + sigma_lg_ZSA * correlated_TLSP['ZSA'])
        ZSA = min(ZSA, 52.0)

        # Generate ZSD
        ZSD = 10 ** (mu_lg_ZSD + sigma_lg_ZSD * correlated_TLSP['ZSD'])
        ZSD = min(ZSD, 52.0)
        # tODO: SEE NOTE 4-5 BELOW TABLE 7.5.11

        # Generate SF
        SF = sigma_SF * correlated_TLSP['SF']

        if los == 'LOS':
            # Generate K
            K = mu_K + sigma_K * correlated_TLSP['K']

            LSP = {'DS': DS, 'ASA': ASA, 'ASD': ASD, 'ZSA': ZSA, 'ZSD': ZSD, 'K': K, 'SF': SF,
                   'N': N, 'M': M, 'r_tau': r_tau, 'c_DS': c_DS, 'c_ASA': c_ASA, 'c_ASD': c_ASD, 'c_ZSA': c_ZSA,
                   'xi': xi, 'mu_offset_ZOD': mu_offset_ZOD, 'mu_lg_ZSD': mu_lg_ZSD, 'mu_XPR': mu_XPR,
                   'sigma_xpr': sigma_xpr}
        else:
            LSP = {'DS': DS, 'ASA': ASA, 'ASD': ASD, 'ZSA': ZSA, 'ZSD': ZSD, 'SF': SF,
                   'N': N, 'M': M, 'r_tau': r_tau, 'c_DS': c_DS, 'c_ASA': c_ASA, 'c_ASD': c_ASD, 'c_ZSA': c_ZSA,
                   'xi': xi, 'mu_offset_ZOD': mu_offset_ZOD, 'mu_lg_ZSD': mu_lg_ZSD, 'mu_XPR': mu_XPR,
                   'sigma_xpr': sigma_xpr}

        return LSP

    def generateSmallScaleParams_link(self, bs: BaseStation, ue: UserEquipment):
        # Small Scale Parameter generation

        ################################################################################################################
        # Step 5: Generate cluster delays Tau_n:
        Xn = np.random.uniform(size=ue.LSP[bs.ID]['N'])
        cluster_delay = - ue.LSP[bs.ID]['r_tau'] * ue.LSP[bs.ID]['DS'] * np.log(Xn)
        cluster_delay = np.sort(cluster_delay - min(cluster_delay))

        if self.getLOS(self.los_Matrix[ue.ID][bs.ID]) == LOS.LOS:
            C_tau = 0.7705 - 0.0433 * ue.LSP[bs.ID]['K'] + 0.0002 * (ue.LSP[bs.ID]['K'] ** 2) + 0.000017 * (
                    ue.LSP[bs.ID]['K'] ** 3)
            cluster_delay_LOS = cluster_delay / C_tau

        ################################################################################################################
        # Step 6: Generate cluster powers P_n:
        P_n_notch = np.exp(
            -cluster_delay * ((ue.LSP[bs.ID]['r_tau'] - 1) / (ue.LSP[bs.ID]['r_tau'] * ue.LSP[bs.ID]['DS'])))
        P_n_notch = P_n_notch * (10 ** (- np.random.normal(loc=0.0, scale=ue.LSP[bs.ID]['xi']) / 10))

        P_n = P_n_notch / sum(P_n_notch)

        if self.getLOS(self.los_Matrix[ue.ID][bs.ID]) == LOS.LOS:
            K_R_linear = 10 ** (ue.LSP[bs.ID]['K'] / 10)
            P_n = (1 / (K_R_linear + 1)) * P_n
            P_n[0] = P_n[0] + (K_R_linear / (K_R_linear + 1))

        # Discard clusters with power less than -25 dB compared to the maximum cluster power
        P_n_dB = 10 * np.log10(P_n)
        # P_n_dB = P_n_dB[P_n_dB >= max(P_n_dB) - 25]
        P_n = P_n[P_n_dB >= max(P_n_dB) - 25]

        # Power per ray
        P_n_ray = P_n / ue.LSP[bs.ID]['M']

        updated_N_cluster = len(P_n)

        ################################################################################################################
        # Step 7: Generate arrival angles and departure angles for both azimuth and elevation
        # Todo: Check standard (perhaps instead of using updated_N_cluster I should use the M and M only has a finite
        #  set of values, so I wouldn't need the interpolation below)

        # Azimuth
        if updated_N_cluster == 4:
            C_phi_NLOS = 0.779
        elif updated_N_cluster == 5:
            C_phi_NLOS = 0.860
        elif updated_N_cluster == 6:  # Not in the standard - obtained from linear interpolation
            C_phi_NLOS = 0.913
        elif updated_N_cluster == 7:  # Not in the standard - obtained from linear interpolation
            C_phi_NLOS = 0.965
        elif updated_N_cluster == 8:
            C_phi_NLOS = 1.018
        elif updated_N_cluster == 9:  # Not in the standard - obtained from linear interpolation
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

        if self.getLOS(self.los_Matrix[ue.ID][bs.ID]) == LOS.LOS:
            C_phi = C_phi_NLOS * (1.1035 - 0.028 * ue.LSP[bs.ID]['K'] - 0.002 * (ue.LSP[bs.ID]['K'] ** 2) + 0.0001 * (
                    ue.LSP[bs.ID]['K'] ** 3))
        else:
            C_phi = C_phi_NLOS

        # Azimuth of Arrival
        Xn = np.random.choice([-1, 1], size=updated_N_cluster)
        Yn = np.random.normal(loc=0, scale=(ue.LSP[bs.ID]['ASA'] / 7), size=updated_N_cluster)
        phi_notch_n_AOA = 2 * (ue.LSP[bs.ID]['ASA'] / 1.4) * np.sqrt(-np.log(P_n / max(P_n))) / C_phi

        # Todo: Not sure if computed correctly - review (may need to add orientation to UEs)
        phi_LOS_AOA = np.rad2deg(self.los_azi_angle_rad_Matrix[ue.ID][bs.ID])

        if not (self.getLOS(self.los_Matrix[ue.ID][bs.ID]) == LOS.LOS):
            phi_n_AOA = Xn * phi_notch_n_AOA + Yn + phi_LOS_AOA
        else:
            phi_n_AOA = (Xn * phi_notch_n_AOA + Yn) - (Xn[0] * phi_notch_n_AOA[0] + Yn[0] - phi_LOS_AOA)

        alpha_m = np.array(
            [0.0447, -0.0447, 0.1413, -0.1413, 0.2492, -0.2492, 0.3715, -0.3715, 0.5129, -0.5129, 0.6797, -0.6797,
             0.8844, -0.8844, 1.1481, -1.1481, 1.5195, -1.5195, 2.1551, -2.1551])

        # phi_n_m_AOA = [None]*ue.LSP[bs.ID]['M']
        # for m in range(ue.LSP[bs.ID]['M']):
        #     phi_n_m_AOA[m] = phi_n_AOA + ue.LSP[bs.ID]['c_ASA'] * alpha_m[m]

        phi_n_m_AOA = np.zeros((updated_N_cluster, ue.LSP[bs.ID]['M']))
        for n in range(updated_N_cluster):
            phi_n_m_AOA[n] = phi_n_AOA[n] + ue.LSP[bs.ID]['c_ASA'] * alpha_m

        # Azimuth of Departure
        Xn = np.random.choice([-1, 1], size=updated_N_cluster)
        Yn = np.random.normal(loc=0, scale=(ue.LSP[bs.ID]['ASD'] / 7), size=updated_N_cluster)
        phi_notch_n_AOD = 2 * (ue.LSP[bs.ID]['ASD'] / 1.4) * np.sqrt(-np.log(P_n / max(P_n))) / C_phi

        # Todo: Not sure if computed correctly - review (may need to add orientation to UEs)
        phi_LOS_AOD = np.rad2deg(self.los_azi_angle_rad_Matrix[ue.ID][bs.ID])

        if not (self.getLOS(self.los_Matrix[ue.ID][bs.ID]) == LOS.LOS):
            phi_n_AOD = Xn * phi_notch_n_AOD + Yn + phi_LOS_AOD
        else:
            phi_n_AOD = (Xn * phi_notch_n_AOD + Yn) - (Xn[0] * phi_notch_n_AOD[0] + Yn[0] - phi_LOS_AOD)

        alpha_m = np.array([0.0447, -0.0447, 0.1413, -0.1413, 0.2492, -0.2492, 0.3715, -0.3715, 0.5129, -0.5129, 0.6797,
                            -0.6797, 0.8844, -0.8844, 1.1481, -1.1481, 1.5195, -1.5195, 2.1551, -2.1551])

        # phi_n_m_AOD = [None] * ue.LSP[bs.ID]['M']
        # for m in range(ue.LSP[bs.ID]['M']):
        #     phi_n_m_AOD[m] = phi_n_AOD + ue.LSP[bs.ID]['c_ASD'] * alpha_m[m]

        phi_n_m_AOD = np.zeros((updated_N_cluster, ue.LSP[bs.ID]['M']))
        for n in range(updated_N_cluster):
            phi_n_m_AOD[n] = phi_n_AOA[n] + ue.LSP[bs.ID]['c_ASD'] * alpha_m

        # Zenith
        # Todo: Check standard (perhaps instead of using updated_N_cluster I should use the M and M only has a finite
        #  set of values, so I wouldn't need the interpolation below)
        if updated_N_cluster == 4:  # Not in the standard - incorrect # TODO
            C_theta_NLOS = 0.779
        elif updated_N_cluster == 5:  # Not in the standard - incorrect # TODO
            C_theta_NLOS = 0.860
        elif updated_N_cluster == 6:  # Not in the standard - incorrect # TODO
            C_theta_NLOS = 0.913
        elif updated_N_cluster == 7:  # Not in the standard - incorrect # TODO
            C_theta_NLOS = 0.965
        elif updated_N_cluster == 8:
            C_theta_NLOS = 0.889
        elif updated_N_cluster == 9:  # Not in the standard - incorrect # TODO
            C_theta_NLOS = 1.054
        elif updated_N_cluster == 10:
            C_theta_NLOS = 0.957
        elif updated_N_cluster == 11:
            C_theta_NLOS = 1.031
        elif updated_N_cluster == 12:
            C_theta_NLOS = 1.104
        elif updated_N_cluster == 13:  # Not in the standard - incorrect # TODO
            C_theta_NLOS = 1.168
        elif updated_N_cluster == 14:  # Not in the standard - incorrect # TODO
            C_theta_NLOS = 1.190
        elif updated_N_cluster == 15:
            C_theta_NLOS = 1.1088
        elif updated_N_cluster == 16:  # Not in the standard - incorrect # TODO
            C_theta_NLOS = 1.226
        elif updated_N_cluster == 17:  # Not in the standard - incorrect # TODO
            C_theta_NLOS = 1.242
        elif updated_N_cluster == 18:  # Not in the standard - incorrect # TODO
            C_theta_NLOS = 1.257
        elif updated_N_cluster == 19:
            C_theta_NLOS = 1.184
        elif updated_N_cluster == 20:
            C_theta_NLOS = 1.178
        elif updated_N_cluster == 21:  # Not in the standard - incorrect # TODO
            C_theta_NLOS = 1.303
        elif updated_N_cluster == 22:  # Not in the standard - incorrect # TODO
            C_theta_NLOS = 1.317
        elif updated_N_cluster == 23:  # Not in the standard - incorrect # TODO
            C_theta_NLOS = 1.330
        elif updated_N_cluster == 24:  # Not in the standard - incorrect # TODO
            C_theta_NLOS = 1.344
        elif updated_N_cluster == 25:
            C_theta_NLOS = 1.282
        else:
            raise "Invalid number of clusters"

        if self.getLOS(self.los_Matrix[ue.ID][bs.ID]) == LOS.LOS:
            C_theta = C_theta_NLOS * (
                    1.3086 + 0.0339 * ue.LSP[bs.ID]['K'] - 0.0077 * (ue.LSP[bs.ID]['K'] ** 2) + 0.0002 * (
                    ue.LSP[bs.ID]['K'] ** 3))
        else:
            C_theta = C_theta_NLOS

        # Zenith of Arrival
        Xn = np.random.choice([-1, 1], size=updated_N_cluster)
        Yn = np.random.normal(loc=0, scale=(ue.LSP[bs.ID]['ZSA'] / 7), size=updated_N_cluster)
        theta_notch_n_ZOA = - ue.LSP[bs.ID]['ZSA'] * np.log(P_n / max(P_n)) / C_theta

        if self.getLOS(self.los_Matrix[ue.ID][bs.ID]) == 'O2I':
            theta_LOS_ZOA = 90
        else:
            # Todo: Not sure if computed correctly - review (may need to add orientation to UEs)
            theta_LOS_ZOA = np.rad2deg(self.los_zen_angle_rad_Matrix[ue.ID][bs.ID])

        if not (self.getLOS(self.los_Matrix[ue.ID][bs.ID]) == LOS.LOS):
            theta_n_ZOA = Xn * theta_notch_n_ZOA + Yn + theta_LOS_ZOA
        else:
            theta_n_ZOA = (Xn * theta_notch_n_ZOA + Yn) - (Xn[0] * theta_notch_n_ZOA[0] + Yn[0] - theta_LOS_ZOA)

        alpha_m = np.array([0.0447, -0.0447, 0.1413, -0.1413, 0.2492, -0.2492, 0.3715, -0.3715, 0.5129, -0.5129, 0.6797,
                            -0.6797, 0.8844, -0.8844, 1.1481, -1.1481, 1.5195, -1.5195, 2.1551, -2.1551])

        # theta_n_m_ZOA = [None] * ue.LSP[bs.ID]['M']
        # for m in range(ue.LSP[bs.ID]['M']):
        #     theta_n_m_ZOA[m] = theta_n_ZOA + ue.LSP[bs.ID]['c_ZSA'] * alpha_m[m]
        #     if (theta_n_m_ZOA[m] >= 180) and (theta_n_m_ZOA[m] <= 360):
        #         theta_n_m_ZOA[m] = 360 - theta_n_m_ZOA[m]

        theta_n_m_ZOA = np.zeros((updated_N_cluster, ue.LSP[bs.ID]['M']))
        for n in range(updated_N_cluster):
            temp = theta_n_ZOA[n] + ue.LSP[bs.ID]['c_ZSA'] * alpha_m
            temp[(temp >= 180) & (temp <= 360)] = 360 - temp[(temp >= 180) & (temp <= 360)]
            theta_n_m_ZOA[n] = temp

        # Zenith of Departure
        Xn = np.random.choice([-1, 1], size=updated_N_cluster)
        Yn = np.random.normal(loc=0, scale=(ue.LSP[bs.ID]['ZSD'] / 7), size=updated_N_cluster)
        theta_notch_n_ZOD = - ue.LSP[bs.ID]['ZSD'] * np.log(P_n / max(P_n)) / C_theta

        # Todo: Not sure if computed correctly - review (may need to add orientation to UEs)
        theta_LOS_ZOD = np.rad2deg(self.los_zen_angle_rad_Matrix[ue.ID][bs.ID])

        if not (self.getLOS(self.los_Matrix[ue.ID][bs.ID]) == LOS.LOS):
            theta_n_ZOD = Xn * theta_notch_n_ZOD + Yn + theta_LOS_ZOD + ue.LSP[bs.ID]['mu_offset_ZOD']
        else:
            theta_n_ZOD = (Xn * theta_notch_n_ZOD + Yn) - (Xn[0] * theta_notch_n_ZOD[0] + Yn[0] - theta_LOS_ZOD)

        alpha_m = np.array([0.0447, -0.0447, 0.1413, -0.1413, 0.2492, -0.2492, 0.3715, -0.3715, 0.5129, -0.5129, 0.6797,
                            -0.6797, 0.8844, -0.8844, 1.1481, -1.1481, 1.5195, -1.5195, 2.1551, -2.1551])

        # theta_n_m_ZOD = [None] * ue.LSP[bs.ID]['M']
        # for m in range(ue.LSP[bs.ID]['M']):
        #     theta_n_m_ZOD[m] = theta_n_ZOD + (3/8)*(10**ue.LSP[bs.ID]['mu_lg_ZSD']) * alpha_m[m]
        #     if (theta_n_m_ZOD[m] >= 180) and (theta_n_m_ZOD[m] <= 360):
        #         theta_n_m_ZOD[m] = 360 - theta_n_m_ZOD[m]

        theta_n_m_ZOD = np.zeros((updated_N_cluster, ue.LSP[bs.ID]['M']))
        for n in range(updated_N_cluster):
            temp = theta_n_ZOD[n] + (3 / 8) * (10 ** ue.LSP[bs.ID]['mu_lg_ZSD']) * alpha_m
            temp[(temp >= 180) & (temp <= 360)] = 360 - temp[(temp >= 180) & (temp <= 360)]
            theta_n_m_ZOD[n] = temp

        ################################################################################################################
        # Step 8: Coupling of rays within a cluster for both azimuth and elevation

        ray_mapping_AoD_AoA = np.zeros((updated_N_cluster, ue.LSP[bs.ID]['M']), dtype=int)
        ray_mapping_ZoD_ZoA = np.zeros((updated_N_cluster, ue.LSP[bs.ID]['M']), dtype=int)
        ray_mapping_AoD_ZoA = np.zeros((updated_N_cluster, ue.LSP[bs.ID]['M']), dtype=int)
        for n in range(updated_N_cluster):
            ray_mapping_AoD_AoA[n] = np.random.permutation(ue.LSP[bs.ID]['M']).astype(int)
            ray_mapping_ZoD_ZoA[n] = np.random.permutation(ue.LSP[bs.ID]['M']).astype(int)
            ray_mapping_AoD_ZoA[n] = np.random.permutation(ue.LSP[bs.ID]['M']).astype(int)

        ################################################################################################################
        # Step 9: Generate the cross polarization power ratios

        Xnm = np.random.normal(loc=ue.LSP[bs.ID]['mu_XPR'], scale=ue.LSP[bs.ID]['sigma_xpr'],
                               size=(updated_N_cluster, ue.LSP[bs.ID]['M']))
        Knm = np.power(10, Xnm / 10)

        ################################################################################################################
        # Step 10: Coefficient generation - Draw initial random phases
        # TODO: I think, from now on, parameters should be defined per UE-Sector pair (instead of UE-BS as previous
        #       parameters.

        ################################################################################################################
        # Step 11: Coefficient generation - Generate channel coefficients for each cluster n and each receiver and
        # transmitter element pair u, s.

        ################################################################################################################
        # Step 12: Apply pathloss and shadowing for the channel coefficients.

        return 0

    def computeRSRP(self, BS_list: list[BaseStation] = None, UE_list: list[UserEquipment] = None):
        """
        Compute the RSRP
        :param BS_list: list of BaseStation
        :param UE_list: list of UserEquipment
        :return: update Network attribute RsrpMatrix
        """

        if BS_list is None:
            BS_list = self.BSs

        if UE_list is None:
            UE_list = self.UEs

        self.cell_sector_mapping(BS_list)

        nBS = len(BS_list)
        nUE = len(UE_list)
        nSectors = len(self.cellSectorMap)

        self.RSRP_Matrix = np.zeros((nUE, nSectors))
        for bs in BS_list:
            for sec in bs.sector:
                for ue in UE_list:
                    self.RSRP_Matrix[ue.ID][sec.ID] = sec.tx_power_dB - 10 * np.log10(12 * sec.number_of_PRBs) - \
                                                      self.pathlossMatrix[ue.ID][sec.ID]

    def UE_attach(self, BS_list: list[BaseStation] = None, UE_list: list[UserEquipment] = None):
        """
        Performs UE attachment -> finds the sector for which UE senses the highest RSRP and 'connect' to them
        :param BS_list:
        :param UE_list:
        :return: updates attributes 'serving_sector' and 'serving_base_station' of the UserEquipment object
        """
        if BS_list is None:
            BS_list = self.BSs

        if UE_list is None:
            UE_list = self.UEs

        self.computeRSRP(BS_list=BS_list, UE_list=UE_list)
        for eu_ind, ue in enumerate(UE_list):
            highestRSRP_sectorIndex = np.argmax(self.RSRP_Matrix[eu_ind][:])
            if self.RSRP_Matrix[eu_ind][highestRSRP_sectorIndex] > self.UE_attach_threshold:
                ue.serving_sector = highestRSRP_sectorIndex
                ue.serving_base_station = self.cellSectorMap[highestRSRP_sectorIndex]
            else:
                ue.serving_sector = None
                ue.serving_base_station = None

            # with np.printoptions(precision=1, suppress=True):
            #     print(f'UE:{eu_ind} - RSRP:{self.RSRP_Matrix[eu_ind][:]}')
            #     print(f'UE:{eu_ind} - RSRP:{self.RSRP_Matrix[eu_ind][highestRSRP_sectorIndex]}')

    def computeSINR(self, BS_list: list[BaseStation] = None, UE_list: list[UserEquipment] = None):
        """
        Computes the SINR for the UEs in the UE list; it assumes that the desired signal is received from the serving
        sector and the interference is received for all other sectors.
        :param BS_list: list of BaseStation
        :param UE_list: list of UserEquipment
        :return:
        """
        if BS_list is None:
            BS_list = self.BSs

        if UE_list is None:
            UE_list = self.UEs

        nBS = len(BS_list)
        nUE = len(UE_list)
        nSectors = sum([bs.number_of_sectors for bs in BS_list])
        SEC_list = [sec.ID for bs in BS_list for sec in bs.sector]

        self.SINR_Matrix = np.zeros(nUE)
        for ue in UE_list:
            if ue.serving_sector is not None:
                signal_power_dB = self.RSRP_Matrix[ue.ID][ue.serving_sector]
                sector_idx_in_BS = self.BSs[self.cellSectorMap[ue.serving_sector]].get_sector_by_ID(ue.serving_sector)

                # interference_plus_noise = 10 ** (ue.noise_floor/ 10)
                # Adjust noise level to the number of PRBs
                interference_plus_noise = 10 ** ((ue.noise_floor - 10 * np.log10(
                    12 * self.BSs[self.cellSectorMap[ue.serving_sector]].sector[sector_idx_in_BS].number_of_PRBs)) / 10)

                for sec_idx in (np.delete(np.array(SEC_list), ue.serving_sector)):
                    interference_plus_noise = interference_plus_noise + 10 ** (self.RSRP_Matrix[ue.ID][sec_idx] / 10)

                interference_plus_noise_dB = 10 * np.log10(interference_plus_noise)
                self.SINR_Matrix[ue.ID] = signal_power_dB - interference_plus_noise_dB
            else:
                self.SINR_Matrix[ue.ID] = -np.inf

        pass

    def cell_sector_mapping(self, BS_list: list[BaseStation]):
        self.cellSectorMap = []
        for bs in BS_list:
            self.cellSectorMap += [None]*bs.number_of_sectors
            for sec in bs.sector:
                self.cellSectorMap[sec.ID] = bs.ID

    @staticmethod
    def getLOS(los):
        if los == 0:
            return 'LOS'
        if los == 1:
            return 'NLOS'
        if los == 2:
            return 'O2I'

    @property
    def number_of_ue(self):
        return len(self.UEs)

    @property
    def number_of_bs(self):
        return len(self.BSs)


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


def dist2d(device1: Union[BaseStation, UserEquipment], device2: Union[BaseStation, UserEquipment]):
    return np.sqrt(np.power(device1.pos_x - device2.pos_x, 2) + np.power(device1.pos_y - device2.pos_y, 2))


def dist3d(device1: Union[BaseStation, UserEquipment], device2: Union[BaseStation, UserEquipment]):
    return np.sqrt(np.power(device1.pos_x - device2.pos_x, 2)
                   + np.power(device1.pos_y - device2.pos_y, 2)
                   + np.power(device1.height - device2.height, 2))
