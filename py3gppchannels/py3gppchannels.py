import numpy as np
import itertools
from typing import List, Union, Type
from scipy.signal import convolve2d
from scipy.linalg import sqrtm
from scipy.interpolate import RegularGridInterpolator
from enum import Enum

light_speed = 299_792_458  # m/s


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


class AntennaElement:
    def __init__(self, antenna_model='Model-2', radiation_pattern_model='3GPP'):
        self.antenna_model = antenna_model
        self.radiation_pattern_model = radiation_pattern_model

    def radiation_power_pattern(self, theta_lcs, phi_lcs):
        """
        :param phi_lcs: azimuth angle in LCS
        :param theta_lcs: elevation angle in LCS
        """
        if self.radiation_pattern_model == '3GPP':
            A_dB_vertical_cut = -np.minimum(12 * ((theta_lcs - 90) / 65) ** 2, 30)
            A_dB_horizontal_cut = -np.minimum(12 * (phi_lcs / 65) ** 2, 30)
            A_dB = -np.minimum(-(A_dB_vertical_cut + A_dB_horizontal_cut), 30)
            return A_dB

        else:
            raise 'Invalid radiation pattern model'

        # TODO: How does the Maximum directional gain come into play?

    def field_pattern_vertical(self, theta_lcs, phi_lcs, zeta):
        """
        :param phi_lcs: azimuth angle in LCS
        :param theta_lcs: elevation angle in LCS
        :param zeta: polarization slant angle
        """
        A_lcs_linear = 10 ** (self.radiation_power_pattern(theta_lcs, phi_lcs) / 10)
        return np.sqrt(A_lcs_linear) * np.cos(zeta)

    def field_pattern_horizontal(self, theta_lcs, phi_lcs, zeta):
        """
        :param phi_lcs: azimuth angle in LCS
        :param theta_lcs: elevation angle in LCS
        :param zeta: polarization slant angle
        """
        A_lcs_linear = 10 ** (self.radiation_power_pattern(theta_lcs, phi_lcs) / 10)
        return np.sqrt(A_lcs_linear) * np.sin(zeta)

    def field_pattern_vector(self, theta_lcs, phi_lcs, zeta):
        """
        :param phi_lcs: azimuth angle in LCS
        :param theta_lcs: elevation angle in LCS
        :param zeta: polarization slant angle
        """
        A_lcs_linear = 10 ** (self.radiation_power_pattern(theta_lcs, phi_lcs) / 10)
        return np.array([np.sqrt(A_lcs_linear) * np.cos(zeta), np.sqrt(A_lcs_linear) * np.sin(zeta)]).T


class AntennaPanel:
    def __init__(self, n_panel_col: int = 1, n_panel_row: int = 1,
                 panel_v_spacing: float = 0.5, panel_h_spacing: float = 0.5,
                 n_antenna_col: int = 2, n_antenna_row: int = 2,
                 antenna_v_spacing: float = 0.5, antenna_h_spacing: float = 0.5,
                 polarization: str = 'single', radiation_pattern_model: str = '3GPP'
                 ):
        """
        Create the antenna panel object. Each antenna panel contains an antenna array.

        :param sector_id: sector ID associated with the antenna panel
        :param n_panel_col: number of columns in the panel array (Ng)
        :param n_panel_row: number of rows in the panel array (Mg)
        :param panel_v_spacing: vertical spacing between panels (i.e., space between panels of two consecutive rows
                                in the same column) measured from the center of the panel
        :param panel_h_spacing: horizontal spacing between panels (i.e., space between panels of two consecutive
                                columns in the same row) measured from the center of the panel
        :param n_antenna_col: number of antennas columns in the antenna array (i.e., within the panel) (N)
        :param n_antenna_row: number of antennas rows in the antenna array (i.e., within the panel) (M)
        :param antenna_v_spacing: vertical spacing between antennas
        :param antenna_h_spacing: horizontal spacing between antennas
        :param polarization: polarization, either 'single' or 'dual'
        """

        # Pannels geometry
        self.n_panel_col = n_panel_col  # Ng
        self.n_panel_row = n_panel_row  # Mg
        self.panel_v_spacing = panel_v_spacing
        self.panel_h_spacing = panel_h_spacing

        # Array geometry
        self.n_antenna_col = n_antenna_col  # N
        self.n_antenna_row = n_antenna_row  # M
        self.antenna_v_spacing = antenna_v_spacing
        self.antenna_h_spacing = antenna_h_spacing
        self.polarization = polarization
        self.antenna_element = AntennaElement(radiation_pattern_model=radiation_pattern_model)

        # Orientation
        # self.bearing_angle = np.random.uniform(0, 2 * np.pi)  # Omega_UE_alpha
        # self.downtilt_angle = np.random.uniform(0, 2 * np.pi)  # Omega_UE_beta
        # self.slant = np.random.uniform(0, 2 * np.pi)  # Omega_UE_gamma
        self.bearing_angle = 0  # Omega_UE_alpha
        self.downtilt_angle = 0  # Omega_UE_beta
        self.slant = 0  # Omega_UE_gamma

        # Array location vector
        self._array_location_tensor = self.array_location_tensor
        self._array_location_vector = self.array_location_vector

        self.coordinate_system = CoordinateSystem

    def vector_field_transformation_from_gcs(self, theta_gcs, phi_gcs):
        alpha = self.bearing_angle
        beta = self.downtilt_angle
        gamma = self.slant

        theta_lcs, phi_lcs = self.coordinate_system.GCS2LCS_angle(alpha=alpha, beta=beta, gamma=gamma,
                                                                  theta=theta_gcs, phi=phi_gcs)
        vector_field_transformation_matrix = self.coordinate_system.polarized_field_component_transformation_matrix(
                                                                    alpha=alpha, beta=beta, gamma=gamma,
                                                                    theta_lcs=theta_lcs, phi_lcs=phi_lcs,
                                                                    theta_gcs=theta_gcs, phi_gcs=phi_gcs)
        vector_field_gcs = np.matmul(vector_field_transformation_matrix,
                                     self.antenna_element.field_pattern_vector(theta_lcs=theta_lcs, phi_lcs=phi_lcs,
                                                                               zeta=0))

        return vector_field_gcs

    @property
    def array_location_tensor(self):
        """
        Computes the array_location_vector with shape (Mg*M, Ng*N, 3), where Mg*M and Ng*N are, respectively, the total
        number of antenna element rows and columns in the panel, and the last dimension contains the x, y, z location of
        the respective antenna element. Ex: array_location_vector[u,v,:] = [x_uv, y_uv, y_uv] are the coordinates of the
        element in the u-th row and v-th column of the panel.
        :return: array_location_vector
        """
        self._array_location_tensor = np.zeros((self.n_panel_row * self.n_antenna_row,
                                                self.n_panel_col * self.n_antenna_col, 3))

        y_axis = np.array([p + a for p in np.arange(self.n_panel_row) * self.panel_h_spacing for a in
                           np.arange(self.n_antenna_row) * self.antenna_h_spacing])

        z_axis = np.array([p + a for p in np.arange(self.n_panel_col) * self.panel_v_spacing for a in
                           np.arange(self.n_antenna_col) * self.antenna_v_spacing])

        A = np.meshgrid(y_axis, z_axis, indexing='xy')

        # Y-position
        self._array_location_tensor[:, :, 1] = A[0]

        # Z-position
        self._array_location_tensor[:, :, 2] = A[1]

        return self._array_location_tensor

    @property
    def array_location_vector(self):
        self._array_location_vector = np.reshape(self._array_location_tensor, (
                                      self.n_panel_row * self.n_antenna_row * self.n_panel_col * self.n_antenna_col, 3))
        return self._array_location_vector

    # @property
    # def number_of_antenna_col(self):
    #     return len(self.antenna_panels)
    # 
    # @property
    # def number_of_antenna_row(self):
    #     return len(self.antenna_panels)
    # 
    # @property
    # def number_of_antenna(self):
    #     return len(self.antenna_panels)

    def panel_field_pattern_vector(self, theta_gcs, phi_gcs, zeta):
        theta_lcs, phi_lcs = CoordinateSystem.GCS2LCS_angle(alpha=self.bearing_angle,
                                                            beta=self.downtilt_angle,
                                                            gamma=self.slant,
                                                            theta=theta_gcs,
                                                            phi=phi_gcs)

        field_pattern_vector_lcs = self.antenna_element.field_pattern_vector(theta_lcs, phi_lcs, zeta)

        return field_pattern_vector_lcs

class Sector:
    # Auto ID generation
    _sector_id = itertools.count()

    def __init__(self, bs_id: int, orientation: float = 0, sector_width: float = 120,
                 frequency_ghz: int = 3.5, tx_power_dB: float = 20,
                 antenna_panels: List[AntennaPanel] = []):
        """
        Create sectors within a Base Station
        :param bs_id: Base Station ID
        :param orientation: orientation of the sector center [degrees] > 0 points toward North
        :param sector_width: width of the sector [degrees]
        :param frequency_ghz: operation frequency of the sector [GHz]
        :param tx_power_dB: transmission power [dBm]
        """
        self.ID = next(Sector._sector_id)
        self.BS_ID = bs_id
        self.sector_width = sector_width
        self.orientation = orientation
        self.frequency_ghz = frequency_ghz
        self.tx_power_dB = tx_power_dB
        self.number_of_PRBs = 100  # 20 MHz channel with SCS 15 KHz (LTE)

        self.antenna_panels = antenna_panels
        if antenna_panels:
            self.antenna_panels = antenna_panels
        else:
            self.add_antenna_panel(AntennaPanel())

    def add_antenna_panel(self, ant_panel: AntennaPanel):
        if ant_panel.__class__ == AntennaPanel:
            self.antenna_panels.append(ant_panel)

    @property
    def number_of_antenna_panels(self):
        return len(self.antenna_panels)

    @property
    def number_of_antennas(self):
        number_of_antennas = 0
        for panel in self.antenna_panels:
            number_of_antennas += panel.n_antenna_col * panel.n_panel_col * panel.n_antenna_row * panel.n_panel_row
        return number_of_antennas


class BaseStation:
    # Auto ID generation
    _bs_id = itertools.count()

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
        self.ID = next(BaseStation._bs_id)
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
                return sec
        else:
            return None


class UserEquipment:
    # pos_x = float
    # pos_y = float
    # height = float
    # location = ''
    # los = ''
    _id_iter = itertools.count()

    def __init__(self, pos_x: float = 0, pos_y: float = 0, height: float = 1.5,
                 location: Location = Location.Outdoor, tx_power_dB: float = 0, noise_floor: float = -125,
                 antenna_panels: List[AntennaPanel] = [], mobility_speed: float = 3):
        """
        Create User Equipment (UE)
        :param pos_x: position of the UE in the x axis [meters]
        :param pos_y: position of the UEt in the y axis [meters]
        :param height: height of the UE [meters]
        :param location: location of the UE ['Indoor','Outdoor']
        :param tx_power_dB: transmission power [dBm]
        :param noise_floor: noise floor [dBm]
        """

        self.ID = next(UserEquipment._id_iter)

        # Positioning
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.height = height
        self.location = location

        # Orientation
        # self.orientation = np.random.uniform(0, 2 * np.pi)
        self.bearing_angle = np.random.uniform(0, 2 * np.pi)  # Omega_UE_alpha
        self.downtilt_angle = np.random.uniform(0, 2 * np.pi)  # Omega_UE_beta
        self.slant = np.random.uniform(0, 2 * np.pi)  # Omega_UE_gamma

        # Network Information
        self.tx_power_dB = tx_power_dB
        self.serving_sector = None
        self.serving_base_station = None
        self.neighbor_cells = []
        self.noise_floor = noise_floor  # dB
        self.LSP = []

        self.antenna_panels = antenna_panels
        if antenna_panels:
            self.antenna_panels = antenna_panels
        else:
            self.add_antenna_panel(AntennaPanel())

        # Mobility
        self.velocity_ms = mobility_speed * 1000 / 3600  # 3 km/h
        self.mobility_direction_phi = np.random.uniform(0, 2 * np.pi)
        self.mobility_direction_theta = 90

    def add_antenna_panel(self, ant_panel: AntennaPanel):
        if ant_panel.__class__ == AntennaPanel:
            self.antenna_panels.append(ant_panel)

    @property
    def number_of_antenna_panels(self):
        return len(self.antenna_panels)

    @property
    def number_of_antennas(self):
        number_of_antennas = 0
        for panel in self.antenna_panels:
            number_of_antennas += panel.n_antenna_col * panel.n_panel_col * panel.n_antenna_row * panel.n_panel_row
        return number_of_antennas

    # @property
    # def number_of_antennas_col(self):
    #     number_of_antennas = 0
    #     for panel in self.antenna_panels:
    #         number_of_antennas += panel.n_antenna_col * panel.n_panel_col
    #     return number_of_antennas
    #
    # @property
    # def number_of_antennas_row(self):
    #     number_of_antennas = 0
    #     for panel in self.antenna_panels:
    #         number_of_antennas += panel.n_antenna_row * panel.n_panel_row
    #     return number_of_antennas


class Sector_UE_Link:
    def __init__(self, sector: Sector, ue: UserEquipment, los: LOS):
        self.base_station_ID = sector.BS_ID
        self.sector_ID = sector.ID
        self.ue_ID = ue.ID
        self.los = los
        self.active = False  # Currently not used; will be used to indicate that link is active - ie, sector is serving sector for the ue

        # Channel coefficients
        self.channel = Channel()
        self.pathloss = np.inf
        self.sigma_sf = 0.0
        self.RSRP = -np.inf



class BS_UE_Link:
    def __init__(self, bs: BaseStation, ue: UserEquipment):
        self.base_station_ID = bs.ID
        self.ue_ID = ue.ID
        self.active = False  # Currently not used; will be used to indicate that link is active - ie, sector is serving sector for the ue

        self.sector_links = []  # Holds all Sector_UE links for this BS-UE pair
        self.serving_sector = None  # Indicates the serving sector if any

        # Geometry
        self.distance_2D = None
        self.distance_3D = None
        self.LOS_AOD_GCS = None
        self.LOS_ZOD_GCS = None
        self.LOS_AOA_GCS = None
        self.LOS_ZOA_GCS = None

        # Large Scale Link parameters
        self.los = None
        self.pathloss = None
        self.sigma_sf = None
        self.delay_spread = None
        self.AOD_spread = None
        self.ZOD_spread = None
        self.AOA_spread = None
        self.ZOA_spread = None
        self.Ricean_factor = None

        # Small Scale Link parameters
        self.AOD_nm = None
        self.ZOD_nm = None
        self.AOA_nm = None
        self.ZOA_nm = None
        self.ray_coupling = None
        self.cross_pol_power_ratio = None

        self.LSP = None


class Channel:
    def __init__(self, Nr=1, Nt=1):
        self.Nr = Nr
        self.Nt = Nt
        self.cluster_matrices = []
        self.cluster_delays = []

    def add_cluster(self, H, tau):
        self.cluster_matrices.append(H)
        self.cluster_delays.append(tau)


class Network:
    BS_UE_link_container: Type[dict]

    def __init__(self, scenario: str = 'UMa', free_space: bool = False,
                 BSs: Union[List[BaseStation], BaseStation] = [],
                 UEs: Union[List[UserEquipment], UserEquipment] = [],
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

        self.BSs = BSs
        self.BS_tx_power_dB = 23
        self.BS_noise_floor_dB = -125

        self.UEs = UEs
        self.UE_tx_power_dB = 18
        self.UE_noise_floor_dB = -125

        self.SINR_Matrix = np.array([])

        self.UE_attach_threshold = -140  # Todo: Use reasonable value

        # Correlation Parameters
        self._C = {}
        self._delta_m = {}
        self._LSP_list = {}
        self._Q = {}

        self._parameter_grid = {}

        # Link Containers
        self._BS_UE_link_container = {}
        self._Sector_UE_link_container = {}

        # Coordinate System
        self.coordinate_system = CoordinateSystem


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



    @property
    def BS_UE_link_container(self):
        for bs in self.BSs:
            for ue in self.UEs:
                bs_ue_key = (bs.ID, ue.ID)
                if not (bs_ue_key in self._BS_UE_link_container):
                    self._BS_UE_link_container[bs_ue_key] = BS_UE_Link(bs=bs,ue=ue)
        return self._BS_UE_link_container

    @property
    def Sector_UE_link_container(self):
        for ue in self.UEs:
            for bs in self.BSs:
                los = self._BS_UE_link_container[(bs.ID, ue.ID)].los
                for sector in bs.sector:
                    sector_ue_key = (sector.ID, ue.ID)
                    if not (sector_ue_key in self._Sector_UE_link_container):
                        self._Sector_UE_link_container[sector_ue_key] = Sector_UE_Link(sector, ue, los)
        return self._Sector_UE_link_container

    @property
    def parameter_grid(self):
        [(x_min, y_min), (x_max, y_max)] = self.network_limits
        for los in [LOS.LOS, LOS.NLOS]:
            for bs in self.BSs:
                for lsp in self._LSP_list[los]:
                    if not((los.value, bs.ID, lsp) in self._parameter_grid):
                        self._parameter_grid[(los.value, bs.ID, lsp)] = CorrelationGrid(x_min=x_min, y_min=y_min,
                                                                                        x_max=x_max, y_max=y_max,
                                                                                        corr_dist=self._delta_m[los][
                                                                                            lsp])
        return self._parameter_grid

    def add_ue(self, pos_x: float = 0, pos_y: float = 0, height: float = None,
               location: Location = Location.UNDETERMINED,
               speed: float = None, tx_power_dB: float = None, noise_floor: float = None):

        if height is None:
            height = self.UE_height
        if location is None:
            location = self.UELocation()
        if tx_power_dB is None:
            tx_power_dB = self.UE_tx_power_dB
        if noise_floor is None:
            noise_floor = self.UE_noise_floor_dB
        if speed is None:
            speed = self.UE_mobility

        self.UEs.append(UserEquipment(pos_x=pos_x,
                                      pos_y=pos_y,
                                      tx_power_dB=tx_power_dB,
                                      height=height,
                                      location=location,
                                      noise_floor=noise_floor))

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

    def LineOfSight(self, bs_ue_link: BS_UE_Link) -> LOS:
        """
        Determine if a given BS and UE pair is in 'LOS' or 'NLOS'
        0: 'LOS'        1: 'NLOS'       2: 'IOF'

        :param bs_ue_link: BS_UE_Link object
        :return: the line-of-sight condition, i.e.: 'LOS' or 'NLOS'
        """

        dist_2D = self.BS_UE_link_container[bs_ue_link].distance_2D
        bs_id, ue_id = bs_ue_link
        ue = self.UEs[ue_id]

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
            return LOS.LOS
        else:
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

        for link in self.BS_UE_link_container:
            bs_id, ue_id = link
            ue = self.UEs[ue_id]
            bs = self.BSs[bs_id]

            dist_2D = dist2d(bs, ue)
            dist_3D = dist3d(bs, ue)

            # Compute relative UE position (assuming BS is the origin)
            xy_position = np.array([(ue.pos_x - bs.pos_x), (ue.pos_y - bs.pos_y)])

            # Compute the AoD angle
            if xy_position[1] >= 0:
                aod_rad_gcs = np.arccos(xy_position[0] / dist_2D)
            else:
                aod_rad_gcs = 2 * np.pi - np.arccos(xy_position[0] / dist_2D)

            # Compute the AoA angle
            xy_position = -xy_position      # Invert xy_position
            if xy_position[1] >= 0:
                aoa_rad_gcs = np.arccos(xy_position[0] / dist_2D)
            else:
                aoa_rad_gcs = 2 * np.pi - np.arccos(xy_position[0] / dist_2D)

            # Compute relative BS height
            h_e = bs.height - ue.height

            # Compute ZoD angle
            zod_rad_gcs = np.pi / 2  # If h_e == 0
            if h_e > 0:
                zod_rad_gcs = np.pi - np.arctan(dist_2D / h_e)
            elif h_e < 0:
                zod_rad_gcs = - np.arctan(dist_2D / h_e)

            # Compute ZoA angle
            h_e = -h_e      # Invert h_e
            zoa_rad_gcs = np.pi / 2  # If h_e == 0
            if h_e > 0:
                zoa_rad_gcs = np.pi - np.arctan(dist_2D / h_e)
            elif h_e < 0:
                zoa_rad_gcs = - np.arctan(dist_2D / h_e)

            self.BS_UE_link_container[link].distance_2D = dist_2D
            self.BS_UE_link_container[link].distance_3D = dist_3D

            self.BS_UE_link_container[link].LOS_AOD_GCS = aod_rad_gcs
            self.BS_UE_link_container[link].LOS_ZOD_GCS = zod_rad_gcs
            self.BS_UE_link_container[link].LOS_AOA_GCS = aoa_rad_gcs
            self.BS_UE_link_container[link].LOS_ZOA_GCS = zoa_rad_gcs

    def computeLOS(self):
        """
        Computes line-of-sight conditions for BSs and UEs.
        Results are stored in the class attribute 'los_Matrix'.
        """
        for link in self._BS_UE_link_container:
            self._BS_UE_link_container[link].los = self.LineOfSight(link)

    def Pathloss(self, link: BS_UE_Link):
        """
        Computes the Pathloss between a BS sector and UE pair based on the Network scenario
        :param link: BS_UE_Link object
        :return: Pathloss [dB]
        """
        # Load Sector-UE links
        self.Sector_UE_link_container

        dist_2D = link.distance_2D
        dist_3D = link.distance_3D
        los = link.los

        bs_id = link.base_station_ID
        ue_id = link.ue_ID
        ue = self.UEs[ue_id]
        bs = self.BSs[bs_id]

        # Split sector by frequency:
        fc_per_sector = [sector.frequency_ghz for sector in self.BSs[bs_id].sector]
        sectors_per_fc = {}
        for sector_idx, key_fc in enumerate(fc_per_sector):
            sectors_per_fc.setdefault(key_fc, []).append(sector_idx)

        for key_fc in sectors_per_fc.keys():
            sectors_in_fc = sectors_per_fc[key_fc]
            sector_ids = [self.BSs[bs_id].sector[sector_idx].ID for sector_idx in sectors_in_fc]

            fc = float(key_fc)
            pathloss = 0
            sigma_sf = 0


            # Basic Pathloss
            # If free_space flag is True, compute pathloss according to the Free Space model
            # If free_space flag is False, compute pathloss according to scenario
            if self.free_space:
                pathloss = 20 * np.log10(4 * np.pi * fc * 10 ** 9 / light_speed) + 20 * np.log10(dist_3D)

            else:
                # Pathloss for Rural Macro scenario
                if self.scenario == 'RMa':
                    if not (ue.height == 1.5):
                        raise "UE height is not the default value"  # Todo: need to check for correction formulas
                    if not (bs.height == 35):
                        raise "BS height is not the default value"  # Todo: need to check for correction formulas

                    # Break point distance (Table 7.4.1.1, Note 5)
                    d_bp = 2 * np.pi * bs.height * ue.height * fc * 1_000_000_000 / light_speed

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

                    d_bp = 4 * (bs.height - h_e) * (ue.height - h_e) * fc * 1_000_000_000 / light_speed

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
                    d_bp = 4 * (bs.height - h_e) * (ue.height - h_e) * fc * 1_000_000_000 / light_speed

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

            for sectors in sector_ids:
                # Sectorization
                # Todo: incorporate ZoA/ZoD
                # Todo: this should be removed once the antenna patterns are defined.
                # For now, this is doing simple sectorization
                bs_sector_id = [x for x, s in enumerate(self.BSs[bs_id].sector) if s.ID == sectors][0]

                sec = self.BSs[link.base_station_ID].sector[bs_sector_id]
                angle_diff = link.LOS_AOD_GCS - np.deg2rad(sec.orientation)
                if angle_diff > np.pi:
                    angle_diff = angle_diff - 2 * np.pi
                if angle_diff < - np.pi:
                    angle_diff = angle_diff + 2 * np.pi

                angle_diff = np.rad2deg(angle_diff)
                antenna_gain = -np.minimum(- (-np.minimum(12 * ((np.rad2deg(link.LOS_ZOD_GCS) - 90)/65)**2, 30) +
                                              -np.minimum(12 * (angle_diff / 65) ** 2, 30)), 30)

                sector_ue_link = (sectors, ue_id)
                self._Sector_UE_link_container[sector_ue_link].pathloss = pathloss - antenna_gain
                self._Sector_UE_link_container[sector_ue_link].sigma_sf = sigma_sf

            # # Sectorization
            # # Todo: incorporate ZoA/ZoD
            # # Todo: this should be removed once the antenna patterns are defined.
            # # For now, this is doing simple sectorization
            # bs_sector_id = [x for x, s in enumerate(self.BSs[bs_id].sector) if s.ID == sectors][0]
            #
            # sec = self.BSs[link.base_station_ID].sector[bs_sector_id]
            # angle_diff = link.LOS_AOD_GCS - np.deg2rad(sec.orientation)
            # if angle_diff > np.pi:
            #     angle_diff = angle_diff - 2 * np.pi
            # if angle_diff < - np.pi:
            #     angle_diff = angle_diff + 2 * np.pi
            # if np.abs(angle_diff) <= (np.deg2rad(sec.sector_width / 2)):
            #     pathloss = pathloss
            # else:
            #     # pathloss = np.inf
            #     pathloss = pathloss + 30  # Add 30 dB (which is the minimum antenna gain as per the standard)

    def NetworkPathloss(self):
        """
        Computes the Pathloss and Line of Sight parameters for combinations of BSs and UEs
        """
        self.computeGeometry()
        for link in self.BS_UE_link_container:
            self.Pathloss(self._BS_UE_link_container[link])

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
                    # if self.getLOS(self.los_Matrix[ue.ID][bs.ID]) == los:
                    if self.los_Matrix[ue.ID][bs.ID] == los:
                        s_tilde = np.dot(Q, correlated_epsilon[:, ue_idx])
                        correlated_TLSP = dict(zip(LSP, s_tilde))
                        ue.LSP[bs.ID] = self.generateLinkLPS(bs, ue, correlated_TLSP)
                        self.generateSmallScaleParams_link(bs=bs, ue=ue)


    def large_scale_parameter_correlation_method_three(self):
        # Method 3: Create grid for each parameter; points are spaced according to the correlation distances
        ################################################################################################################
        # Get geometry and design the grid
        ################################################################################################################
        # Load Large Scale Params Correlation
        self.LargeScaleParamsCorrelation()

        # Load sector-UE links
        self.Sector_UE_link_container

        for bs_ue_link in self.BS_UE_link_container:
            bs_id, ue_id = bs_ue_link
            los = self._BS_UE_link_container[bs_ue_link].los
            ue = self.UEs[ue_id]

            # for lsp in self._LSP_list[los]:
            correlated_epsilon = np.array([
                self.parameter_grid[(los.value, bs_id, lsp)].get_correlated_value_at_point(ue.pos_x, ue.pos_y)
                for lsp in self._LSP_list[los]
            ])

            s_tilde = np.dot(self._Q[los], correlated_epsilon)
            correlated_TLSP = dict(zip(self._LSP_list[los], s_tilde))

            # self._BS_UE_link_container[bs_ue_link].LSP = self.generateLinkLSP(self._BS_UE_link_container[bs_ue_link],
            #                                                                   correlated_TLSP)
            # self.generateSmallScaleParams_link(self._BS_UE_link_container[bs_ue_link])


            # Split sector by frequency:
            fc_per_sector = [sector.frequency_ghz for sector in self.BSs[bs_id].sector]
            sectors_per_fc = {}
            for sector_idx, key_fc in enumerate(fc_per_sector):
                sectors_per_fc.setdefault(key_fc, []).append(sector_idx)

            for key_fc in sectors_per_fc.keys():
                sectors_in_fc = sectors_per_fc[key_fc]
                sector_ids = [self.BSs[bs_id].sector[sector_idx].ID for sector_idx in sectors_in_fc]

                sector_ue_link = (sector_ids[0], ue_id)
                LSP = self.generateLinkLSP(self._Sector_UE_link_container[sector_ue_link], correlated_TLSP)

                for sectors in sector_ids:
                    sector_ue_link = (sectors, ue_id)
                    self._Sector_UE_link_container[sector_ue_link].LSP = LSP

            self.generateSmallScaleParams_link(self._BS_UE_link_container[bs_ue_link])

    def LargeScaleParamsCorrelation(self):
        for los in [LOS.LOS, LOS.NLOS]:
            self._C[los], self._delta_m[los], self._LSP_list[los] = self._generateLargeScaleParamsCorrelation(los=los)
            self._Q[los] = sqrtm(self._C[los])

    def _generateLargeScaleParamsCorrelation(self, los):
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
            LSP_list = ['SF', 'K', 'DS', 'ASD', 'ASA', 'ZSD', 'ZSA']
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
            LSP_list = ['SF', 'DS', 'ASD', 'ASA', 'ZSD', 'ZSA']
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

        C = C + C.T + np.eye(len(LSP_list))

        return C, delta_m, LSP_list

    def generateLinkLSP(self, link: Sector_UE_Link, correlated_TLSP):
        ue_id = link.ue_ID
        sector_id = link.sector_ID
        bs_id = link.base_station_ID

        ue = self.UEs[ue_id]
        bs = self.BSs[bs_id]
        sector = self.BSs[bs_id].get_sector_by_ID(sector_id)

        # Large Scale Parameters (LSP) for different BS-UE links are uncorrelated, but the LSPs for links from co-sited
        # sectors to a UE are the same. In addition, LSPs for the links of UEs on different floors are uncorrelated.

        fc = sector.frequency_ghz  # GHz
        los = self._BS_UE_link_container[(bs_id,ue_id)].los
        dist_2D = self._BS_UE_link_container[(bs_id,ue_id)].distance_2D

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
                mu_lg_ZSD = np.maximum(-0.21, -14.8 * (dist_2D / 1000) + 0.01 * np.abs(
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
                mu_lg_ZSD = np.maximum(-0.5, -3.1 * (dist_2D / 1000) + 0.01 * np.maximum(
                    ue.height - bs.height, 0) + 0.2)
                sigma_lg_ZSD = 0.35
                mu_offset_ZOD = -10 ** (-1.5 * np.log10(np.maximum(10, dist_2D)) + 3.3)

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

            if los == LOS.LOS:
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
                                       -2.1 * (dist_2D / 1000) - 0.01 * (
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

            if los == LOS.NLOS:
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
                                       -2.1 * (dist_2D / 1000) - 0.01 * (
                                               ue.height - 1.5) + 0.9)
                sigma_lg_ZSD = 0.49
                mu_offset_ZOD = (7.66 * np.log10(fc) - 5.96) - 10 ** (
                        (0.208 * np.log10(fc) - 0.782) * np.log10(np.maximum(25, dist_2D))
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

        if self.scenario == 'RMa':
            if los == LOS.LOS:
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
                mu_lg_ZSD = np.maximum(-1, -0.17 * (dist_2D / 1000) - 0.01 * (
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

            if los == LOS.NLOS:
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
                mu_lg_ZSD = np.maximum(-1, -0.19 * (dist_2D / 1000) - 0.01 * (
                        ue.height - bs.height) + 0.28)
                sigma_lg_ZSD = 0.30
                mu_offset_ZOD = np.arctan((35 - 3.5) / dist_2D) - np.arctan(
                    (35 - 1.5) / dist_2D)

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

            if los == LOS.O2I:
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
                mu_lg_ZSD = np.maximum(-1, -0.19 * (dist_2D / 1000) - 0.01 * (
                        ue.height - bs.height) + 0.28)
                sigma_lg_ZSD = 0.30
                mu_offset_ZOD = np.arctan((35 - 3.5) / dist_2D) - np.arctan(
                    (35 - 1.5) / dist_2D)

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

            if los == LOS.LOS:
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

            if los == LOS.NLOS:
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

        if los == LOS.LOS:
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

    def generateSmallScaleParams_link(self, bs_ue_link: BS_UE_Link):
        ue_id = bs_ue_link.ue_ID
        bs_id = bs_ue_link.base_station_ID
        ue = self.UEs[ue_id]
        bs = self.BSs[bs_id]
        # Small Scale Parameter generation

        # Get first sector to compute frequency independent parameters:
        sector_id = self.BSs[bs_id].sector[0].ID
        link = self._Sector_UE_link_container[(sector_id, ue_id)]

        # Frequency Independent Parameters
        Xn = np.random.uniform(size=link.LSP['N'])
        Xn_aoa = np.random.choice([-1, 1], size=link.LSP['N'])
        Yn_aoa = np.random.normal(loc=0, scale=(link.LSP['ASA'] / 7), size=link.LSP['N'])
        Xn_aod = np.random.choice([-1, 1], size=link.LSP['N'])
        Yn_aod = np.random.normal(loc=0, scale=(link.LSP['ASD'] / 7), size=link.LSP['N'])
        Xn_zoa = np.random.choice([-1, 1], size=link.LSP['N'])
        Yn_zoa = np.random.normal(loc=0, scale=(link.LSP['ZSA'] / 7), size=link.LSP['N'])
        Xn_zod = np.random.choice([-1, 1], size=link.LSP['N'])
        Yn_zod = np.random.normal(loc=0, scale=(link.LSP['ZSD'] / 7), size=link.LSP['N'])

        phi_LOS_AOA = np.rad2deg(bs_ue_link.LOS_AOA_GCS)
        phi_LOS_AOD = np.rad2deg(bs_ue_link.LOS_AOD_GCS)
        theta_LOS_ZOA = np.rad2deg(bs_ue_link.LOS_ZOA_GCS)
        theta_LOS_ZOD = np.rad2deg(bs_ue_link.LOS_ZOA_GCS)


        # Split sector by frequency:
        fc_per_sector = [sector.frequency_ghz for sector in self.BSs[bs_id].sector]
        sectors_per_fc = {}
        for sector_idx, key_fc in enumerate(fc_per_sector):
            sectors_per_fc.setdefault(key_fc, []).append(sector_idx)

        for key_fc in sectors_per_fc.keys():
            sectors_in_fc = sectors_per_fc[key_fc]
            sector_ids = [self.BSs[bs_id].sector[sector_idx].ID for sector_idx in sectors_in_fc]

            sector_ue_link_key = (sector_ids[0], ue_id)
            link = self._Sector_UE_link_container[sector_ue_link_key]

            ################################################################################################################
            # Step 5: Generate cluster delays Tau_n:
            # Xn = np.random.uniform(size=link.LSP['N'])
            cluster_delay = - link.LSP['r_tau'] * link.LSP['DS'] * np.log(Xn)
            cluster_delay = np.sort(cluster_delay - min(cluster_delay))

            if link.los == LOS.LOS:
                C_tau = 0.7705 - 0.0433 * link.LSP['K'] + 0.0002 * (link.LSP['K'] ** 2) + 0.000017 * (
                        link.LSP['K'] ** 3)
                cluster_delay_LOS = cluster_delay / C_tau

            ################################################################################################################
            # Step 6: Generate cluster powers P_n:
            P_n_notch = np.exp(
                -cluster_delay * ((link.LSP['r_tau'] - 1) / (link.LSP['r_tau'] * link.LSP['DS'])))
            Zn = (10 ** (- np.random.normal(loc=0.0, scale=link.LSP['xi']) / 10))
            P_n_notch = P_n_notch * Zn

            P_n = P_n_notch / sum(P_n_notch)

            if link.los == LOS.LOS:
                K_R_linear = 10 ** (link.LSP['K'] / 10)
                P_n = (1 / (K_R_linear + 1)) * P_n
                P_n[0] = P_n[0] + (K_R_linear / (K_R_linear + 1))

            # Discard clusters with power less than -25 dB compared to the maximum cluster power
            P_n_dB = 10 * np.log10(P_n)
            P_n = P_n[P_n_dB >= max(P_n_dB) - 25]

            cluster_delay = cluster_delay[P_n_dB >= max(P_n_dB) - 25]

            # Power per ray
            P_n_ray = P_n / link.LSP['M']

            updated_N_cluster = len(P_n)

            ################################################################################################################
            # Step 7: Generate arrival angles and departure angles for both azimuth and elevation
            # Todo: Check standard (perhaps instead of using updated_N_cluster I should use the M and M only has a finite
            #  set of values, so I wouldn't need the interpolation below)

            # Azimuth
            if updated_N_cluster == 1:  # Not in the standard - obtained from GUESSING
                C_phi_NLOS = 0.200
            elif updated_N_cluster == 2:  # Not in the standard - obtained from GUESSING
                C_phi_NLOS = 0.500
            elif updated_N_cluster == 3:  # Not in the standard - obtained from GUESSING
                C_phi_NLOS = 0.699
            elif updated_N_cluster == 4:
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

            if link.los == LOS.LOS:
                C_phi = C_phi_NLOS * (1.1035 - 0.028 * link.LSP['K'] - 0.002 * (link.LSP['K'] ** 2) + 0.0001 * (
                        link.LSP['K'] ** 3))
            else:
                C_phi = C_phi_NLOS

            # Azimuth of Arrival
            # Xn_aoa = np.random.choice([-1, 1], size=updated_N_cluster)
            # Yn_aoa = np.random.normal(loc=0, scale=(link.LSP['ASA'] / 7), size=updated_N_cluster)
            Xn_aoa = Xn_aoa[0:updated_N_cluster]
            Yn_aoa = Yn_aoa[0:updated_N_cluster]
            phi_notch_n_AOA = 2 * (link.LSP['ASA'] / 1.4) * np.sqrt(-np.log(P_n / max(P_n))) / C_phi

            # phi_LOS_AOA = np.rad2deg(link.LOS_AOA_GCS)

            if not (link.los == LOS.LOS):
                phi_n_AOA = Xn_aoa * phi_notch_n_AOA + Yn_aoa + phi_LOS_AOA
            else:
                phi_n_AOA = (Xn_aoa * phi_notch_n_AOA + Yn_aoa) - (Xn_aoa[0] * phi_notch_n_AOA[0] + Yn_aoa[0] - phi_LOS_AOA)

            alpha_m = np.array(
                [0.0447, -0.0447, 0.1413, -0.1413, 0.2492, -0.2492, 0.3715, -0.3715, 0.5129, -0.5129, 0.6797, -0.6797,
                 0.8844, -0.8844, 1.1481, -1.1481, 1.5195, -1.5195, 2.1551, -2.1551])

            phi_n_m_AOA = np.zeros((updated_N_cluster, link.LSP['M']))
            for n in range(updated_N_cluster):
                phi_n_m_AOA[n] = phi_n_AOA[n] + link.LSP['c_ASA'] * alpha_m

            # Azimuth of Departure
            # Xn_aod = np.random.choice([-1, 1], size=updated_N_cluster)
            # Yn_aod = np.random.normal(loc=0, scale=(link.LSP['ASD'] / 7), size=updated_N_cluster)
            Xn_aod = Xn_aod[0:updated_N_cluster]
            Yn_aod = Yn_aod[0:updated_N_cluster]
            phi_notch_n_AOD = 2 * (link.LSP['ASD'] / 1.4) * np.sqrt(-np.log(P_n / max(P_n))) / C_phi

            # phi_LOS_AOD = np.rad2deg(link.LOS_AOD_GCS)

            if not (link.los == LOS.LOS):
                phi_n_AOD = Xn_aod * phi_notch_n_AOD + Yn_aod + phi_LOS_AOD
            else:
                phi_n_AOD = (Xn_aod * phi_notch_n_AOD + Yn_aod) - (Xn_aod[0] * phi_notch_n_AOD[0] + Yn_aod[0] - phi_LOS_AOD)

            alpha_m = np.array([0.0447, -0.0447, 0.1413, -0.1413, 0.2492, -0.2492, 0.3715, -0.3715, 0.5129, -0.5129,
                                0.6797, -0.6797, 0.8844, -0.8844, 1.1481, -1.1481, 1.5195, -1.5195, 2.1551, -2.1551])

            phi_n_m_AOD = np.zeros((updated_N_cluster, link.LSP['M']))
            for n in range(updated_N_cluster):
                phi_n_m_AOD[n] = phi_n_AOD[n] + link.LSP['c_ASD'] * alpha_m

            # Zenith
            # Todo: Check standard (perhaps instead of using updated_N_cluster I should use the M and M only has a finite
            #  set of values, so I wouldn't need the interpolation below)
            if updated_N_cluster == 1:  # Not in the standard - incorrect # TODO
                C_theta_NLOS = 0.2
            elif updated_N_cluster == 2:  # Not in the standard - incorrect # TODO
                C_theta_NLOS = 0.5
            elif updated_N_cluster == 3:  # Not in the standard - incorrect # TODO
                C_theta_NLOS = 0.699
            elif updated_N_cluster == 4:  # Not in the standard - incorrect # TODO
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

            # if self.getLOS(self.los_Matrix[ue.ID][bs.ID]) == LOS.LOS:
            if link.los == LOS.LOS:
                C_theta = C_theta_NLOS * (
                        1.3086 + 0.0339 * link.LSP['K'] - 0.0077 * (link.LSP['K'] ** 2) + 0.0002 * (
                        link.LSP['K'] ** 3))
            else:
                C_theta = C_theta_NLOS

            # Zenith of Arrival
            # Xn_zoa = np.random.choice([-1, 1], size=updated_N_cluster)
            # Yn_zoa = np.random.normal(loc=0, scale=(link.LSP['ZSA'] / 7), size=updated_N_cluster)
            Xn_zoa = Xn_zoa[0:updated_N_cluster]
            Yn_zoa = Yn_zoa[0:updated_N_cluster]
            theta_notch_n_ZOA = - link.LSP['ZSA'] * np.log(P_n / max(P_n)) / C_theta

            # if self.getLOS(self.los_Matrix[ue.ID][bs.ID]) == 'O2I':
            if link.los == LOS.O2I:
                theta_LOS_ZOA = 90
            else:
                # theta_LOS_ZOA = np.rad2deg(link.LOS_ZOA_GCS)
                pass

            if not (link.los == LOS.LOS):
                theta_n_ZOA = Xn_zoa * theta_notch_n_ZOA + Yn_zoa + theta_LOS_ZOA
            else:
                theta_n_ZOA = (Xn_zoa * theta_notch_n_ZOA + Yn_zoa) - (Xn_zoa[0] * theta_notch_n_ZOA[0] + Yn_zoa[0]
                                                                       - theta_LOS_ZOA)

            alpha_m = np.array([0.0447, -0.0447, 0.1413, -0.1413, 0.2492, -0.2492, 0.3715, -0.3715, 0.5129, -0.5129, 0.6797,
                                -0.6797, 0.8844, -0.8844, 1.1481, -1.1481, 1.5195, -1.5195, 2.1551, -2.1551])

            theta_n_m_ZOA = np.zeros((updated_N_cluster, link.LSP['M']))
            for n in range(updated_N_cluster):
                temp = theta_n_ZOA[n] + link.LSP['c_ZSA'] * alpha_m
                temp[(temp >= 180) & (temp <= 360)] = 360 - temp[(temp >= 180) & (temp <= 360)]
                theta_n_m_ZOA[n] = temp

            # Zenith of Departure
            # Xn_zod = np.random.choice([-1, 1], size=updated_N_cluster)
            # Yn_zod = np.random.normal(loc=0, scale=(link.LSP['ZSD'] / 7), size=updated_N_cluster)
            Xn_zod = Xn_zod[0:updated_N_cluster]
            Yn_zod = Yn_zod[0:updated_N_cluster]
            theta_notch_n_ZOD = - link.LSP['ZSD'] * np.log(P_n / max(P_n)) / C_theta

            # theta_LOS_ZOD = np.rad2deg(link.LOS_ZOD_GCS)

            if not (link.los == LOS.LOS):
                theta_n_ZOD = Xn_zod * theta_notch_n_ZOD + Yn_zod + theta_LOS_ZOD + link.LSP['mu_offset_ZOD']
            else:
                theta_n_ZOD = (Xn_zod * theta_notch_n_ZOD + Yn_zod) - (Xn_zod[0] * theta_notch_n_ZOD[0] + Yn_zod[0]
                                                                       - theta_LOS_ZOD)

            alpha_m = np.array([0.0447, -0.0447, 0.1413, -0.1413, 0.2492, -0.2492, 0.3715, -0.3715, 0.5129, -0.5129,
                                0.6797, -0.6797, 0.8844, -0.8844, 1.1481, -1.1481, 1.5195, -1.5195, 2.1551, -2.1551])

            theta_n_m_ZOD = np.zeros((updated_N_cluster, link.LSP['M']))
            for n in range(updated_N_cluster):
                temp = theta_n_ZOD[n] + (3 / 8) * (10 ** link.LSP['mu_lg_ZSD']) * alpha_m
                temp[(temp >= 180) & (temp <= 360)] = 360 - temp[(temp >= 180) & (temp <= 360)]
                theta_n_m_ZOD[n] = temp

            ################################################################################################################
            # Step 8: Coupling of rays within a cluster for both azimuth and elevation

            ray_mapping_AoD_AoA = np.zeros((updated_N_cluster, link.LSP['M']), dtype=int)
            ray_mapping_ZoD_ZoA = np.zeros((updated_N_cluster, link.LSP['M']), dtype=int)
            ray_mapping_AoD_ZoA = np.zeros((updated_N_cluster, link.LSP['M']), dtype=int)
            for n in range(updated_N_cluster):
                ray_mapping_AoD_AoA[n] = np.random.permutation(link.LSP['M']).astype(int)
                ray_mapping_ZoD_ZoA[n] = np.random.permutation(link.LSP['M']).astype(int)
                ray_mapping_AoD_ZoA[n] = np.random.permutation(link.LSP['M']).astype(int)

            ################################################################################################################
            # Step 9: Generate the cross polarization power ratios
            Xnm = np.random.normal(loc=link.LSP['mu_XPR'], scale=link.LSP['sigma_xpr'],
                                   size=(updated_N_cluster, link.LSP['M']))
            Knm = np.power(10, Xnm / 10)

            ############################################################################################################

            # Compute LOS/NLOS gain using Ricean Factor
            if link.los == LOS.LOS:
                gain_LOS = np.sqrt(K_R_linear / (K_R_linear + 1))
                gain_NLOS = np.sqrt(1 / (K_R_linear + 1))
            else:
                gain_NLOS = 1
            ################################################################################################################
            for sectors in sector_ids:
                sector_ue_link = (sectors, ue_id)

                ################################################################################################################
                # Step 10: Coefficient generation - Draw initial random phases
                Phi_nm = np.random.uniform(-np.pi, -np.pi, size=(updated_N_cluster, link.LSP['M'], 2, 2))
                # Phi theta-theta: Phi_nm[n,m,0,0]
                # Phi theta-phi: Phi_nm[n,m,0,1]
                # Phi phi-theta: Phi_nm[n,m,1,0]
                # Phi phi-phi: Phi_nm[n,m,1,1]

                ################################################################################################################
                # Step 11: Coefficient generation - Generate channel coefficients for each cluster n and each receiver and
                # transmitter element pair u, s.
                bs_sector_id = [x for x, s in enumerate(self.BSs[bs_id].sector) if s.ID == sectors][0]

                n_U = ue.number_of_antennas
                n_S = bs.sector[bs_sector_id].number_of_antennas
                channel = Channel(Nr=n_U, Nt=n_S)

                lambda_0 = light_speed / (bs.sector[bs_sector_id].frequency_ghz * 1E9)

                # # Compute LOS/NLOS gain using Ricean Factor
                # if link.los == LOS.LOS:
                #     gain_LOS = np.sqrt(K_R_linear / (K_R_linear + 1))
                #     gain_NLOS = np.sqrt(1 / (K_R_linear + 1))
                # else:
                #     gain_NLOS = 1

                d_rx = bs.sector[0].antenna_panels[0].array_location_vector
                d_tx = bs.sector[0].antenna_panels[0].array_location_vector

                if link.los == LOS.LOS:
                    HLOS = np.zeros((n_U, n_S), dtype=complex)
                    n = 0
                    PolM = np.array([[1, 0],
                                     [0, 1]])

                    r_rx_mn = np.array([np.sin(np.deg2rad(theta_LOS_ZOA)) * np.cos(np.deg2rad(phi_LOS_AOA)),
                                        np.sin(np.deg2rad(theta_LOS_ZOA)) * np.sin(np.deg2rad(phi_LOS_AOA)),
                                        np.cos(np.deg2rad(theta_LOS_ZOA))]).T
                    r_tx_mn = np.array([np.sin(np.deg2rad(theta_LOS_ZOD)) * np.cos(np.deg2rad(phi_LOS_AOD)),
                                        np.sin(np.deg2rad(theta_LOS_ZOD)) * np.sin(np.deg2rad(phi_LOS_AOD)),
                                        np.cos(np.deg2rad(theta_LOS_ZOD))]).T
                    r_rx_vel = ue.velocity_ms * np.array(
                        [np.sin(np.deg2rad(ue.mobility_direction_theta)) * np.cos(np.deg2rad(ue.mobility_direction_phi)),
                         np.sin(np.deg2rad(ue.mobility_direction_theta)) * np.sin(np.deg2rad(ue.mobility_direction_phi)),
                         np.cos(np.deg2rad(ue.mobility_direction_theta))]).T

                    Frx_u = ue.antenna_panels[0].vector_field_transformation_from_gcs(theta_LOS_ZOA, phi_LOS_AOA)
                    Ftx_s = bs.sector[0].antenna_panels[0].vector_field_transformation_from_gcs(theta_LOS_ZOD, phi_LOS_AOD)

                    for u in range(n_U):
                        for s in range(n_S):
                            exp_rx = np.exp(1j * 2 * np.pi * np.dot(r_rx_mn, d_rx[u]) / lambda_0)
                            exp_tx = np.exp(1j * 2 * np.pi * np.dot(r_tx_mn, d_tx[s]) / lambda_0)
                            exp_vel = np.exp(1j * 2 * np.pi * np.dot(r_rx_mn, r_rx_vel) / lambda_0)
                            HLOS[u, s] += np.dot(Frx_u.T, np.dot(PolM, Ftx_s)) * exp_rx * exp_tx * exp_vel
                    channel.add_cluster(H=gain_LOS * HLOS, tau=cluster_delay[n])

                # Todo: Need to think of how to make this more efficient (perform some vector operation to avoid so many fors:
                # HNLOS = np.zeros((n_U, n_S))
                for n in range(2, updated_N_cluster):
                    H_NLOS_temp = np.zeros((n_U, n_S), dtype=complex)

                    for m in range(link.LSP['M']):
                        PolM = np.array(
                            [[np.exp(1j * Phi_nm[n, m, 0, 0]), np.sqrt(1 / Knm[n, m]) * np.exp(1j * Phi_nm[n, m, 0, 1])],
                             [np.sqrt(1 / Knm[n, m]) * np.exp(1j * Phi_nm[n, m, 1, 0]), np.exp(1j * Phi_nm[n, m, 1, 0])]])
                        r_rx_mn = np.array([np.sin(np.deg2rad(theta_n_m_ZOA[n, m])) * np.cos(np.deg2rad(phi_n_m_AOA[n, m])),
                                            np.sin(np.deg2rad(theta_n_m_ZOA[n, m])) * np.sin(np.deg2rad(phi_n_m_AOA[n, m])),
                                            np.cos(np.deg2rad(theta_n_m_ZOA[n, m]))]).T
                        r_tx_mn = np.array([np.sin(np.deg2rad(theta_n_m_ZOD[n, m])) * np.cos(np.deg2rad(phi_n_m_AOD[n, m])),
                                            np.sin(np.deg2rad(theta_n_m_ZOD[n, m])) * np.sin(np.deg2rad(phi_n_m_AOD[n, m])),
                                            np.cos(np.deg2rad(theta_n_m_ZOD[n, m]))]).T
                        r_rx_vel = ue.velocity_ms * np.array(
                            [np.sin(np.deg2rad(ue.mobility_direction_theta)) * np.cos(np.deg2rad(ue.mobility_direction_phi)),
                             np.sin(np.deg2rad(ue.mobility_direction_theta)) * np.sin(np.deg2rad(ue.mobility_direction_phi)),
                             np.cos(np.deg2rad(ue.mobility_direction_theta))]).T

                        Frx_u = ue.antenna_panels[0].vector_field_transformation_from_gcs(theta_n_m_ZOA[n, m],
                                                                                          phi_n_m_AOA[n, m])
                        Ftx_s = bs.sector[0].antenna_panels[0].vector_field_transformation_from_gcs(theta_n_m_ZOD[n, m],
                                                                                                    phi_n_m_AOD[n, m])

                        for u in range(n_U):
                            for s in range(n_S):
                                exp_rx = np.exp(1j * 2 * np.pi * np.dot(r_rx_mn, d_rx[u]) / lambda_0)
                                exp_tx = np.exp(1j * 2 * np.pi * np.dot(r_tx_mn, d_tx[s]) / lambda_0)
                                exp_vel = np.exp(1j * 2 * np.pi * np.dot(r_rx_mn, r_rx_vel) / lambda_0)
                                H_NLOS_temp[u, s] += np.dot(Frx_u.T, np.dot(PolM, Ftx_s)) * exp_rx * exp_tx * exp_vel

                    H_NLOS_temp = np.sqrt(P_n[n] / link.LSP['M']) * H_NLOS_temp
                    channel.add_cluster(H=H_NLOS_temp, tau=cluster_delay[n])

                if link.los == LOS.LOS:
                    start_subcluster = 1
                else:
                    start_subcluster = 0


                subcluster_mapping = {0: [1, 2, 3, 4, 5, 6, 7, 8, 19, 20],
                                      1: [9, 10, 11, 12, 17, 18],
                                      2: [13, 14, 15, 16]}
                delay_offset = np.array([0, 1.28 * link.LSP['c_DS'], 2.56 * link.LSP['c_DS']])
                for n in [nn for nn in range(start_subcluster, 2) if nn <= (updated_N_cluster - 1)]:
                    H_NLOS_temp = np.zeros((n_U, n_S), dtype=complex)
                    for sub_cluster in range(3):
                        # Make sure it won't go over the existing number of rays
                        for m in [x for x in subcluster_mapping[sub_cluster] if x <= (link.LSP['M'] - 1)]:
                            PolM = np.array(
                                [[np.exp(1j * Phi_nm[n, m, 0, 0]), np.sqrt(1 / Knm[n, m]) * np.exp(1j * Phi_nm[n, m, 0, 1])],
                                 [np.sqrt(1 / Knm[n, m]) * np.exp(1j * Phi_nm[n, m, 1, 0]), np.exp(1j * Phi_nm[n, m, 1, 0])]])

                            r_rx_mn = np.array([np.sin(np.deg2rad(theta_n_m_ZOA[n, m])) * np.cos(np.deg2rad(phi_n_m_AOA[n, m])),
                                                np.sin(np.deg2rad(theta_n_m_ZOA[n, m])) * np.sin(np.deg2rad(phi_n_m_AOA[n, m])),
                                                np.cos(np.deg2rad(theta_n_m_ZOA[n, m]))]).T

                            r_tx_mn = np.array([np.sin(np.deg2rad(theta_n_m_ZOD[n, m])) * np.cos(np.deg2rad(phi_n_m_AOD[n, m])),
                                                np.sin(np.deg2rad(theta_n_m_ZOD[n, m])) * np.sin(np.deg2rad(phi_n_m_AOD[n, m])),
                                                np.cos(np.deg2rad(theta_n_m_ZOD[n, m]))]).T

                            r_rx_vel = ue.velocity_ms * np.array(
                                [np.sin(np.deg2rad(ue.mobility_direction_theta)) * np.cos(
                                    np.deg2rad(ue.mobility_direction_phi)),
                                 np.sin(np.deg2rad(ue.mobility_direction_theta)) * np.sin(
                                     np.deg2rad(ue.mobility_direction_phi)),
                                 np.cos(np.deg2rad(ue.mobility_direction_theta))]).T

                            Frx_u = ue.antenna_panels[0].vector_field_transformation_from_gcs(theta_n_m_ZOA[n, m],
                                                                                              phi_n_m_AOA[n, m])
                            Ftx_s = bs.sector[0].antenna_panels[0].vector_field_transformation_from_gcs(theta_n_m_ZOD[n, m],
                                                                                                        phi_n_m_AOD[n, m])

                            for u in range(n_U):
                                for s in range(n_S):
                                    exp_rx = np.exp(1j * 2 * np.pi * np.dot(r_rx_mn, d_rx[u]) / lambda_0)
                                    exp_tx = np.exp(1j * 2 * np.pi * np.dot(r_tx_mn, d_tx[s]) / lambda_0)
                                    exp_vel = np.exp(1j * 2 * np.pi * np.dot(r_rx_mn, r_rx_vel) / lambda_0)
                                    H_NLOS_temp[u, s] += np.dot(Frx_u.T, np.dot(PolM, Ftx_s)) * exp_rx * exp_tx * exp_vel

                        H_NLOS_temp = np.sqrt(P_n[n] / link.LSP['M']) * H_NLOS_temp
                        channel.add_cluster(H=gain_NLOS * H_NLOS_temp, tau=cluster_delay[n] + 1E-9 * delay_offset[sub_cluster])

                ################################################################################################################
                # Step 12: Apply pathloss and shadowing for the channel coefficients.
                # Todo: Pathloss needs update to compute per frequency

                self._Sector_UE_link_container[sector_ue_link].Channel = channel

            return 0

    def computeRSRP(self):
        """
        Compute the RSRP
        :return: update Network attribute RsrpMatrix
        """
        # for link in self.BS_UE_link_container:
        #     sec = self.BSs[self._BS_UE_link_container[link].base_station_ID].sector[0]
        #     self._BS_UE_link_container[link].RSRP = sec.tx_power_dB - 10 * np.log10(12 * sec.number_of_PRBs) \
        #                                             - self._BS_UE_link_container[link].pathloss

        for link in self.BS_UE_link_container:
            ue_id = self.BS_UE_link_container[link].ue_ID
            for sec in self.BSs[self._BS_UE_link_container[link].base_station_ID].sector:
                sector_id = sec.ID
                sector_ue_link_key = (sector_id, ue_id)

                self._Sector_UE_link_container[sector_ue_link_key].RSRP = sec.tx_power_dB - 10 * np.log10(
                    12 * sec.number_of_PRBs) - self._Sector_UE_link_container[sector_ue_link_key].pathloss


    def UE_attach(self):
        """
        Performs UE attachment -> finds the sector for which UE senses the highest RSRP and 'connect' to them
        :return: updates attributes 'serving_sector' and 'serving_base_station' of the UserEquipment object
        """
        for ue in self.UEs:

            # Find links related to the ue:
            sector_ue_links = [sector_id for (sector_id, ue_id) in self.Sector_UE_link_container if ue_id == ue.ID]
            ue_links_rsrp = np.array([self.Sector_UE_link_container[(sector_id, ue.ID)].RSRP for sector_id in sector_ue_links])

            highestRSRP_idx = np.argmax(ue_links_rsrp)
            highestRSRP_link = sector_ue_links[highestRSRP_idx]

            if ue_links_rsrp[highestRSRP_idx] >= self.UE_attach_threshold:
                ue.serving_sector = highestRSRP_link
                ue.serving_base_station = self.Sector_UE_link_container[(highestRSRP_idx, ue.ID)].base_station_ID
            else:
                ue.serving_sector = None
                ue.serving_base_station = None

    def computeSINR(self):
        """
        Computes the SINR for the UEs in the UE list; it assumes that the desired signal is received from the serving
        sector and the interference is received for all other sectors.
        :param BS_list: list of BaseStation
        :param UE_list: list of UserEquipment
        :return:
        """

        self.SINR_Matrix = np.zeros(len(self.UEs))

        for ue in self.UEs:
            if (ue.serving_sector is not None) and (ue.serving_base_station is not None):
                # Find all links for the UE
                bs_links = [bs_id for (bs_id, ue_id) in self.BS_UE_link_container if ue_id == ue.ID]
                sec_link = [sec.ID for bs_id in bs_links for sec in self.BSs[bs_id].sector]

                signal_power_dB = self._Sector_UE_link_container[(ue.serving_sector, ue.ID)].RSRP

                # interference_plus_noise = 10 ** (ue.noise_floor/ 10)

                # Adjust noise level to the number of PRBs
                bs_sector_id = [x for x, s in enumerate(self.BSs[ue.serving_base_station].sector) if s.ID == ue.serving_sector][0]
                interference_plus_noise = 10 ** ((ue.noise_floor - 10 * np.log10(12 * self.BSs[ue.serving_base_station].sector[bs_sector_id].number_of_PRBs)) / 10)

                for sector_id in (np.delete(np.array(sec_link), ue.serving_sector)):
                    interference_plus_noise = interference_plus_noise + 10 ** (self._Sector_UE_link_container[(sector_id, ue.ID)].RSRP / 10)

                interference_plus_noise_dB = 10 * np.log10(interference_plus_noise)
                self.SINR_Matrix[ue.ID] = signal_power_dB - interference_plus_noise_dB
            else:
                self.SINR_Matrix[ue.ID] = -np.inf

        pass

    @property
    def cellSectorMap(self):
        self._cellSectorMap = []
        for bs in self.BSs:
            self._cellSectorMap += [None] * bs.number_of_sectors
            for sec in bs.sector:
                self._cellSectorMap[sec.ID] = bs.ID

    @property
    def number_of_ue(self):
        return len(self.UEs)

    @property
    def number_of_bs(self):
        return len(self.BSs)

    @property
    def network_limits(self):
        min_x = np.inf
        min_y = np.inf
        max_x = -np.inf
        max_y = -np.inf

        for bs in self.BSs:
            if (bs.pos_x < min_x):
                min_x = bs.pos_x
            if (bs.pos_x > max_x):
                max_x = bs.pos_x
            if (bs.pos_y < min_y):
                min_y = bs.pos_y
            if (bs.pos_y > max_y):
                max_y = bs.pos_y

        r = self.ISD / 2.0  # Hexagonal cell's inner radius
        R = (2.0 / np.sqrt(3.0)) * r  # Hexagonal cell's outer radius
        return [(min_x - R, min_y - R), (max_x + R, max_y + R)]


class CorrelationGrid:
    def __init__(self, x_min: float = -100, x_max: float = 100, y_min: float = -100, y_max: float = 100,
                 corr_dist: float = 10):
        # Get number of points in each axis
        self.Nx = int(np.ceil((x_max - x_min) / corr_dist)) + 1
        self.Ny = int(np.ceil((y_max - y_min) / corr_dist)) + 1
        self.corr_dist = corr_dist

        # Find center of grid
        x_center = x_min + (x_max - x_min) / 2
        y_center = y_min + (y_max - y_min) / 2

        # Update limits to be a multiple of correlation distance
        self.x_min = x_center - (self.Nx - 1) * self.corr_dist / 2
        self.x_max = x_center + (self.Nx - 1) * self.corr_dist / 2
        self.y_min = y_center - (self.Ny - 1) * self.corr_dist / 2
        self.y_max = y_center + (self.Ny - 1) * self.corr_dist / 2

        self.coord_x = np.linspace(self.x_min, self.x_max, self.Nx)
        self.coord_y = np.linspace(self.y_min, self.y_max, self.Ny)

        self.grid_values = np.random.normal(0.0, 1.0, (self.Nx, self.Ny))

    def get_correlated_value_at_point(self, point_x, point_y):
        # Find index in the grid
        idx_x = int(np.floor((point_x - self.x_min) / self.corr_dist))
        idx_y = int(np.floor((point_y - self.y_min) / self.corr_dist))

        # Find correlated value by looking at 4 points closes to (point_x, point_y)
        value = 0.0
        for xi in [idx_x, idx_x + 1]:
            for yi in [idx_y, idx_y + 1]:
                dist = np.sqrt((self.coord_x[xi] - point_x) ** 2 + (self.coord_y[yi] - point_y) ** 2)
                value += self.grid_values[xi][yi] * np.exp(-(dist / self.corr_dist))
        return value


class CoordinateSystem(object):
    _instance = None

    @classmethod
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CoordinateSystem, cls).__new__(cls)
        return cls._instance

    @classmethod
    def GCS2LCS_point(cls, alpha, beta, gamma, point):
        return cls._inverse_rotation(alpha=alpha, beta=beta, gamma=gamma, point=point)

    @classmethod
    def GCS2LCS_angle(cls, alpha, beta, gamma, theta, phi):
        rho_unit = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]).T
        z_i = np.array([0, 0, 1]).T
        theta_prime = np.arccos(
            np.matmul(z_i, cls._inverse_rotation(alpha=alpha, beta=beta, gamma=gamma, point=rho_unit)))

        # z_i = np.array([1, 1j, 0]).T
        # phi_prime = np.angle(
        #     np.matmul(z_i, self._inverse_rotation(alpha=alpha, beta=beta, gamma=gamma, point=rho_unit)))
        z_i = cls._inverse_rotation(alpha=alpha, beta=beta, gamma=gamma, point=rho_unit)
        phi_prime = np.arctan2(z_i[1], z_i[0])

        return theta_prime, phi_prime

    @classmethod
    def get_angles(cls, point):
        rho_unit = point/np.sqrt(np.dot(point,point))
        z_i = np.array([0, 0, 1]).T
        theta = np.arccos(np.matmul(z_i, rho_unit))

        z_i = np.array([1, 1j, 0]).T
        # phi = np.angle(np.matmul(z_i, rho_unit))
        phi = np.arctan2(rho_unit[1],rho_unit[0])

        return theta, phi

    @staticmethod
    def _rotation(alpha, beta, gamma, point):
        Rz = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                       [np.sin(alpha), np.cos(alpha), 0],
                       [0, 0, 1]])

        Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                       [0, 1, 0],
                       [-np.sin(beta), 0, np.cos(beta)]])

        Rx = np.array([[1, 0, 0],
                       [0, np.cos(gamma), -np.sin(gamma)],
                       [0, np.sin(gamma), np.cos(gamma)]])

        R = np.matmul(Rz, np.matmul(Ry, Rx))

        return np.matmul(R, point)

    @staticmethod
    def _rotation_matrix(alpha, beta, gamma):
        Rz = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                       [np.sin(alpha), np.cos(alpha), 0],
                       [0, 0, 1]])

        Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                       [0, 1, 0],
                       [-np.sin(beta), 0, np.cos(beta)]])

        Rx = np.array([[1, 0, 0],
                       [0, np.cos(gamma), -np.sin(gamma)],
                       [0, np.sin(gamma), np.cos(gamma)]])

        return np.matmul(Rz, np.matmul(Ry, Rx))

    @staticmethod
    def _inverse_rotation(alpha, beta, gamma, point):
        Rz = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                       [np.sin(alpha), np.cos(alpha), 0],
                       [0, 0, 1]])

        Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                       [0, 1, 0],
                       [-np.sin(beta), 0, np.cos(beta)]])

        Rx = np.array([[1, 0, 0],
                       [0, np.cos(gamma), -np.sin(gamma)],
                       [0, np.sin(gamma), np.cos(gamma)]])

        R = np.matmul(Rz, np.matmul(Ry, Rx)).T

        return np.matmul(R, point)

    @staticmethod
    def _spherical_unit_vector_theta(theta, phi):
        return np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)]).T

    @ staticmethod
    def _spherical_unit_vector_phi(theta, phi):
        return np.array([-np.sin(phi), np.cos(phi), 0]).T

    @classmethod
    def polarized_field_component_transformation_matrix(cls, alpha, beta, gamma, theta_lcs, phi_lcs, theta_gcs, phi_gcs):
        unit_gcs_theta = cls._spherical_unit_vector_theta(theta_gcs, phi_gcs)
        unit_gcs_phi = cls._spherical_unit_vector_phi(theta_gcs, phi_gcs)

        unit_lcs_theta = cls._spherical_unit_vector_theta(theta_lcs, phi_lcs)
        unit_lcs_phi = cls._spherical_unit_vector_phi(theta_lcs, phi_lcs)

        R = cls._rotation_matrix(alpha, beta, gamma)
        M = np.matmul(R, np.vstack((unit_lcs_theta, unit_lcs_phi)).T)
        M = np.matmul(np.vstack((unit_gcs_theta, unit_gcs_phi)), M)

        return M


def dist2d(device1: Union[BaseStation, UserEquipment], device2: Union[BaseStation, UserEquipment]):
    return np.sqrt(np.power(device1.pos_x - device2.pos_x, 2) + np.power(device1.pos_y - device2.pos_y, 2))


def dist3d(device1: Union[BaseStation, UserEquipment], device2: Union[BaseStation, UserEquipment]):
    return np.sqrt(np.power(device1.pos_x - device2.pos_x, 2)
                   + np.power(device1.pos_y - device2.pos_y, 2)
                   + np.power(device1.height - device2.height, 2))
