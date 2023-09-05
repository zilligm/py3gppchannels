import numpy as np
import itertools
from typing import List, Union
# from scipy.signal import convolve2d
from scipy.linalg import sqrtm
# from scipy.interpolate import RegularGridInterpolator
from enum import Enum
from dataclasses import dataclass
import numpy.typing as npt

light_speed = 299_792_458  # m/s

C_phi_NLOS_dict = {4: 0.779, 5: 0.860, 8: 1.018, 10: 1.090, 11: 1.123, 12: 1.146, 14: 1.190, 15: 1.211, 16: 1.226,
                   19: 1.273, 20: 1.289, 25: 1.358}

C_theta_NLOS_dict = {8: 0.889, 10: 0.957, 11: 1.031, 12: 1.104, 15: 1.1088, 19: 1.184, 20: 1.178, 25: 1.282}

alpha_m = np.array([0.0447, -0.0447, 0.1413, -0.1413, 0.2492, -0.2492, 0.3715, -0.3715, 0.5129, -0.5129,
                    0.6797, -0.6797, 0.8844, -0.8844, 1.1481, -1.1481, 1.5195, -1.5195, 2.1551, -2.1551])

subcluster_mapping = {0: [1, 2, 3, 4, 5, 6, 7, 8, 19, 20],
                      1: [9, 10, 11, 12, 17, 18],
                      2: [13, 14, 15, 16]}

COMPUTE_ALL_FREQUENCIES = False


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        else:
            cls._instances[cls].__init__(*args, **kwargs)
        return cls._instances[cls]


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


# class POL(Enum):
#     UNDETERMINED = 0
#     LINEAR = 1
#     CROSS = 2


@dataclass
class FrequencyDependentLargeScaleParameters:
    """
    Data class: stores the frequency-dependent LSP for a BS-UE link;
    The class BsUeLink's attribute lspContainer is a dict with keys given by the frequency (GHz) and the value given by
    a FrequencyDependentLargeScaleParameters object.
    """
    Pathloss: float = 0.0
    DS: float = 0.0
    ASA: float = 0.0
    ASD: float = 0.0
    ZSA: float = 0.0
    ZSD: float = 0.0
    mu_offset_ZOD: float = 0.0
    mu_lg_ZSD: float = 0.0
    c_DS: float = 0.0

    updated_N_cluster: int = 0
    cluster_delay: npt.ArrayLike = np.array([])
    cluster_delay_LOS: npt.ArrayLike = np.array([])
    P_n: npt.ArrayLike = np.array([])
    phi_n_m_AOA: npt.ArrayLike = np.array([])
    phi_n_m_AOD: npt.ArrayLike = np.array([])
    theta_n_m_ZOA: npt.ArrayLike = np.array([])
    theta_n_m_ZOD: npt.ArrayLike = np.array([])
    # ray_mapping_AoD_AoA: npt.ArrayLike = np.array([])
    # ray_mapping_ZoD_ZoA: npt.ArrayLike = np.array([])
    # ray_mapping_AoD_ZoD: npt.ArrayLike = np.array([])
    Knm: npt.ArrayLike = np.array([])


@dataclass
class FrequencyIndependentLargeScaleParameters:
    """
    Data class: stores the frequency-dependent LSP for a BS-UE link;
    The class BsUeLink's attribute frequency_independent_lsp stores the FrequencyIndependentLargeScaleParameters object.
    """
    Pathloss: float = 0.0
    SF: float = 0.0
    N: int = 0
    M: int = 0
    r_tau: float = 0.0
    c_ASA: float = 0.0
    c_ASD: float = 0.0
    c_ZSA: float = 0.0
    xi: float = 0.0
    mu_XPR: float = 0.0
    sigma_xpr: float = 0.0
    K: float = 0.0


@dataclass
class SmallScaleParameters:
    pass
    # Pathloss: float = 0.0
    # DS: float = 0.0
    # ASA: float = 0.0
    # ASD: float = 0.0
    # ZSA: float = 0.0
    # ZSD: float = 0.0
    # SF: float = 0.0
    # N: int = 0
    # M: int = 0
    # r_tau: float = 0.0
    # c_DS: float = 0.0
    # c_ASA: float = 0.0
    # c_ASD: float = 0.0
    # c_ZSA: float = 0.0
    # xi: float = 0.0
    # mu_offset_ZOD: float = 0.0
    # mu_lg_ZSD: float = 0.0
    # mu_XPR: float = 0.0
    # sigma_xpr: float = 0.0
    # K: float = 0.0      ## Only used in LOS


class ChannelBands(metaclass=Singleton):
    _center_frequency_ghz = 3.5
    _channel_bandwidth_mhz = 10.0
    _number_of_channels = 1

    def __init__(cls, center_frequency_ghz: float = None, channel_bandwidth_mhz: float = None,
                 number_of_channels: int = None):
        if center_frequency_ghz is None:
            cls.center_frequency_ghz = ChannelBands._center_frequency_ghz
        else:
            cls.center_frequency_ghz = center_frequency_ghz

        if channel_bandwidth_mhz is None:
            cls.channel_bandwidth_mhz = ChannelBands._channel_bandwidth_mhz
        else:
            cls.channel_bandwidth_mhz = channel_bandwidth_mhz

        if number_of_channels is None:
            cls.number_of_channels = ChannelBands._number_of_channels
        else:
            cls.number_of_channels = number_of_channels

        ChannelBands._center_frequency_ghz = cls.center_frequency_ghz
        ChannelBands._channel_bandwidth_mhz = cls.channel_bandwidth_mhz
        ChannelBands._number_of_channels = cls.number_of_channels

        cls._frequency_list = list()
        cls.channel_bands = []
        fc_start = cls.center_frequency_ghz \
                   - ((cls.channel_bandwidth_mhz * 0.001 * cls.number_of_channels) / 2) \
                   + (cls.channel_bandwidth_mhz * 0.001 / 2)

        fc_end = cls.center_frequency_ghz \
                 + ((cls.channel_bandwidth_mhz * 0.001 * cls.number_of_channels) / 2) \
                 - (cls.channel_bandwidth_mhz * 0.001 / 2)

        fc_list = np.linspace(start=fc_start, stop=fc_end, num=cls.number_of_channels)
        for n in range(cls.number_of_channels):
            fc = fc_list[n]
            cls.channel_bands.append({'fc': fc, 'BW': cls.channel_bandwidth_mhz * 0.001})

    @property
    def frequency_list(cls):
        cls._frequency_list = list()
        for channel in cls.channel_bands:
            cls._frequency_list.append(channel['fc'])

        return cls._frequency_list


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
            A_dB_vertical_cut = -np.minimum(12 * ((np.rad2deg(theta_lcs) - 90) / 65) ** 2, 30)
            A_dB_horizontal_cut = -np.minimum(12 * (np.rad2deg(phi_lcs) / 65) ** 2, 30)
            A_dB = -np.minimum(-(A_dB_vertical_cut + A_dB_horizontal_cut), 30)
            return A_dB
        else:
            raise 'Invalid radiation pattern model'

        # TODO: How does the "Maximum directional gain" come into play?

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
    _default_n_panel_col = 1
    _default_n_panel_row = 1
    _default_panel_v_spacing = 2
    _default_panel_h_spacing = 2
    _default_n_antenna_col = 1
    _default_n_antenna_row = 1
    _default_antenna_v_spacing = 0.5
    _default_antenna_h_spacing = 0.5
    _default_polarization = 2
    _default_polarization_slant = (-45.0, 45.0)
    _default_radiation_pattern_model = '3GPP'
    _default_bearing = 0.0
    _default_downtilt = 0.0
    _default_slant = 0.0

    def __init__(self, n_panel_col: int = None, n_panel_row: int = None,
                 panel_v_spacing: float = None, panel_h_spacing: float = None,
                 n_antenna_col: int = None, n_antenna_row: int = None,
                 antenna_v_spacing: float = None, antenna_h_spacing: float = None,
                 polarization: int = None, polarization_slant: tuple = None,
                 radiation_pattern_model: str = None,
                 bearing: float = None, downtilt: float = None, slant: float = None):
        """
        Create the antenna panel object. Each antenna panel contains an antenna array.
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
        :param polarization: polarization, either 1 or 2 [linear, cross-pol]
        :param polarization_slant: (-45,45) for eNB or (0,90) for UE
        :param radiation_pattern_model: indicates the radiation pattern model that is used to generate the antenna
                element pattern
        """
        # ##############################################################################################################
        # Panel geometry
        # ##############################################################################################################
        if n_panel_col is None:
            self.n_panel_col = AntennaPanel._default_n_panel_col
        else:
            self.n_panel_col = n_panel_col

        if n_panel_row is None:
            self.n_panel_row = AntennaPanel._default_n_panel_row
        else:
            self.n_panel_row = n_panel_row

        if panel_v_spacing is None:
            self.panel_v_spacing = AntennaPanel._default_panel_v_spacing
        else:
            self.panel_v_spacing = panel_v_spacing

        if panel_h_spacing is None:
            self.panel_h_spacing = AntennaPanel._default_panel_h_spacing
        else:
            self.panel_h_spacing = panel_h_spacing

        # ##############################################################################################################
        # Array geometry
        # ##############################################################################################################
        if n_antenna_col is None:
            self.n_antenna_col = AntennaPanel._default_n_antenna_col
        else:
            self.n_antenna_col = n_antenna_col

        if n_antenna_row is None:
            self.n_antenna_row = AntennaPanel._default_n_antenna_row
        else:
            self.n_antenna_row = n_antenna_row

        if antenna_v_spacing is None:
            self.antenna_v_spacing = AntennaPanel._default_antenna_v_spacing
        else:
            self.antenna_v_spacing = antenna_v_spacing

        if antenna_h_spacing is None:
            self.antenna_h_spacing = AntennaPanel._default_antenna_h_spacing
        else:
            self.antenna_h_spacing = antenna_h_spacing

        # ##############################################################################################################
        # Orientation
        # ##############################################################################################################
        if bearing is None:  # alpha
            # self.bearing_angle = np.random.uniform(0, 2 * np.pi)
            self.bearing_angle = 0.0
        else:
            self.bearing_angle = bearing

        if downtilt is None:  # beta
            # self.downtilt_angle = np.random.uniform(0, 2 * np.pi)
            self.downtilt_angle = 0.0
        else:
            self.downtilt_angle = downtilt

        if slant is None:  # gamma
            # self.slant = np.random.uniform(0, 2 * np.pi)
            self.slant = 0.0
        else:
            self.slant = slant

        # ##############################################################################################################
        # Polarization
        # ##############################################################################################################
        if polarization is None:
            self.polarization = AntennaPanel._default_polarization
        else:
            if polarization in [1, 2]:
                self.polarization = polarization
            else:
                raise 'Invalid polarization: polarization should be either 1 or 2'

        if polarization_slant is None:
            self.polarization_slant = AntennaPanel._default_polarization_slant
        else:
            if polarization_slant in [(-45, 45), (0, 90)]:
                self.polarization_slant = polarization_slant
            else:
                raise 'Invalid polarization slant: ' \
                      'polarization slant should be either [-45,45] (for eNBs) or [0, 90] for UEs'

        # ##############################################################################################################
        # Antenna Element
        # ##############################################################################################################
        if radiation_pattern_model is None:
            self.radiation_pattern_model = AntennaPanel._default_radiation_pattern_model
        else:
            self.radiation_pattern_model = radiation_pattern_model

        self.antenna_element = AntennaElement(radiation_pattern_model=self.radiation_pattern_model)

        # ##############################################################################################################
        # Array location vector
        # ##############################################################################################################
        self._array_location_tensor = self.array_location_tensor
        self._array_location_vector = self.array_location_vector

        # ##############################################################################################################
        # Coordinate System
        # ##############################################################################################################
        self.coordinate_system = CoordinateSystem

    def vector_field_transformation_from_gcs(self, theta_gcs, phi_gcs):
        alpha = np.deg2rad(self.bearing_angle)
        beta = np.deg2rad(self.downtilt_angle)
        gamma = np.deg2rad(self.slant)

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
        the respective antenna element. Ex: array_location_vector[u,v,:] = [x_uv, y_uv, z_uv] are the coordinates of the
        element in the u-th row and v-th column of the panel.
        :return: array_location_vector
        """
        self._array_location_tensor = np.zeros((self.n_panel_row * self.n_antenna_row,
                                                self.n_panel_col * self.n_antenna_col, 3))

        y_axis = np.array([p + a for p in np.arange(self.n_panel_row) * self.panel_h_spacing for a in
                           np.arange(self.n_antenna_row) * self.antenna_h_spacing])

        z_axis = np.array([p + a for p in np.arange(self.n_panel_col) * self.panel_v_spacing for a in
                           np.arange(self.n_antenna_col) * self.antenna_v_spacing])

        A = np.meshgrid(y_axis, z_axis, indexing='ij')

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

    def panel_field_pattern_vector(self, theta_gcs, phi_gcs, zeta):
        theta_lcs, phi_lcs = CoordinateSystem.GCS2LCS_angle(alpha=np.deg2rad(self.bearing_angle),
                                                            beta=np.deg2rad(self.downtilt_angle),
                                                            gamma=np.deg2rad(self.slant),
                                                            theta=theta_gcs,
                                                            phi=phi_gcs)

        field_pattern_vector_lcs = self.antenna_element.field_pattern_vector(theta_lcs, phi_lcs, zeta)

        return field_pattern_vector_lcs

    @staticmethod
    def set_default(n_panel_col: int = None, n_panel_row: int = None,
                    panel_v_spacing: float = None, panel_h_spacing: float = None,
                    n_antenna_col: int = None, n_antenna_row: int = None,
                    antenna_v_spacing: float = None, antenna_h_spacing: float = None,
                    polarization: int = None, polarization_slant: tuple = None, radiation_pattern_model: str = None,
                    bearing: float = None, downtilt: float = None, slant: float = None):
        """
        Create the antenna panel object. Each antenna panel contains an antenna array.
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
        :param polarization: polarization, either 1 (linear) or 2 (cross-pol)
        :param polarization_slant: (-45,45) for eNB or (0,90) for UE
        :param radiation_pattern_model: indicates the radiation pattern model that is used to generate the antenna
                element pattern
        :param bearing: bearing angle
        :param downtilt: downtilt angle
        :param slant: slant angle
        """

        if n_panel_col is not None:
            AntennaPanel._default_n_panel_col = n_panel_col

        if n_panel_row is not None:
            AntennaPanel._default_n_panel_row = n_panel_row

        if panel_v_spacing is not None:
            AntennaPanel._default_panel_v_spacing = panel_v_spacing

        if panel_h_spacing is not None:
            AntennaPanel._default_panel_h_spacing = panel_h_spacing

        if n_antenna_col is not None:
            AntennaPanel._default_n_antenna_col = n_antenna_col

        if n_antenna_row is not None:
            AntennaPanel._default_n_antenna_row = n_antenna_row

        if antenna_v_spacing is not None:
            AntennaPanel._default_antenna_v_spacing = antenna_v_spacing

        if antenna_h_spacing is not None:
            AntennaPanel._default_antenna_h_spacing = antenna_h_spacing

        if polarization is not None:
            if polarization in [1, 2]:
                AntennaPanel._default_polarization = polarization
            else:
                raise 'Invalid polarization: polarization slant should be either 1 or 2'

        if polarization_slant is not None:
            if polarization_slant in [(-45, 45), (0, 90)]:
                AntennaPanel._default_polarization_slant = polarization_slant
            else:
                raise 'Invalid polarization slant: ' \
                      'polarization slant should be either [-45,45] (for eNBs) or [0, 90] for UEs'

        if radiation_pattern_model is not None:
            AntennaPanel._default_radiation_pattern_model = radiation_pattern_model

        if bearing is not None:
            AntennaPanel._default_bearing = bearing

        if downtilt is not None:
            AntennaPanel._default_downtilt = downtilt

        if slant is not None:
            AntennaPanel._default_slant = slant


class Beamforming:
    def __init__(self, antenna_panel: AntennaPanel, beamforming_type='dft'):
        # Ng = antenna_panel.n_panel_col
        # Mg = antenna_panel.n_panel_row
        # dg_V = antenna_panel.panel_v_spacing
        # dg_H = antenna_panel.panel_h_spacing
        # N = antenna_panel.n_antenna_col
        # M = antenna_panel.n_antenna_row
        # d_v = antenna_panel.antenna_v_spacing
        # d_H = antenna_panel.antenna_h_spacing
        # pol = antenna_panel.polarization
        self.codebook = np.zeros
        self.antenna_panel = antenna_panel
        self.beamforming_type = beamforming_type


class Sector:
    """
    Sector class defines each sector of a base-station; it contains information about the sector orientation (bearing,
    downtilt, and slant angles), the Tx power, sector ID and corresponding base-station ID, operating frequency
    """
    # Auto ID generation
    _sector_id = itertools.count()

    _default_bearing = 0.0
    _default_downtilt = 0.0
    _default_slant = 0.0
    _default_sector_width = 120.0
    # _default_frequency_ghz = 3.5
    _default_tx_power_dBm = 30.0

    def __init__(self, bs_id: int,
                 bearing: float = None, downtilt: float = None, slant: float = None, sector_width: float = None,
                 frequency_ghz: float = None, tx_power_dBm: float = None, antenna_panels: List[AntennaPanel] = None):
        """
        Create sectors within a Base Station
        :param bs_id: Base Station ID
        :param bearing: Bearing angle [degrees]
        :param downtilt: Downtilt angle [degrees]
        :param slant: Slant angle [degrees]
        :param sector_width: width of the sector [degrees]
        :param frequency_ghz: operation frequency of the sector [GHz]
        :param tx_power_dBm: transmission power [dBm]
        :param antenna_panels: list of antenna panels (if empty, will create a default antenna panel)
        """
        self.ID = next(Sector._sector_id)
        self.BS_ID = bs_id

        # ##############################################################################################################
        # Orientation
        # ##############################################################################################################
        if bearing is None:  # alpha
            self.bearing_angle = Sector._default_bearing
        else:
            self.bearing_angle = bearing

        if downtilt is None:  # beta
            self.downtilt_angle = Sector._default_downtilt
        else:
            self.downtilt_angle = downtilt

        if slant is None:  # gamma
            self.slant = Sector._default_slant
        else:
            self.slant = slant

        # ##############################################################################################################
        # Sector Width
        # ##############################################################################################################
        if sector_width is None:
            self.sector_width = Sector._default_sector_width
        else:
            self.sector_width = sector_width

        # ##############################################################################################################
        # Transmitted Power
        # ##############################################################################################################
        if tx_power_dBm is None:
            self.tx_power_dBm = Sector._default_tx_power_dBm
        else:
            self.tx_power_dBm = tx_power_dBm

        # ##############################################################################################################
        # Channel Frequency Info
        # ##############################################################################################################
        if frequency_ghz is None:
            channel_bands = ChannelBands()
            self.frequency_ghz = channel_bands.frequency_list[0]
            # self.frequency_ghz = Sector._default_frequency_ghz
        else:
            self.frequency_ghz = frequency_ghz

        # ##############################################################################################################
        # Antenna Panel
        # ##############################################################################################################
        if antenna_panels is None:
            self.antenna_panels = []
            self.add_antenna_panel(AntennaPanel(bearing=self.bearing_angle,
                                                downtilt=self.downtilt_angle,
                                                slant=self.slant, polarization_slant=(-45, 45)))
        else:
            self.antenna_panels = antenna_panels

        self.number_of_PRBs = 50  # 10 MHz channel with SCS 15 KHz (LTE)

    def add_antenna_panel(self, ant_panel: AntennaPanel):
        if ant_panel.__class__ == AntennaPanel:
            self.antenna_panels.append(ant_panel)

    @property
    def number_of_antenna_panels(self):
        """
        :return: Number of antenna panels in the Sector
        """
        return len(self.antenna_panels)

    @property
    def number_of_antennas(self):
        """
        :return: Total number of antenna elements in the Sector
        """
        number_of_antennas = 0
        for panel in self.antenna_panels:
            number_of_antennas += panel.n_antenna_col * panel.n_panel_col * panel.n_antenna_row * panel.n_panel_row
        return number_of_antennas

    @staticmethod
    def set_default(bearing: float = None, downtilt: float = None, slant: float = None, sector_width: float = None,
                    frequency_ghz: float = None, tx_power_dBm: float = None):
        """
        :param bearing: Bearing angle [degrees]
        :param downtilt: Downtilt angle [degrees]
        :param slant: Slant angle [degrees]
        :param sector_width: width of the sector [degrees]
        :param frequency_ghz: operation frequency of the sector [GHz]
        :param tx_power_dBm: transmission power [dBm]
        """

        if bearing is not None:
            AntennaPanel._default_bearing = bearing

        if downtilt is not None:
            AntennaPanel._default_downtilt = downtilt

        if slant is not None:
            AntennaPanel._default_slant = slant

        if sector_width is not None:
            AntennaPanel._default_sector_width = sector_width

        if frequency_ghz is not None:
            AntennaPanel._default_frequency_ghz = frequency_ghz

        if tx_power_dBm is not None:
            AntennaPanel._default_tx_power_dBm = tx_power_dBm


class BaseStation:
    """
    BaseStation class defines a base-station; it contains information about the sector orientation (bearing,
    downtilt, and slant angles), the Tx power, sector ID and corresponding base-station ID, operating frequency
    """
    # Auto ID generation
    _bs_id = itertools.count()

    def __init__(self, pos_x: float = 0, pos_y: float = 0, height: float = 10, number_of_sectors: int = 3,
                 tx_power_dBm: float = 30, rotation: float = 0):
        """
        Create a Base Station
        :param pos_x: position of the Base Station in the x axis [meters]
        :param pos_y: position of the Base Station in the y axis [meters]
        :param height: height of the Base Station [meters]
        :param number_of_sectors: number of sectors in the Base Station
        :param tx_power_dBm: transmission power [dBm]
        :param rotation: offset to the Base Station' sector orientation [degrees]
        """
        self.ID = next(BaseStation._bs_id)

        # Positioning
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.height = height

        # Network Information
        if tx_power_dBm is None:
            self.tx_power_dBm = 30
        else:
            self.tx_power_dBm = tx_power_dBm

        # Initialize sectors
        self.sectors = []
        self.number_of_sectors = number_of_sectors
        self.sector_width = 360 / number_of_sectors
        for sec in range(number_of_sectors):
            bearing = rotation + sec * 360 / number_of_sectors

            self.add_sector(bearing=bearing, downtilt=0, slant=0, sector_width=self.sector_width, frequency_ghz=None,
                            tx_power_dBm=self.tx_power_dBm)

        # Update BS with Channel Frequency Information
        # Initialize frequency list:
        self._frequency_list = list()

        # Update frequency list:
        self._frequency_list = self.getBsFrequencyList()

    def add_sector(self, bearing: float = None, downtilt: float = 0, slant: float = 0, sector_width: float = 120,
                   frequency_ghz: float = None, tx_power_dBm: float = None, antenna_panels: List[AntennaPanel] = None):

        sector = Sector(bs_id=self.ID, bearing=bearing, downtilt=downtilt, slant=slant, sector_width=sector_width,
                        frequency_ghz=frequency_ghz, tx_power_dBm=tx_power_dBm, antenna_panels=antenna_panels)

        print(f'Base Station {self.ID} - Sector {sector.ID}')

        self.sectors.append(sector)

    def getBsFrequencyList(self):
        """
        :return: List of frequencies all Sectors associated with the Base Station
        """
        self._frequency_list = list()
        for sector in self.sectors:
            if sector.frequency_ghz not in self._frequency_list:
                self._frequency_list.append(sector.frequency_ghz)

        return self._frequency_list

    def get_sector_by_ID(self, ID):
        """
        :return: Return the sector with the input ID
        """
        for sec_idx, sec in enumerate(self.sectors):
            if sec.ID == ID:
                return sec
        else:
            return None


class UserEquipment:
    _id_iter = itertools.count()

    _default_height = 1.5
    _default_location = Location.Outdoor
    _default_tx_power_dBm = 23.0
    _default_noise_floor = -125.0
    _default_sector_width = 120.0
    _default_frequency_ghz = 3.5
    _default_mobility_speed_kmh = 3.0

    def __init__(self, pos_x: float = 0, pos_y: float = 0, height: float = None,
                 bearing: float = None, downtilt: float = None, slant: float = None,
                 location: Location = Location.Outdoor, tx_power_dBm: float = None, noise_floor: float = None,
                 antenna_panels: List[AntennaPanel] = None, mobility_speed_kmh: float = None):
        """
        Create User Equipment (UE)
        :param pos_x: position of the UE in the x axis [meters]
        :param pos_y: position of the UEt in the y axis [meters]
        :param height: height of the UE [meters]
        :param location: location of the UE ['Indoor','Outdoor']
        :param tx_power_dBm: transmission power [dBm]
        :param noise_floor: noise floor [dBm]
        """

        self.ID = next(UserEquipment._id_iter)

        # ##############################################################################################################
        # Positioning
        # ##############################################################################################################
        self.pos_x = pos_x
        self.pos_y = pos_y

        if height is None:
            self.height = UserEquipment._default_height
        else:
            self.height = height

        self.height = height
        self.location = location

        # ##############################################################################################################
        # Orientation
        # ##############################################################################################################
        if bearing is None:
            self.bearing_angle = np.random.uniform(0, 360)  # Omega_UE_alpha
        else:
            self.bearing_angle = bearing

        if downtilt is None:
            self.downtilt_angle = np.random.uniform(0, 360)  # Omega_UE_beta
        else:
            self.downtilt_angle = downtilt

        if slant is None:
            self.slant = np.random.uniform(0, 360)  # Omega_UE_gamma
        else:
            self.slant = slant

        # ##############################################################################################################
        # Transmit Power
        # ##############################################################################################################
        if tx_power_dBm is None:
            self.tx_power_dBm = UserEquipment._default_tx_power_dBm
        else:
            self.tx_power_dBm = tx_power_dBm

        if noise_floor is None:
            self.noise_floor = UserEquipment._default_noise_floor
        else:
            self.noise_floor = noise_floor

        # ##############################################################################################################
        # Mobility
        # ##############################################################################################################
        if mobility_speed_kmh is None:
            self.mobility_speed_kmh = UserEquipment._default_mobility_speed_kmh
        else:
            self.mobility_speed_kmh = mobility_speed_kmh

        self.velocity_ms = self.mobility_speed_kmh * 1000 / 3600  # 3 km/h
        self.mobility_direction_phi = np.random.uniform(0, 2 * np.pi)
        self.mobility_direction_theta = 90

        # ##############################################################################################################
        # Network Info
        # ##############################################################################################################
        self.serving_sector = None
        self.serving_base_station = None
        self.neighbor_cells = []
        self.noise_floor = UserEquipment._default_noise_floor  # dB

        # ##############################################################################################################
        # Antenna Panels
        # ##############################################################################################################
        self.antenna_panels = []
        if antenna_panels:
            self.antenna_panels = antenna_panels
        else:
            self.add_antenna_panel(AntennaPanel(polarization_slant=(0, 90)))

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


class SectorUeLink:
    def __init__(self, sector: Sector, ue: UserEquipment, los: LOS, bs_sector_id: int):
        self.base_station_ID = sector.BS_ID
        self.sector_ID = sector.ID
        self.ue_ID = ue.ID
        self.bs_sector_id = bs_sector_id
        self.los = los
        self.active = False
        # Currently not used; # will be used to indicate that link is active - ie, sector is
        # serving sector for the ue

        # Random Phases for Step 10
        self.Phi_nm = []

        # Channel coefficients
        self.channel = Channel()
        self.pathloss = np.inf
        self.sigma_sf = 0.0
        self.RSRP = -np.inf

        # self.h_los_u_s = np.array([])
        # self.h_nlos_u_s_n_m = np.array([])
        # self.los_delay = np.array([])
        # self.nlos_delay = np.array([])


class BsUeLink:
    def __init__(self, bs: BaseStation, ue: UserEquipment):
        self.base_station_ID = bs.ID
        self.ue_ID = ue.ID
        self.active = False  # Currently not used; will be used to indicate that link is active - ie, sector is serving
        # sector for the ue

        self.sector_links = []  # Holds all Sector_UE links for this BS-UE pair
        self.serving_sector = None  # Indicates the serving sector if any

        # Geometry
        self.distance_2D = None
        self.distance_3D = None
        self.LOS_AOD_GCS = None
        self.LOS_ZOD_GCS = None
        self.LOS_AOA_GCS = None
        self.LOS_ZOA_GCS = None

        # Frequency Independent Large Scale Link Parameter Initialization
        self.frequency_independent_lsp = FrequencyIndependentLargeScaleParameters()

        # Frequency Dependent Large Scale Link Parameter Initialization
        if COMPUTE_ALL_FREQUENCIES is True:
            a = ChannelBands()
            frequency_list = a.frequency_list
        else:
            frequency_list = bs.getBsFrequencyList()

        self.lspContainer = {}

        for fc in frequency_list:
            self.lspContainer.setdefault(fc, FrequencyDependentLargeScaleParameters())

        self.sspContainer = {}
        for fc in frequency_list:
            self.sspContainer.setdefault(fc, SmallScaleParameters())

        # Small Scale Link parameters
        # self.AOD_nm = None
        # self.ZOD_nm = None
        # self.AOA_nm = None
        # self.ZOA_nm = None
        # self.ray_coupling = None
        # self.cross_pol_power_ratio = None


class Channel:
    def __init__(self):
        # self.Nr = Nr
        # self.Nt = Nt
        # self.cluster_matrices = []
        # self.cluster_delays = []

        self.h_los_u_s = np.array([])
        self.h_nlos_u_s_n_m = np.array([])
        self.los_delay = np.array([])
        self.nlos_delay = np.array([])

    # def add_cluster(self, H, tau):
    #     self.cluster_matrices.append(H)
    #     self.cluster_delays.append(tau)


class Network:
    # _BsUeLink_container: Type[dict]

    def __init__(self, scenario: str = 'UMa', free_space: bool = False,
                 BSs: Union[List[BaseStation], BaseStation] = None,
                 UEs: Union[List[UserEquipment], UserEquipment] = None,
                 wraparound: bool = False, seed: int = None):
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

        if BSs is None:
            self.BSs = []
        else:
            self.BSs = BSs

        if UEs is None:
            self.UEs = []
        else:
            self.UEs = UEs

        # Spectrum
        self.channel_bands = ChannelBands(center_frequency_ghz=3.5, channel_bandwidth_mhz=10, number_of_channels=2)

        self._rsrp_matrix = np.array([])
        self.SINR_Matrix = np.array([])
        self._rsrpServingCell = np.array([])

        # self.UE_attach_threshold = -140  # Todo: Use reasonable value
        self.UE_attach_threshold = -np.inf  # Todo: Use reasonable value

        # Correlation Parameters
        self._C = {}
        self._delta_m = {}
        self._LSP_list = {}
        self._Q = {}

        self._parameter_grid = {}

        # Link Containers
        self._BsUeLink_container = {}
        self._SectorUeLink_container = {}

        # Coordinate System
        self.coordinate_system = CoordinateSystem

        # Random Generators
        self.random_seed = seed
        self.random_generator_LOS = np.random.default_rng(seed=self.random_seed)
        self.random_generator_Location = np.random.default_rng(seed=self.random_seed)
        self.random_generator_PL = np.random.default_rng(seed=self.random_seed)

        # Scenario Parameters
        self.layout = ''
        self.number_of_BSs = 0.0
        self.number_of_sectors = 0.0
        self.ISD = 0.0
        self.BS_height = 0.0
        # self.UE_location = []
        # self.UE_los = ['LOS', 'NLOS']
        self.UE_height = 0.0
        self.UE_indoor_ratio = 0.0
        self.UE_mobility = 0.0
        self.min_BS_UE_distance = 0.0
        self.UE_distribution = ''

        # Optional Parameters
        self.avg_building_height = 0.0
        self.average_street_width = 0.0

        self.getScenarioAttributes()

        # Todo Implement Wrap-Around
        if wraparound:
            raise 'Wrap-around not supported!'

    @property
    def BsUeLink_container(self):
        for bs in self.BSs:
            for ue in self.UEs:
                bs_ue_key = (bs.ID, ue.ID)
                if not (bs_ue_key in self._BsUeLink_container):
                    self._BsUeLink_container[bs_ue_key] = BsUeLink(bs=bs, ue=ue)
        return self._BsUeLink_container

    @property
    def SectorUeLink_container(self):
        for ue in self.UEs:
            for bs in self.BSs:
                los = self._BsUeLink_container[(bs.ID, ue.ID)].los
                for bs_sector_id, sector in enumerate(bs.sectors):
                    sector_ue_key = (sector.ID, ue.ID)
                    if not (sector_ue_key in self._SectorUeLink_container):
                        self._SectorUeLink_container[sector_ue_key] = SectorUeLink(sector, ue, los, bs_sector_id)
        return self._SectorUeLink_container

    @property
    def parameter_grid(self):
        [(x_min, y_min), (x_max, y_max)] = self.network_limits
        for los in [LOS.LOS, LOS.NLOS]:
            for bs in self.BSs:
                for lsp in self._LSP_list[los]:
                    if not ((los.value, bs.ID, lsp) in self._parameter_grid):
                        self._parameter_grid[(los.value, bs.ID, lsp)] = CorrelationGrid(x_min=x_min, y_min=y_min,
                                                                                        x_max=x_max, y_max=y_max,
                                                                                        corr_dist=self._delta_m[los][
                                                                                            lsp])
        return self._parameter_grid

    def add_ue(self, pos_x: float = 0, pos_y: float = 0, height: float = None,
               location: Location = Location.UNDETERMINED,
               speed: float = None, tx_power_dBm: float = None):

        if height is None:
            height = self.UE_height
        if location is None:
            location = self.UELocation()
        if speed is None:
            speed = self.UE_mobility

        self.UEs.append(UserEquipment(pos_x=pos_x,
                                      pos_y=pos_y,
                                      tx_power_dBm=tx_power_dBm,
                                      height=height,
                                      location=location,
                                      mobility_speed_kmh=speed))

    def add_bs(self, pos_x: float = 0, pos_y: float = 0, height: float = None, number_of_sectors: int = None,
               rotation: float = None):
        if height is None:
            height = self.BS_height
        if number_of_sectors is None:
            number_of_sectors = self.number_of_sectors
        if rotation is None:
            rotation = 0

        self.BSs.append(BaseStation(pos_x=pos_x,
                                    pos_y=pos_y,
                                    height=height,
                                    number_of_sectors=number_of_sectors,
                                    rotation=rotation))

    # Step 1-a)
    def getScenarioAttributes(self):
        if self.scenario == 'UMi':
            self.layout = 'Hexagonal'
            self.number_of_BSs = 19
            self.number_of_sectors = 3
            self.ISD = 200  # meters
            self.BS_height = 10  # meters
            # self.UE_location = ['Outdoor', 'Indoor']
            # self.UE_los = ['LOS', 'NLOS']
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
            # self.UE_location = ['Outdoor', 'Indoor']
            # self.UE_los = ['LOS', 'NLOS']
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
            # self.UE_location = ['Indoor', 'Car']
            # self.UE_los = ['LOS', 'NLOS']
            self.UE_height = 1.5  # meters    # Todo: Check TR 36.873
            self.UE_indoor_ratio = 0.50  # 0..1
            self.UE_mobility = 3  # km/h
            self.min_BS_UE_distance = 35  # meters
            self.UE_distribution = 'Uniform'

            # Optional Parametes
            self.avg_building_height = 5.0  # meters [5..50]
            self.average_street_width = 5.0  # meters [5..50]

        # TODO: Implement other scenarios:  Indoor Factory (InF) - (InF-SL, InF-DL, InF-SH, InF-DH, InF-HH)
        #                                   Indoor Office/Hotspot (InH)

    # Step 1-c) and 1-d)
    def computeGeometry(self):
        """
        Computes the 2D and 3D distances and the line-of-sight azimuth and zenith angles between all BSs and UEs.
        Results are stored in the class attributes 'dist2D_Matrix', 'dist3D_Matrix', 'los_azi_angle_rad_Matrix', and
        'los_zen_angle_rad_Matrix'.
        """

        for link in self.BsUeLink_container.keys():
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
            xy_position = -xy_position  # Invert xy_position
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
            h_e = -h_e  # Invert h_e
            zoa_rad_gcs = np.pi / 2  # If h_e == 0
            if h_e > 0:
                zoa_rad_gcs = np.pi - np.arctan(dist_2D / h_e)
            elif h_e < 0:
                zoa_rad_gcs = - np.arctan(dist_2D / h_e)

            self._BsUeLink_container[link].distance_2D = dist_2D
            self._BsUeLink_container[link].distance_3D = dist_3D

            self._BsUeLink_container[link].LOS_AOD_GCS = aod_rad_gcs
            self._BsUeLink_container[link].LOS_ZOD_GCS = zod_rad_gcs
            self._BsUeLink_container[link].LOS_AOA_GCS = aoa_rad_gcs
            self._BsUeLink_container[link].LOS_ZOA_GCS = zoa_rad_gcs

    # Step 2
    def computeLOS(self):
        """
        Computes line-of-sight conditions for BSs and UEs.
        Results are stored in the class attribute 'los_Matrix'.
        """
        for link in self.BsUeLink_container.keys():
            self._BsUeLink_container[link].los = self.LineOfSight(link)

    def LineOfSight(self, bs_ue_link: BsUeLink) -> LOS:
        """
        Determine if a given BS and UE pair is in 'LOS' or 'NLOS'
        0: 'LOS'        1: 'NLOS'       2: 'IOF'

        :param bs_ue_link: BsUeLink object
        :return: the line-of-sight condition, i.e.: 'LOS' or 'NLOS'
        """

        dist_2D = self._BsUeLink_container[bs_ue_link].distance_2D
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

    def NetworkPathloss(self):
        """
        Computes the Pathloss and Line of Sight parameters for combinations of BSs and UEs
        """
        self.computeGeometry()
        for link in self.BsUeLink_container.keys():
            # Get link parameters
            dist_2D = self._BsUeLink_container[link].distance_2D
            dist_3D = self._BsUeLink_container[link].distance_3D
            los = self._BsUeLink_container[link].los

            # Get BS/UE information
            bs_height = self.BSs[self._BsUeLink_container[link].base_station_ID].height
            ue_height = self.UEs[self._BsUeLink_container[link].ue_ID].height
            ue_location = self.UEs[self._BsUeLink_container[link].ue_ID].location

            # Compute pathloss for each of the BS frequencies
            if COMPUTE_ALL_FREQUENCIES is True:
                a = ChannelBands()
                frequency_list = a.frequency_list
            else:
                frequency_list = self.BSs[self._BsUeLink_container[link].base_station_ID].getBsFrequencyList()

            for key_fc in frequency_list:
                fc = float(key_fc)
                pathloss, sf = self.Pathloss(dist_2D=dist_2D, dist_3D=dist_3D, los=los, bs_height=bs_height,
                                             ue_height=ue_height, ue_location=ue_location, fc=fc)
                self._BsUeLink_container[link].lspContainer[key_fc].Pathloss = pathloss
                self._BsUeLink_container[link].lspContainer[key_fc].SF = sf

    def Pathloss(self, dist_2D: float, dist_3D: float, los: LOS, bs_height: float, ue_height: float,
                 ue_location: Location, fc: float):
        """
        Computes the Pathloss between a BS sector and UE pair based on the Network scenario
        :param dist_2D: 2D distance
        :param dist_3D: 3D distance
        :param los: 2D distance
        :param bs_height: BS height
        :param ue_height: UE height
        :param ue_location: UE Location
        :param fc: Frequency (GHz)
        :return: Pathloss [dB]
        """

        pathloss = 0
        sigma_sf = 0

        # Basic Pathloss
        # If free_space flag is True, compute pathloss according to the Free Space model
        # If free_space flag is False, compute pathloss according to scenario
        if self.free_space:
            pathloss = 20 * np.log10(4 * np.pi * fc * 10 ** 9 / light_speed) + 20 * np.log10(dist_3D)
            shadow_fading = 0
        else:
            # Pathloss for Rural Macro scenario
            if self.scenario == 'RMa':
                if not (ue_height == 1.5):
                    raise "UE height is not the default value"  # Todo: need to check for correction formulas
                if not (bs_height == 35):
                    raise "BS height is not the default value"  # Todo: need to check for correction formulas

                # Break point distance (Table 7.4.1.1, Note 5)
                d_bp = 2 * np.pi * bs_height * ue_height * fc * 1_000_000_000 / light_speed

                if los == LOS.LOS:
                    # Compute PL_RMa-LOS
                    if (10 <= dist_2D) and (dist_2D <= d_bp):
                        pathloss = 20 * np.log10(40 * np.pi * dist_3D * fc / 3) \
                                   + min(0.03 * (self.avg_building_height ** 1.72), 10) * np.log10(dist_3D) \
                                   - min(0.044 * (self.avg_building_height ** 1.72), 14.77) \
                                   + 0.002 * np.log10(self.avg_building_height) * dist_3D
                        sigma_sf = 4
                    elif (d_bp <= dist_2D) and (dist_2D <= 10000):
                        pathloss = 20 * np.log10(40 * np.pi * d_bp * fc / 3) \
                                   + min(0.03 * (self.avg_building_height ** 1.72), 10) * np.log10(d_bp) \
                                   - min(0.044 * (self.avg_building_height ** 1.72), 14.77) \
                                   + 0.002 * np.log10(self.avg_building_height) * d_bp \
                                   + 40 * np.log10(dist_3D / d_bp)
                        sigma_sf = 6
                    else:
                        # TODO: remove
                        pathloss = np.inf
                        sigma_sf = 0

                if LOS.NLOS:
                    # Compute PL_RMa-LOS
                    if (10 <= dist_2D) and (dist_2D <= d_bp):
                        PL_RMa_LOS = 20 * np.log10(40 * np.pi * dist_3D * fc / 3) \
                                     + min(0.03 * (self.avg_building_height ** 1.72), 10) * np.log10(dist_3D) \
                                     - min(0.044 * (self.avg_building_height ** 1.72), 14.77) \
                                     + 0.002 * np.log10(self.avg_building_height) * dist_3D
                    elif (d_bp <= dist_2D) and (dist_2D <= 5000):
                        PL_RMa_LOS = 20 * np.log10(40 * np.pi * dist_3D * fc / 3) \
                                     + min(0.03 * (self.avg_building_height ** 1.72), 10) * np.log10(dist_3D) \
                                     - min(0.044 * (self.avg_building_height ** 1.72), 14.77) \
                                     + 0.002 * np.log10(self.avg_building_height) * dist_3D \
                                     + 40 * np.log10(dist_3D / d_bp)
                    else:
                        # TODO: remove
                        PL_RMa_LOS = np.inf
                        sigma_sf = 0

                    # Compute PL_RMa-NLOS
                    PL_RMa_NLOS = 161.04 - 7.1 * np.log10(self.average_street_width) \
                                  + 7.5 * np.log10(self.avg_building_height) \
                                  - (24.37 - 3.7 * ((self.avg_building_height / bs_height) ** 2)) * np.log10(bs_height) \
                                  + (43.42 - 3.1 * np.log10(bs_height)) * (np.log10(dist_3D) - 3) \
                                  + 20 * np.log10(fc) - (3.2 * (np.log10(11.75 * ue_height) ** 2) - 4.97)

                    pathloss = max(PL_RMa_LOS, PL_RMa_NLOS)
                    sigma_sf = 8

            # Pathloss for Urban Macro scenario
            if self.scenario == 'UMa':
                if not ((1.5 <= ue_height) and (ue_height <= 22.5)):
                    raise "UE height outside the pathloss formula's applicability range"
                if not (bs_height == 25):
                    raise "BS height is not the default value"  # Todo: need to check for correction formulas

                # Breakpoint Distance (Table 7.4.1.1, Note 1)
                C = 0
                if ue_height < 13:
                    C = 0
                elif (13 <= ue_height) and (ue_height <= 23):
                    if dist_2D <= 18:
                        g = 0
                    else:
                        g = (5 / 4) * ((dist_2D / 100) ** 3) * np.exp(-dist_2D / 150)
                    C = (((ue_height - 13) / 10) ** 1.5) * g

                if self.random_generator_PL.random() < 1 / (1 + C):
                    h_e = 1
                else:
                    h_e = self.random_generator_PL.choice(np.arange(12, ue_height - 1.5, 3))

                d_bp = 4 * (bs_height - h_e) * (ue_height - h_e) * fc * 1_000_000_000 / light_speed

                # Pathloss computation for LOS
                if los == LOS.LOS:
                    # Compute PL_UMa-LOS
                    if (10 <= dist_2D) and (dist_2D <= d_bp):
                        pathloss = 28.0 + 22 * np.log10(dist_3D) + 20 * np.log10(fc)
                        sigma_sf = 4.0
                    elif (d_bp <= dist_2D) and (dist_2D <= 5000):
                        pathloss = 28.0 + 40 * np.log10(dist_3D) + 20 * np.log10(fc) \
                                   - 9 * np.log10(d_bp ** 2 + (bs_height - ue_height) ** 2)
                        sigma_sf = 4.0
                    # elif dist_2D > 5000:  # Todo: Is this valid?
                    #     pathloss = np.inf
                    #     sigma_sf = 4.0
                    else:
                        # TODO: remove
                        pathloss = np.inf
                        sigma_sf = 0

                # Pathloss computation for NLOS
                if los == LOS.NLOS:
                    # Compute PL_UMa-LOS
                    if (10 <= dist_2D) and (dist_2D <= d_bp):
                        PL_UMa_LOS = 28.0 + 22 * np.log10(dist_3D) + 20 * np.log10(fc)
                    elif (d_bp <= dist_2D) and (dist_2D <= 5000):
                        PL_UMa_LOS = 28.0 + 40 * np.log10(dist_3D) + 20 * np.log10(fc) \
                                     - 9 * np.log10(d_bp ** 2 + (bs_height - ue_height) ** 2)
                    # elif dist_2D > 5000:  # Todo: Is this valid?
                    else:
                        # TODO: remove
                        PL_UMa_LOS = np.inf
                        sigma_sf = 0

                    # Compute PL_UMa-NLOS
                    PL_UMa_NLOS = 13.54 + 39.08 * np.log10(dist_3D) + 20 * np.log10(fc) - 0.6 * (ue_height - 1.5)

                    pathloss = max(PL_UMa_LOS, PL_UMa_NLOS)
                    sigma_sf = 6

                    # Optional
                    # pathloss = 32.4 + 20 * np.log10(fc) + 30 * np.log10(dist_3D)
                    # sigma_sf = 7.8

            # Pathloss for Urban Micro scenario
            if self.scenario == 'UMi':
                if not ((1.5 <= ue_height) and (ue_height <= 22.5)):
                    raise "UE height outside the pathloss formula's applicability range"
                if not (bs_height == 10):
                    raise "BS is not the default value"  # Todo: need to check for correction formulas

                # Breakpoint Distance (Table 7.4.1.1, Note 1)
                h_e = 1.0  # meter
                d_bp = 4 * (bs_height - h_e) * (ue_height - h_e) * fc * 1_000_000_000 / light_speed

                # Pathloss computation for LOS
                if los == LOS.LOS:
                    # Compute PL_UMi-LOS
                    if (10 <= dist_2D) and (dist_2D <= d_bp):
                        pathloss = 32.4 + 21 * np.log10(dist_3D) + 20 * np.log10(fc)
                        sigma_sf = 4.0
                    elif (d_bp <= dist_2D) and (dist_2D <= 5000):
                        pathloss = 32.4 + 40 * np.log10(dist_3D) + 20 * np.log10(fc) \
                                   - 9.5 * np.log10(d_bp ** 2 + (bs_height - ue_height) ** 2)
                        sigma_sf = 4.0
                    # elif dist_2D > 5000:  # Todo: Is this valid?
                    #     pathloss = np.inf
                    #     sigma_sf = 4.0
                    else:
                        # TODO: remove
                        pathloss = np.inf
                        sigma_sf = 0

                # Pathloss computation for NLOS
                if los == LOS.NLOS:

                    # Compute PL_UMi-LOS
                    if (10 <= dist_2D) and (dist_2D <= d_bp):
                        PL_UMi_LOS = 32.4 + 21 * np.log10(dist_3D) + 20 * np.log10(fc)
                    elif (d_bp <= dist_2D) and (dist_2D <= 5000):
                        PL_UMi_LOS = 32.4 + 40 * np.log10(dist_3D) + 20 * np.log10(fc) \
                                     - 9.5 * np.log10(d_bp ** 2 + (bs_height - ue_height) ** 2)
                    # elif dist_2D > 5000:  # Todo: Is this valid?
                    #     PL_UMi_LOS = np.inf
                    else:
                        PL_UMi_LOS = np.inf

                    # Compute PL_UMi-NLOS
                    PL_UMi_NLOS = 35.3 * np.log10(dist_3D) + 22.4 + 21.3 * np.log10(fc) - 0.3 * (ue_height - 1.5)
                    pathloss = max(PL_UMi_LOS, PL_UMi_NLOS)
                    sigma_sf = 7.82

                    # Optional
                    # pathloss = 32.4 + 20 * np.log10(fc) + 31.9 * np.log10(dist_3D)
                    # sigma_sf = 8.2

                    # try:
                    #     PL_UMi_NLOS = 35.3 * np.log10(dist_3D) \
                    #                   + 22.4 + 21.3 * np.log10(fc) \
                    #                   - 0.3 * (ue_height - 1.5)
                    #     pathloss = max(PL_UMi_LOS, PL_UMi_NLOS)
                    #     sigma_sf = 7.82
                    # except:
                    #     # Optional
                    #     pathloss = 32.4 + 20 * np.log10(fc) + 31.9 * np.log10(dist_3D)
                    #     sigma_sf = 8.2

            # Additional Pathloss terms for Indoor UEs
            if ue_location == Location.Indoor:
                # Todo: implement the penetration loss model - See TR 38.901 - 7.4.3
                PL_tw = 0
                PL_in = 0
                sigma_p_sq = 0
                pathloss = pathloss + PL_tw + PL_in + self.random_generator_PL.normal(scale=np.sqrt(sigma_p_sq))

            # Additional Pathloss terms for in Car UEs
            if ue_location == Location.Car:
                mu = 9.0  # 20 for metalized window
                sigma_p = 5.0
                pathloss = pathloss + self.random_generator_PL.normal(loc=mu, scale=sigma_p)

            # Compute shadow fading
            shadow_fading = self.random_generator_PL.lognormal(sigma=sigma_sf)

        return pathloss, shadow_fading

    def NetworkLargeScaleParameters(self):
        # if True:
        #     self._large_scale_parameter_correlation_method_three()
        # else:
        #     self._large_scale_parameter_correlation_method_two()
        self._large_scale_parameter_correlation_method_three()

    # def _large_scale_parameter_correlation_method_two(self):
    #     # Method 2: Create grid; 2D-filter normal iid points in the grid; use filtered values to compute LSP
    #
    #     ################################################################################################################
    #     # Get geometry and design the grid
    #     ################################################################################################################
    #     # Get UE positions:
    #     ue_position = np.zeros((len(self.UEs), 2))
    #     for ue_idx, ue in enumerate(self.UEs):
    #         ue_position[ue_idx][:] = [ue.pos_x, ue.pos_y]
    #         ue.LSP = [None] * len(self.BSs)
    #
    #     x_min, y_min = np.min(ue_position, axis=0)
    #     x_max, y_max = np.max(ue_position, axis=0)
    #     delta_d = .5
    #
    #     x = np.arange(x_min - delta_d, x_max + delta_d, delta_d)
    #     y = np.arange(y_min - delta_d, y_max + delta_d, delta_d)
    #     Nx = len(x)
    #     Ny = len(y)
    #
    #     # Filter grid
    #     D = 5  # Filter Length
    #     if D % 2 == 0:
    #         D = D + 1
    #     xf = np.linspace(-delta_d * (D - 1) / 2, delta_d * (D - 1) / 2, num=D)
    #     yf = np.linspace(-delta_d * (D - 1) / 2, delta_d * (D - 1) / 2, num=D)
    #     xv, yv = np.meshgrid(xf, yf)
    #     d = np.sqrt(xv ** 2 + yv ** 2)
    #     # Todo: for efficiency, this should select D according to delta_m (when delta_m is small eg. 3, D has to be
    #     #  larger eg. 11; when when delta_m is small eg. 50, D can be smaller eg. 5
    #
    #     for los in [LOS.LOS, LOS.NLOS]:
    #         C, Delta_m, LSP = self._generateLargeScaleParamsCorrelation(los=los)
    #         Q = sqrtm(C)
    #
    #         for bs in self.BSs:
    #             print(f'LOS:{los.value} - BS:{bs.ID}')
    #             normal_epsilon = np.random.normal(0, 1, (Nx, Ny))
    #             correlated_epsilon = np.zeros((len(LSP), len(self.UEs)))
    #
    #             for m, lsp in enumerate(LSP):
    #                 alpha = 1 / Delta_m[lsp]
    #                 filter_coeff = (alpha ** d) * np.exp(-alpha * d)
    #                 filter_coeff = filter_coeff / sum(sum(filter_coeff))
    #
    #                 filtered_grid = convolve2d(normal_epsilon, filter_coeff, mode='same', boundary='wrap',
    #                                            fillvalue=0)
    #
    #                 interp = RegularGridInterpolator((x, y), filtered_grid, method='linear')
    #                 correlated_epsilon[m, :] = interp(ue_position)
    #
    #             for ue_idx, ue in enumerate(self.UEs):
    #                 # if self.getLOS(self.los_Matrix[ue.ID][bs.ID]) == los:
    #                 if self.los_Matrix[ue.ID][bs.ID] == los:
    #                     s_tilde = np.dot(Q, correlated_epsilon[:, ue_idx])
    #                     correlated_TLSP = dict(zip(LSP, s_tilde))
    #                     ue.LSP[bs.ID] = self.generateLinkLPS(bs, ue, correlated_TLSP)
    #                     self.generateSmallScaleParams_link(bs=bs, ue=ue)

    def _large_scale_parameter_correlation_method_three(self):
        # Method 3: Create grid for each parameter; points are spaced according to the correlation distances
        ################################################################################################################
        # Get geometry and design the grid
        ################################################################################################################
        # Load Large Scale Params Correlation
        self._LargeScaleParamsCorrelation()
        parameter_grid = self.parameter_grid

        # Load sector-UE links
        # self.SectorUeLink_container

        for link in self.BsUeLink_container.keys():
            bs_id, ue_id = link
            los = self._BsUeLink_container[link].los
            dist_2D = self._BsUeLink_container[link].distance_2D
            ue_height = self.UEs[ue_id].height
            bs_height = self.BSs[bs_id].height
            ue_pos_x = self.UEs[ue_id].pos_x
            ue_pos_y = self.UEs[ue_id].pos_y

            # Get Correlated LSP
            correlated_epsilon = np.array([parameter_grid[(los.value, bs_id, lsp)].get_correlated_value_at(ue_pos_x,
                                                                                                           ue_pos_y)
                                           for lsp in self._LSP_list[los]])

            s_tilde = np.dot(self._Q[los], correlated_epsilon)
            correlated_TLSP = dict(zip(self._LSP_list[los], s_tilde))

            # Frequency Independent LSP
            link_LSP_dict = self._generate_frequency_independent_link_LSP(correlated_TLSP, los)

            self._BsUeLink_container[link].frequency_independent_lsp.SF = link_LSP_dict['SF']
            self._BsUeLink_container[link].frequency_independent_lsp.N = link_LSP_dict['N']
            self._BsUeLink_container[link].frequency_independent_lsp.M = link_LSP_dict['M']
            self._BsUeLink_container[link].frequency_independent_lsp.r_tau = link_LSP_dict['r_tau']
            self._BsUeLink_container[link].frequency_independent_lsp.c_ASA = link_LSP_dict['c_ASA']
            self._BsUeLink_container[link].frequency_independent_lsp.c_ASD = link_LSP_dict['c_ASD']
            self._BsUeLink_container[link].frequency_independent_lsp.c_ZSA = link_LSP_dict['c_ZSA']
            self._BsUeLink_container[link].frequency_independent_lsp.xi = link_LSP_dict['xi']
            self._BsUeLink_container[link].frequency_independent_lsp.mu_XPR = link_LSP_dict['mu_XPR']
            self._BsUeLink_container[link].frequency_independent_lsp.sigma_xpr = link_LSP_dict['sigma_xpr']
            self._BsUeLink_container[link].frequency_independent_lsp.K = link_LSP_dict['K']

            # Compute LSP for each of the BS frequencies
            if COMPUTE_ALL_FREQUENCIES is True:
                a = ChannelBands()
                frequency_list = a.frequency_list
            else:
                frequency_list = self.BSs[self._BsUeLink_container[link].base_station_ID].getBsFrequencyList()

            for key_fc in frequency_list:
                fc = float(key_fc)

                link_LSP_dict = self._generate_frequency_dependent_link_LSP(correlated_TLSP, fc, los, dist_2D,
                                                                            bs_height, ue_height)

                self._BsUeLink_container[link].lspContainer[key_fc].DS = link_LSP_dict['DS']
                self._BsUeLink_container[link].lspContainer[key_fc].ASA = link_LSP_dict['ASA']
                self._BsUeLink_container[link].lspContainer[key_fc].ASD = link_LSP_dict['ASD']
                self._BsUeLink_container[link].lspContainer[key_fc].ZSA = link_LSP_dict['ZSA']
                self._BsUeLink_container[link].lspContainer[key_fc].ZSD = link_LSP_dict['ZSD']
                self._BsUeLink_container[link].lspContainer[key_fc].c_DS = link_LSP_dict['c_DS']
                self._BsUeLink_container[link].lspContainer[key_fc].mu_offset_ZOD = link_LSP_dict['mu_offset_ZOD']
                self._BsUeLink_container[link].lspContainer[key_fc].mu_lg_ZSD = link_LSP_dict['mu_lg_ZSD']

    def _LargeScaleParamsCorrelation(self):
        for los in [LOS.LOS, LOS.NLOS]:
            self._C[los], self._delta_m[los], self._LSP_list[los] = self._generateLargeScaleParamsCorrelation(los=los)
            self._Q[los] = sqrtm(self._C[los])

    def _generateLargeScaleParamsCorrelation(self, los):
        """"
        Generates the Large Scale Parameter correlation matrices and the correlation distance.
        """
        # Initialize variables
        ASD_vs_DS = 0.0
        ASA_vs_DS = 0.0
        ASA_vs_SF = 0.0
        ASD_vs_SF = 0.0
        DS_vs_SF = 0.0
        ASD_vs_ASA = 0.0
        ASD_vs_K = 0.0
        ASA_vs_K = 0.0
        DS_vs_K = 0.0
        SF_vs_K = 0.0
        ZSD_vs_SF = 0.0
        ZSA_vs_SF = 0.0
        ZSD_vs_K = 0.0
        ZSA_vs_K = 0.0
        ZSD_vs_DS = 0.0
        ZSA_vs_DS = 0.0
        ZSD_vs_ASD = 0.0
        ZSA_vs_ASD = 0.0
        ZSD_vs_ASA = 0.0
        ZSA_vs_ASA = 0.0
        ZSD_vs_ZSA = 0.0
        corr_dist_h_plane_DS = 0.0
        corr_dist_h_plane_ASD = 0.0
        corr_dist_h_plane_ASA = 0.0
        corr_dist_h_plane_SF = 0.0
        corr_dist_h_plane_K = 0.0
        corr_dist_h_plane_ZSA = 0.0
        corr_dist_h_plane_ZSD = 0.0

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

    def _generate_frequency_dependent_link_LSP(self, correlated_TLSP, fc, los, dist_2D, bs_height, ue_height):
        # Large Scale Parameters (LSP) for different BS-UE links are uncorrelated, but the LSPs for links from co-sited
        # sectors to a UE are the same. In addition, LSPs for the links of UEs on different floors are uncorrelated.

        # Initialize parameters:
        mu_lg_DS = 0.0
        sigma_lg_DS = 0.0
        mu_lg_ASA = 0.0
        sigma_lg_ASA = 0.0
        mu_lg_ASD = 0.0
        sigma_lg_ASD = 0.0
        mu_lg_ZSA = 0.0
        sigma_lg_ZSA = 0.0
        mu_lg_ZSD = 0.0
        sigma_lg_ZSD = 0.0
        mu_offset_ZOD = 0.0
        c_DS = 0.0

        # Compute parameters according to scenario

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
                mu_lg_ZSD = np.maximum(-0.21, -14.8 * (dist_2D / 1000) + 0.01 * np.abs(ue_height - 1.5) + 0.83)
                sigma_lg_ZSD = 0.35
                mu_offset_ZOD = 0

                # Cluster DS [ns]
                c_DS = 5

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
                mu_lg_ZSD = np.maximum(-0.5,
                                       -3.1 * (dist_2D / 1000) + 0.01 * np.maximum(ue_height - bs_height, 0) + 0.2)
                sigma_lg_ZSD = 0.35
                mu_offset_ZOD = -10 ** (-1.5 * np.log10(np.maximum(10, dist_2D)) + 3.3)

                # Cluster DS [ns]
                c_DS = 11

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

                # Cluster DS [ns]
                c_DS = 11

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
                                               ue_height - 1.5) + 0.75)
                sigma_lg_ZSD = 0.4
                mu_offset_ZOD = 0

                # Cluster DS [ns]
                c_DS = max(0.25, 6.5622 - 3.4084 * np.log10(fc))

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
                                               ue_height - 1.5) + 0.9)
                sigma_lg_ZSD = 0.49
                mu_offset_ZOD = (7.66 * np.log10(fc) - 5.96) - 10 ** (
                        (0.208 * np.log10(fc) - 0.782) * np.log10(np.maximum(25, dist_2D))
                        - 0.13 * np.log10(fc) + 2.03 - 0.07 * (ue_height - 1.5))

                # Cluster DS [ns]
                c_DS = max(0.25, 6.5622 - 3.4084 * np.log10(fc))

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

                # Cluster DS [ns]
                c_DS = 11

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
                        ue_height - bs_height) + 0.22)
                sigma_lg_ZSD = 0.34
                mu_offset_ZOD = 0

                # Cluster DS [ns]
                c_DS = None

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
                        ue_height - bs_height) + 0.28)
                sigma_lg_ZSD = 0.30
                mu_offset_ZOD = np.arctan((35 - 3.5) / dist_2D) - np.arctan(
                    (35 - 1.5) / dist_2D)

                # Cluster DS [ns]
                c_DS = None

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
                        ue_height - bs_height) + 0.28)
                sigma_lg_ZSD = 0.30
                mu_offset_ZOD = np.arctan((35 - 3.5) / dist_2D) - np.arctan(
                    (35 - 1.5) / dist_2D)

                # Cluster DS [ns]
                c_DS = None

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

                # Cluster DS [ns]
                c_DS = None

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

                # Cluster DS [ns]
                c_DS = None

        # Use parameter to determine the LSP
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

        LSP_dict = {'DS': DS, 'ASA': ASA, 'ASD': ASD, 'ZSA': ZSA, 'ZSD': ZSD, 'c_DS': c_DS,
                    'mu_offset_ZOD': mu_offset_ZOD, 'mu_lg_ZSD': mu_lg_ZSD}

        return LSP_dict

    def _generate_frequency_independent_link_LSP(self, correlated_TLSP, los):
        # Large Scale Parameters (LSP) for different BS-UE links are uncorrelated, but the LSPs for links from co-sited
        # sectors to a UE are the same. In addition, LSPs for the links of UEs on different floors are uncorrelated.

        # Initialize parameters:
        mu_K = 0.0
        sigma_K = 0.0
        sigma_SF = 0.0
        r_tau = 0.0
        mu_XPR = 0.0
        sigma_xpr = 0.0
        N = 0
        M = 0
        c_ASD = 0.0
        c_ASA = 0.0
        c_ZSA = 0.0
        xi = 0.0

        # Compute parameters according to scenario

        if self.scenario == 'UMi':
            if los == LOS.LOS:
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

                # Cluster ASD [deg]
                c_ASD = 3

                # Cluster ASA [deg]
                c_ASA = 17

                # Cluster ZSA [deg]
                c_ZSA = 7

                # Per cluster shadowing std [dB]
                xi = 7

            if los == LOS.NLOS:
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

                # Cluster ASD [deg]
                c_ASD = 10

                # Cluster ASA [deg]
                c_ASA = 22

                # Cluster ZSA [deg]
                c_ZSA = 7

                # Per cluster shadowing std [dB]
                xi = 7

            if los == LOS.O2I:
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

                # Cluster ASD [deg]
                c_ASD = 5

                # Cluster ASA [deg]
                c_ASA = 8

                # Cluster ZSA [deg]
                c_ZSA = 3

                # Per cluster shadowing std [dB]
                xi = 4

        if self.scenario == 'UMa':
            if los == LOS.LOS:
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

                # Cluster ASD [deg]
                c_ASD = 5

                # Cluster ASA [deg]
                c_ASA = 11

                # Cluster ZSA [deg]
                c_ZSA = 7

                # Per cluster shadowing std [dB]
                xi = 3

            if los == LOS.NLOS:
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

                # Cluster ASD [deg]
                c_ASD = 5

                # Cluster ASA [deg]
                c_ASA = 11

                # Cluster ZSA [deg]
                c_ZSA = 7

                # Per cluster shadowing std [dB]
                xi = 3

            if los == LOS.O2I:
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

                # Cluster ASD [deg]
                c_ASD = 2

                # Cluster ASA [deg]
                c_ASA = 3

                # Cluster ZSA [deg]
                c_ZSA = 3

                # Per cluster shadowing std [dB]
                xi = 3

            if los == LOS.NLOS:
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

                # Cluster ASD [deg]
                c_ASD = 2

                # Cluster ASA [deg]
                c_ASA = 3

                # Cluster ZSA [deg]
                c_ZSA = 3

                # Per cluster shadowing std [dB]
                xi = 3

            if los == LOS.O2I:
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

                # Cluster ASD [deg]
                c_ASD = 2

                # Cluster ASA [deg]
                c_ASA = 3

                # Cluster ZSA [deg]
                c_ZSA = 3

                # Per cluster shadowing std [dB]
                xi = 3

        if self.scenario == 'InH':
            if los == LOS.LOS:
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

                # Cluster ASD [deg]
                c_ASD = 5

                # Cluster ASA [deg]
                c_ASA = 8

                # Cluster ZSA [deg]
                c_ZSA = 9

                # Per cluster shadowing std [dB]
                xi = 6

            if los == LOS.NLOS:
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

                # Cluster ASD [deg]
                c_ASD = 5

                # Cluster ASA [deg]
                c_ASA = 11

                # Cluster ZSA [deg]
                c_ZSA = 9

                # Per cluster shadowing std [dB]
                xi = 3

        # Use parameters and correlated_TLSP to determine the LSP

        # Generate SF
        SF = sigma_SF * correlated_TLSP['SF']

        if los == LOS.LOS:
            # Generate K
            K = mu_K + sigma_K * correlated_TLSP['K']
        else:
            K = 0

        LSP_dict = {'SF': SF, 'K': K, 'r_tau': r_tau, 'mu_XPR': mu_XPR, 'sigma_xpr': sigma_xpr, 'N': N, 'M': M,
                    'c_ASA': c_ASA, 'c_ASD': c_ASD, 'c_ZSA': c_ZSA, 'xi': xi}

        return LSP_dict

    def _generate_small_scale_random_draws(self):
        for link in self.BsUeLink_container.keys():
            # Number of clusters
            N = self._BsUeLink_container[link].frequency_independent_lsp.N
            M = self._BsUeLink_container[link].frequency_independent_lsp.M
            mu_XPR = self._BsUeLink_container[link].frequency_independent_lsp.mu_XPR
            sigma_xpr = self._BsUeLink_container[link].frequency_independent_lsp.sigma_xpr

            # Generate Random Draws for frequency independent RV
            self._BsUeLink_container[link].Xn = np.random.uniform(size=N)  # Used for Cluster Power Generation (7.5-1)
            self._BsUeLink_container[link].Xn_aoa = np.random.choice([-1, 1], size=N)
            self._BsUeLink_container[link].Xn_aod = np.random.choice([-1, 1], size=N)
            self._BsUeLink_container[link].Xn_zoa = np.random.choice([-1, 1], size=N)
            self._BsUeLink_container[link].Xn_zod = np.random.choice([-1, 1], size=N)

            self._BsUeLink_container[link].Yn_aoa_univar = np.random.normal(loc=0, scale=1, size=N)
            self._BsUeLink_container[link].Yn_aod_univar = np.random.normal(loc=0, scale=1, size=N)
            self._BsUeLink_container[link].Yn_zoa_univar = np.random.normal(loc=0, scale=1, size=N)
            self._BsUeLink_container[link].Yn_zod_univar = np.random.normal(loc=0, scale=1, size=N)

            self._BsUeLink_container[link].Xnm = np.random.normal(loc=mu_XPR, scale=sigma_xpr, size=(N, M))

    def NetworkSmallScaleParameters(self):
        for link in self.BsUeLink_container.keys():
            los = self._BsUeLink_container[link].los

            # Get link Frequency Independent Random Variables
            self._generate_small_scale_random_draws()
            Xn = self._BsUeLink_container[link].Xn
            Xn_aoa = self._BsUeLink_container[link].Xn_aoa
            Xn_aod = self._BsUeLink_container[link].Xn_aod
            Xn_zoa = self._BsUeLink_container[link].Xn_zoa
            Xn_zod = self._BsUeLink_container[link].Xn_zod
            Xnm = self._BsUeLink_container[link].Xnm

            # Get link LOS angles
            phi_LOS_AOA = np.rad2deg(self._BsUeLink_container[link].LOS_AOA_GCS)
            phi_LOS_AOD = np.rad2deg(self._BsUeLink_container[link].LOS_AOD_GCS)
            theta_LOS_ZOA = np.rad2deg(self._BsUeLink_container[link].LOS_ZOA_GCS)
            theta_LOS_ZOD = np.rad2deg(self._BsUeLink_container[link].LOS_ZOD_GCS)

            # Get link Frequency Independent LSP
            N = self._BsUeLink_container[link].frequency_independent_lsp.N
            M = self._BsUeLink_container[link].frequency_independent_lsp.M
            r_tau = self._BsUeLink_container[link].frequency_independent_lsp.r_tau
            K = self._BsUeLink_container[link].frequency_independent_lsp.K
            xi = self._BsUeLink_container[link].frequency_independent_lsp.xi
            c_ASA = self._BsUeLink_container[link].frequency_independent_lsp.c_ASA
            c_ASD = self._BsUeLink_container[link].frequency_independent_lsp.c_ASD
            c_ZSA = self._BsUeLink_container[link].frequency_independent_lsp.c_ZSA

            # Compute LSP for each of the BS frequencies
            if COMPUTE_ALL_FREQUENCIES is True:
                a = ChannelBands()
                frequency_list = a.frequency_list
            else:
                frequency_list = self.BSs[self._BsUeLink_container[link].base_station_ID].getBsFrequencyList()

            # frequency_list = self.BSs[self._BsUeLink_container[link].base_station_ID].getBsFrequencyList()
            for key_fc in frequency_list:
                ASA = self._BsUeLink_container[link].lspContainer[key_fc].ASA
                ASD = self._BsUeLink_container[link].lspContainer[key_fc].ASD
                ZSA = self._BsUeLink_container[link].lspContainer[key_fc].ZSA
                ZSD = self._BsUeLink_container[link].lspContainer[key_fc].ZSD
                DS = self._BsUeLink_container[link].lspContainer[key_fc].DS
                mu_offset_ZOD = self._BsUeLink_container[link].lspContainer[key_fc].mu_offset_ZOD
                mu_lg_ZSD = self._BsUeLink_container[link].lspContainer[key_fc].mu_lg_ZSD

                Yn_aoa = (ASA / 7) * self._BsUeLink_container[link].Yn_aoa_univar
                Yn_aod = (ASD / 7) * self._BsUeLink_container[link].Yn_aod_univar
                Yn_zoa = (ZSA / 7) * self._BsUeLink_container[link].Yn_zoa_univar
                Yn_zod = (ZSD / 7) * self._BsUeLink_container[link].Yn_zod_univar

                # Call Step 5
                cluster_delay, cluster_delay_LOS = self._cluster_delay(los=los, r_tau=r_tau, DS=DS, K=K, Xn=Xn)

                # Call Step 6
                P_n, cluster_delay = self._cluster_power(cluster_delay=cluster_delay, los=los, r_tau=r_tau, DS=DS,
                                                         K=K, xi=xi)
                updated_N_cluster = len(P_n)

                # Call Step 7: Azimuth of Arrival and Departure
                C_phi_NLOS = C_phi_NLOS_dict[N]
                if los == LOS.LOS:
                    C_phi = C_phi_NLOS * (1.1035 - 0.028 * K - 0.002 * (K ** 2) + 0.0001 * (K ** 3))
                else:
                    C_phi = C_phi_NLOS

                phi_n_m_AOA = self._cluster_aoa(los=los, C_phi=C_phi, updated_N_cluster=updated_N_cluster,
                                                Xn_aoa=Xn_aoa, Yn_aoa=Yn_aoa, ASA=ASA, P_n=P_n, phi_LOS_AOA=phi_LOS_AOA,
                                                M=M, c_ASA=c_ASA)

                phi_n_m_AOD = self._cluster_aod(los=los, C_phi=C_phi, updated_N_cluster=updated_N_cluster,
                                                Xn_aod=Xn_aod, Yn_aod=Yn_aod, ASD=ASD, P_n=P_n, phi_LOS_AOD=phi_LOS_AOD,
                                                M=M, c_ASD=c_ASD)

                # Call Step 7: Zenith of Arrival and Departure
                C_theta_NLOS = C_theta_NLOS_dict[N]
                if los == LOS.LOS:
                    C_theta = C_theta_NLOS * (1.3086 + 0.0339 * K - 0.0077 * (K ** 2) + 0.0002 * (K ** 3))
                else:
                    C_theta = C_theta_NLOS

                theta_n_m_ZOA = self._cluster_zoa(los, C_theta, updated_N_cluster, Xn_zoa, Yn_zoa, ZSA, P_n,
                                                  theta_LOS_ZOA, M, c_ZSA)

                theta_n_m_ZOD = self._cluster_zod(los, C_theta, updated_N_cluster, Xn_zod, Yn_zod, ZSD, P_n,
                                                  theta_LOS_ZOD, M, mu_offset_ZOD, mu_lg_ZSD)

                # Call Step 8: Coupling of rays within a cluster for both azimuth and elevation
                ray_mapping_AoD_AoA, ray_mapping_ZoD_ZoA, ray_mapping_AoD_ZoA = self._ray_coupling(updated_N_cluster, M)

                # Reorder angles according to ray coupling:
                for n in range(updated_N_cluster):
                    phi_n_m_AOD[n] = phi_n_m_AOD[n, ray_mapping_AoD_AoA[n]]
                    theta_n_m_ZOA[n] = theta_n_m_ZOA[n, ray_mapping_AoD_ZoA[n]]
                    theta_n_m_ZOD[n] = theta_n_m_ZOD[n, ray_mapping_ZoD_ZoA[n, ray_mapping_AoD_ZoA[n]]]

                # Call Step 9: Generate the cross polarization power ratios
                Knm = self._cross_polarization_power_ratio(Xnm, updated_N_cluster)

                # Stores all information back to link LSP
                self._BsUeLink_container[link].lspContainer[key_fc].cluster_delay = cluster_delay
                self._BsUeLink_container[link].lspContainer[key_fc].cluster_delay_LOS = cluster_delay_LOS
                self._BsUeLink_container[link].lspContainer[key_fc].P_n = P_n
                self._BsUeLink_container[link].lspContainer[key_fc].updated_N_cluster = updated_N_cluster
                self._BsUeLink_container[link].lspContainer[key_fc].phi_n_m_AOA = phi_n_m_AOA
                self._BsUeLink_container[link].lspContainer[key_fc].phi_n_m_AOD = phi_n_m_AOD
                self._BsUeLink_container[link].lspContainer[key_fc].theta_n_m_ZOA = theta_n_m_ZOA
                self._BsUeLink_container[link].lspContainer[key_fc].theta_n_m_ZOD = theta_n_m_ZOD
                self._BsUeLink_container[link].lspContainer[key_fc].Knm = Knm

    def NetworkChannelGeneration(self):
        for sector_ue_key in self.SectorUeLink_container.keys():
            sector_id, ue_id = sector_ue_key
            bs_id = self._SectorUeLink_container[sector_ue_key].base_station_ID
            bs_ue_key = (bs_id, ue_id)
            bs_sector_id = self._SectorUeLink_container[sector_ue_key].bs_sector_id
            sector = self.BSs[bs_id].sectors[bs_sector_id]
            key_fc = sector.frequency_ghz

            # Retrieve LSPs
            M = self._BsUeLink_container[bs_ue_key].frequency_independent_lsp.M
            updated_N_cluster = self._BsUeLink_container[bs_ue_key].lspContainer[key_fc].updated_N_cluster

            # Call Step 10: Coefficient generation - Draw initial random phases
            self._SectorUeLink_container[sector_ue_key].Phi_nm = self._draw_random_phases(updated_N_cluster, M)

            # Step 11: Coefficient generation
            # Generate channel coefficients for each cluster n and each Rx and Tx element pair u, s.
            self._generate_channel_coefficients(sector_ue_key)

    # Step 5: Generate Cluster Delay
    @staticmethod
    def _cluster_delay(los, r_tau, DS, K, Xn):
        ################################################################################################################
        # Step 5: Generate cluster delays Tau_n:
        # Xn = np.random.uniform(size=link.LSP['N'])
        cluster_delay = - r_tau * DS * np.log(Xn)
        cluster_delay = np.sort(cluster_delay - min(cluster_delay))

        if los == LOS.LOS:
            C_tau = 0.7705 - 0.0433 * K + 0.0002 * (K ** 2) + 0.000017 * (
                    K ** 3)
        else:
            C_tau = 1

        cluster_delay_LOS = cluster_delay / C_tau

        return cluster_delay, cluster_delay_LOS

    # Step 6: Generate Cluster Power
    @staticmethod
    def _cluster_power(cluster_delay, los, r_tau, DS, K, xi):
        P_n_notch = np.exp(-cluster_delay * ((r_tau - 1) / (r_tau * DS)))
        Zn = (10 ** (- np.random.normal(loc=0.0, scale=xi) / 10))
        P_n_notch = P_n_notch * Zn

        P_n = P_n_notch / sum(P_n_notch)

        if los == LOS.LOS:
            K_R_linear = 10 ** (K / 10)
            P_n = (1 / (K_R_linear + 1)) * P_n
            P_n[0] = P_n[0] + (K_R_linear / (K_R_linear + 1))

        # Discard clusters with power less than -25 dB compared to the maximum cluster power
        P_n_dB = 10 * np.log10(P_n)
        P_n = P_n[P_n_dB >= max(P_n_dB) - 25]

        cluster_delay = cluster_delay[P_n_dB >= max(P_n_dB) - 25]
        return P_n, cluster_delay

    # Step 7-a: Generate AoA
    @staticmethod
    def _cluster_aoa(los, C_phi, updated_N_cluster, Xn_aoa, Yn_aoa, ASA, P_n, phi_LOS_AOA, M, c_ASA):
        Xn_aoa = Xn_aoa[0:updated_N_cluster]
        Yn_aoa = Yn_aoa[0:updated_N_cluster]
        phi_notch_n_AOA = 2 * (ASA / 1.4) * np.sqrt(-np.log(P_n / max(P_n))) / C_phi

        if not (los == LOS.LOS):
            phi_n_AOA = Xn_aoa * phi_notch_n_AOA + Yn_aoa + phi_LOS_AOA
        else:
            phi_n_AOA = (Xn_aoa * phi_notch_n_AOA + Yn_aoa) - (Xn_aoa[0] * phi_notch_n_AOA[0] + Yn_aoa[0] - phi_LOS_AOA)

        phi_n_m_AOA = np.zeros((updated_N_cluster, M))
        for n in range(updated_N_cluster):
            phi_n_m_AOA[n] = phi_n_AOA[n] + c_ASA * alpha_m[:M]
        return phi_n_m_AOA

    # Step 7-b: Generate AoD
    @staticmethod
    def _cluster_aod(los, C_phi, updated_N_cluster, Xn_aod, Yn_aod, ASD, P_n, phi_LOS_AOD, M, c_ASD):
        Xn_aod = Xn_aod[0:updated_N_cluster]
        Yn_aod = Yn_aod[0:updated_N_cluster]
        phi_notch_n_AOD = 2 * (ASD / 1.4) * np.sqrt(-np.log(P_n / max(P_n))) / C_phi

        if not (los == LOS.LOS):
            phi_n_AOD = Xn_aod * phi_notch_n_AOD + Yn_aod + phi_LOS_AOD
        else:
            phi_n_AOD = (Xn_aod * phi_notch_n_AOD + Yn_aod) - (Xn_aod[0] * phi_notch_n_AOD[0] + Yn_aod[0] - phi_LOS_AOD)

        phi_n_m_AOD = np.zeros((updated_N_cluster, M))
        for n in range(updated_N_cluster):
            phi_n_m_AOD[n] = phi_n_AOD[n] + c_ASD * alpha_m[:M]

        return phi_n_m_AOD

    # Step 7-b: Generate ZoA
    @staticmethod
    def _cluster_zoa(los, C_theta, updated_N_cluster, Xn_zoa, Yn_zoa, ZSA, P_n, theta_LOS_ZOA, M, c_ZSA):
        Xn_zoa = Xn_zoa[0:updated_N_cluster]
        Yn_zoa = Yn_zoa[0:updated_N_cluster]
        theta_notch_n_ZOA = - ZSA * np.log(P_n / max(P_n)) / C_theta

        if los == LOS.O2I:
            theta_LOS_ZOA = 90

        if not (los == LOS.LOS):
            theta_n_ZOA = Xn_zoa * theta_notch_n_ZOA + Yn_zoa + theta_LOS_ZOA
        else:
            theta_n_ZOA = (Xn_zoa * theta_notch_n_ZOA + Yn_zoa) - (Xn_zoa[0] * theta_notch_n_ZOA[0] + Yn_zoa[0]
                                                                   - theta_LOS_ZOA)

        theta_n_m_ZOA = np.zeros((updated_N_cluster, M))
        for n in range(updated_N_cluster):
            temp = theta_n_ZOA[n] + c_ZSA * alpha_m[:M]
            temp[(temp >= 180) & (temp <= 360)] = 360 - temp[(temp >= 180) & (temp <= 360)]
            theta_n_m_ZOA[n] = temp

        return theta_n_m_ZOA

    # Step 7-b: Generate ZoD
    @staticmethod
    def _cluster_zod(los, C_theta, updated_N_cluster, Xn_zod, Yn_zod, ZSD, P_n, theta_LOS_ZOD, M, mu_offset_ZOD,
                     mu_lg_ZSD):
        Xn_zod = Xn_zod[0:updated_N_cluster]
        Yn_zod = Yn_zod[0:updated_N_cluster]
        theta_notch_n_ZOD = - ZSD * np.log(P_n / max(P_n)) / C_theta

        if not (los == LOS.LOS):
            theta_n_ZOD = Xn_zod * theta_notch_n_ZOD + Yn_zod + theta_LOS_ZOD + mu_offset_ZOD
        else:
            theta_n_ZOD = (Xn_zod * theta_notch_n_ZOD + Yn_zod) - (Xn_zod[0] * theta_notch_n_ZOD[0] + Yn_zod[0]
                                                                   - theta_LOS_ZOD)

        theta_n_m_ZOD = np.zeros((updated_N_cluster, M))
        for n in range(updated_N_cluster):
            temp = theta_n_ZOD[n] + (3 / 8) * (10 ** mu_lg_ZSD) * alpha_m[:M]
            temp[(temp >= 180) & (temp <= 360)] = 360 - temp[(temp >= 180) & (temp <= 360)]
            theta_n_m_ZOD[n] = temp

        return theta_n_m_ZOD

    # Step 8: Coupling of rays
    @staticmethod
    def _ray_coupling(updated_N_cluster, M):

        ray_mapping_AoD_AoA = np.zeros((updated_N_cluster, M), dtype=int)
        ray_mapping_ZoD_ZoA = np.zeros((updated_N_cluster, M), dtype=int)
        ray_mapping_AoD_ZoA = np.zeros((updated_N_cluster, M), dtype=int)
        for n in range(updated_N_cluster):
            if n in [0, 1]:
                for sub_cluster_idx in range(3):
                    sub_cluster_mapping_it = [x - 1 for x in subcluster_mapping[sub_cluster_idx]]
                    ray_mapping_AoD_AoA[n, sub_cluster_mapping_it] = \
                        np.random.permutation(sub_cluster_mapping_it).astype(int)
                    ray_mapping_ZoD_ZoA[n, sub_cluster_mapping_it] = \
                        np.random.permutation(sub_cluster_mapping_it).astype(int)
                    ray_mapping_AoD_ZoA[n, sub_cluster_mapping_it] = \
                        np.random.permutation(sub_cluster_mapping_it).astype(int)
            else:
                ray_mapping_AoD_AoA[n] = np.random.permutation(M).astype(int)
                ray_mapping_ZoD_ZoA[n] = np.random.permutation(M).astype(int)
                ray_mapping_AoD_ZoA[n] = np.random.permutation(M).astype(int)

        return ray_mapping_AoD_AoA, ray_mapping_ZoD_ZoA, ray_mapping_AoD_ZoA

    # Step 9: Generate the cross polarization power ratios
    @staticmethod
    def _cross_polarization_power_ratio(Xnm, updated_N_cluster):
        Xnm = Xnm[:updated_N_cluster, :]
        Knm = np.power(10, Xnm / 10)
        return Knm

    # Step 10: Coefficient generation - Draw initial random phases
    @staticmethod
    def _draw_random_phases(updated_N_cluster, M):
        Phi_nm = np.random.uniform(-np.pi, np.pi, size=(updated_N_cluster, M, 2, 2))
        # Phi theta-theta: Phi_nm[n,m,0,0]  -  Phi theta-phi: Phi_nm[n,m,0,1]
        # Phi phi-theta: Phi_nm[n,m,1,0]    -  Phi phi-phi: Phi_nm[n,m,1,1]

        return Phi_nm

    def _generate_channel_coefficients(self, sector_ue_link_id):

        sector_id, ue_id = sector_ue_link_id
        bs_id = self._SectorUeLink_container[sector_ue_link_id].base_station_ID
        bs_ue_key = (bs_id, ue_id)
        bs_sector_id = self._SectorUeLink_container[sector_ue_link_id].bs_sector_id
        sector = self.BSs[bs_id].sectors[bs_sector_id]
        key_fc = sector.frequency_ghz
        los = self._BsUeLink_container[bs_ue_key].los
        dist_3d = self._BsUeLink_container[bs_ue_key].distance_3D

        # Retrieve antenna panels:
        ue_antenna_panel = self.UEs[ue_id].antenna_panels[0]
        sector_antenna_panel = sector.antenna_panels[0]

        lambda_0 = light_speed / (sector.frequency_ghz * 1E9)

        # Retrieve LSPs
        M = self._BsUeLink_container[bs_ue_key].frequency_independent_lsp.M
        K = self._BsUeLink_container[bs_ue_key].frequency_independent_lsp.K
        updated_N_cluster = self._BsUeLink_container[bs_ue_key].lspContainer[key_fc].updated_N_cluster

        c_DS = self._BsUeLink_container[bs_ue_key].lspContainer[key_fc].c_DS
        cluster_delay = self._BsUeLink_container[bs_ue_key].lspContainer[key_fc].cluster_delay
        # Todo: check if should used cluster_delay_LOS
        # cluster_delay_LOS = self._BsUeLink_container[bs_ue_key].lspContainer[key_fc].cluster_delay_LOS
        P_n = self._BsUeLink_container[bs_ue_key].lspContainer[key_fc].P_n
        phi_n_m_AOA = self._BsUeLink_container[bs_ue_key].lspContainer[key_fc].phi_n_m_AOA
        phi_n_m_AOD = self._BsUeLink_container[bs_ue_key].lspContainer[key_fc].phi_n_m_AOD
        theta_n_m_ZOA = self._BsUeLink_container[bs_ue_key].lspContainer[key_fc].theta_n_m_ZOA
        theta_n_m_ZOD = self._BsUeLink_container[bs_ue_key].lspContainer[key_fc].theta_n_m_ZOD
        Knm = self._BsUeLink_container[bs_ue_key].lspContainer[key_fc].Knm

        # Get link LOS angles
        phi_LOS_AOA = np.rad2deg(self._BsUeLink_container[bs_ue_key].LOS_AOA_GCS)
        phi_LOS_AOD = np.rad2deg(self._BsUeLink_container[bs_ue_key].LOS_AOD_GCS)
        theta_LOS_ZOA = np.rad2deg(self._BsUeLink_container[bs_ue_key].LOS_ZOA_GCS)
        theta_LOS_ZOD = np.rad2deg(self._BsUeLink_container[bs_ue_key].LOS_ZOA_GCS)

        # Get UE mobility parameters
        ue_velocity_ms = self.UEs[ue_id].velocity_ms
        ue_mobility_direction_phi = self.UEs[ue_id].mobility_direction_phi
        ue_mobility_direction_theta = self.UEs[ue_id].mobility_direction_theta

        # Sector-specific
        Phi_nm = self._SectorUeLink_container[sector_ue_link_id].Phi_nm

        # TODO: Add support for multiple polarization
        # Get UE Polarization information
        # PolUE = self.UEs[ue_id].antenna_panels[0].polarization
        # PolSlantUE = self.UEs[ue_id].antenna_panels[0].polarization_slant

        # Get BS Polarization information
        # PolBS = sector.antenna_panels[0].polarization
        # PolSlantBS = sector.antenna_panels[0].polarization_slant

        # Initialize channel dimensions
        Nr = self.UEs[ue_id].number_of_antennas
        Nt = sector.number_of_antennas
        n_U = Nr
        n_S = Nt

        # Initialize array location vector
        d_rx = ue_antenna_panel.array_location_vector
        d_tx = sector_antenna_panel.array_location_vector

        # Initialize r_rx_vel, r_tx_mn and r_rx_mn
        r_rx_vel = ue_velocity_ms * np.array(
            [np.sin(np.deg2rad(ue_mobility_direction_theta)) * np.cos(np.deg2rad(ue_mobility_direction_phi)),
             np.sin(np.deg2rad(ue_mobility_direction_theta)) * np.sin(np.deg2rad(ue_mobility_direction_phi)),
             np.cos(np.deg2rad(ue_mobility_direction_theta))]).T

        r_rx_mn = np.array([np.sin(np.deg2rad(theta_n_m_ZOA)) * np.cos(np.deg2rad(phi_n_m_AOA)),
                            np.sin(np.deg2rad(theta_n_m_ZOA)) * np.sin(np.deg2rad(phi_n_m_AOA)),
                            np.cos(np.deg2rad(theta_n_m_ZOA))]).T
        r_tx_mn = np.array([np.sin(np.deg2rad(theta_n_m_ZOD)) * np.cos(np.deg2rad(phi_n_m_AOD)),
                            np.sin(np.deg2rad(theta_n_m_ZOD)) * np.sin(np.deg2rad(phi_n_m_AOD)),
                            np.cos(np.deg2rad(theta_n_m_ZOD))]).T

        # Initialize exp_vel, exp_rx and exp_tx
        exp_vel = np.exp(1j * 2 * np.pi * np.dot(r_rx_mn, r_rx_vel) / lambda_0)
        exp_rx = np.exp(1j * 2 * np.pi * np.dot(r_rx_mn, d_rx.T) / lambda_0)
        exp_tx = np.exp(1j * 2 * np.pi * np.dot(r_tx_mn, d_tx.T) / lambda_0)

        # InitializePolM
        G = np.ones((updated_N_cluster, M, 2, 2))
        G[:, :, 0, 1] = np.sqrt(1 / Knm)
        G[:, :, 1, 0] = np.sqrt(1 / Knm)
        PolM = np.multiply(G, np.exp(1j * Phi_nm))

        # ##############################################################################################################
        # LOS Channel
        # ##############################################################################################################
        # Initialize LOS channel coefficients
        h_los_u_s = np.zeros((n_U, n_S), dtype=complex)

        # Todo: check if this shouldn't be the LOS delay
        los_delay = cluster_delay[0]

        # Compute LOS/NLOS gain using Ricean Factor
        gain_LOS = 0.0
        K_R_linear = 10.0 ** (K / 10.0)
        if los == LOS.LOS:
            gain_LOS = np.sqrt(K_R_linear / (K_R_linear + 1))
            gain_NLOS = np.sqrt(1 / (K_R_linear + 1))
        else:
            gain_NLOS = 1.0

        if los == LOS.LOS:
            PolM_LOS = np.array([[1, 0], [0, -1]])

            r_rx_mn_los = np.array([np.sin(np.deg2rad(theta_LOS_ZOA)) * np.cos(np.deg2rad(phi_LOS_AOA)),
                                    np.sin(np.deg2rad(theta_LOS_ZOA)) * np.sin(np.deg2rad(phi_LOS_AOA)),
                                    np.cos(np.deg2rad(theta_LOS_ZOA))]).T
            r_tx_mn_los = np.array([np.sin(np.deg2rad(theta_LOS_ZOD)) * np.cos(np.deg2rad(phi_LOS_AOD)),
                                    np.sin(np.deg2rad(theta_LOS_ZOD)) * np.sin(np.deg2rad(phi_LOS_AOD)),
                                    np.cos(np.deg2rad(theta_LOS_ZOD))]).T

            Frx_u = ue_antenna_panel.vector_field_transformation_from_gcs(np.deg2rad(theta_LOS_ZOA),
                                                                          np.deg2rad(phi_LOS_AOA))
            Ftx_s = sector_antenna_panel.vector_field_transformation_from_gcs(np.deg2rad(theta_LOS_ZOD),
                                                                              np.deg2rad(phi_LOS_AOD))

            exp_vel_los = np.exp(1j * 2 * np.pi * np.dot(r_rx_mn_los, r_rx_vel) / lambda_0)
            exp_rx_los = np.exp(1j * 2 * np.pi * np.dot(d_rx, r_rx_mn_los) / lambda_0)
            exp_tx_los = np.exp(1j * 2 * np.pi * np.dot(d_tx, r_tx_mn_los) / lambda_0)

            exp_dist = np.exp(- 1j * 2 * np.pi * dist_3d / lambda_0)

            for u in range(n_U):
                for s in range(n_S):
                    h_los_u_s[u, s] = gain_LOS * np.dot(Frx_u.T, np.dot(PolM_LOS, Ftx_s)) \
                                      * exp_dist * exp_rx_los[u] * exp_tx_los[s] * exp_vel_los

        # ##############################################################################################################
        # NLOS Channel (Strongest Paths)
        # ##############################################################################################################
        # Initialize NLOS channel coefficients
        if updated_N_cluster < 2:
            n_plus_subcluster = updated_N_cluster * 3
        else:
            n_plus_subcluster = 2 * 3 + (updated_N_cluster - 2)

        h_nlos_u_s_n_m = np.zeros((n_U, n_S, n_plus_subcluster, M), dtype=complex)
        nlos_delay = np.zeros(n_plus_subcluster)

        cluster_plus_subcluster_idx = -1

        delay_offset = np.array([0, 1.28 * c_DS, 2.56 * c_DS])
        for n in [nn for nn in range(2) if nn <= (updated_N_cluster - 1)]:
            path_gain = np.sqrt(P_n[n] / M) * gain_NLOS

            for sub_cluster in range(3):
                cluster_plus_subcluster_idx += 1

                # Todo: check if this shouldn't be the LOS delay
                nlos_delay[cluster_plus_subcluster_idx] = cluster_delay[n] + 1E-9 * delay_offset[sub_cluster]

                for m in [x - 1 for x in subcluster_mapping[sub_cluster]]:
                    # TODO: Polarization Slant Angles should be included in the vector field
                    Frx_u = ue_antenna_panel.vector_field_transformation_from_gcs(np.deg2rad(theta_n_m_ZOA[n, m]),
                                                                                  np.deg2rad(phi_n_m_AOA[n, m]))
                    Ftx_s = sector_antenna_panel.vector_field_transformation_from_gcs(np.deg2rad(theta_n_m_ZOD[n, m]),
                                                                                      np.deg2rad(phi_n_m_AOD[n, m]))

                    for u in range(n_U):
                        for s in range(n_S):
                            h = path_gain * np.dot(Frx_u.T, np.dot(PolM[n, m], Ftx_s)) * exp_rx[m, n, u] * \
                                 exp_tx[m, n, s] * exp_vel[m, n]

                            h_nlos_u_s_n_m[u, s, cluster_plus_subcluster_idx, m] = h

        # ##############################################################################################################
        # NLOS Channel (Weakest Paths)
        # ##############################################################################################################
        for n in range(2, updated_N_cluster):
            path_gain = np.sqrt(P_n[n] / M) * gain_NLOS

            cluster_plus_subcluster_idx += 1

            # Todo: check if this shouldn't be the LOS delay
            nlos_delay[cluster_plus_subcluster_idx] = cluster_delay[n]

            for m in range(M):
                # TODO: Polarization Slant Angles should be included in the vector field
                Frx_u = ue_antenna_panel.vector_field_transformation_from_gcs(np.deg2rad(theta_n_m_ZOA[n, m]),
                                                                              np.deg2rad(phi_n_m_AOA[n, m]))
                Ftx_s = sector_antenna_panel.vector_field_transformation_from_gcs(np.deg2rad(theta_n_m_ZOD[n, m]),
                                                                                  np.deg2rad(phi_n_m_AOD[n, m]))

                for u in range(n_U):
                    for s in range(n_S):
                        h = path_gain * np.dot(Frx_u.T, np.dot(PolM[n, m], Ftx_s)) * exp_rx[m, n, u] * exp_tx[m, n, s] \
                            * exp_vel[m, n]
                        h_nlos_u_s_n_m[u, s, cluster_plus_subcluster_idx, m] = h

        # TODO: Will need to change the computation of field vector to include polarization and this should double
        #       the number of elements in the channel

        self._SectorUeLink_container[sector_ue_link_id].channel.h_los_u_s = h_los_u_s
        self._SectorUeLink_container[sector_ue_link_id].channel.h_nlos_u_s_n_m = h_nlos_u_s_n_m
        self._SectorUeLink_container[sector_ue_link_id].channel.los_delay = los_delay
        self._SectorUeLink_container[sector_ue_link_id].channel.nlos_delay = nlos_delay

    def computeRSRP(self):
        """
        Compute the RSRP
        :return: update Network attribute RsrpMatrix
        """
        # self.SectorUeLink_container
        #
        # for link in self.BsUeLink_container.keys():
        #     print(link)
        #     ue_id = self._BsUeLink_container[link].ue_ID
        #     for sec in self.BSs[self._BsUeLink_container[link].base_station_ID].sectors:
        #         sector_id = sec.ID
        #         SectorUeLink_key = (sector_id, ue_id)
        #
        #         pathloss = self._BsUeLink_container[link].lspContainer[sec.frequency_ghz].Pathloss
        #
        #         self._SectorUeLink_container[SectorUeLink_key].RSRP = sec.tx_power_dBm - 10 * np.log10(
        #             12 * sec.number_of_PRBs) - pathloss
        for sector_ue_link_id in self.SectorUeLink_container.keys():
            self._computeRSRP_36873(sector_ue_link_id)

    def _computeRSRP_36873(self, sector_ue_key):
        """
        Compute the RSRP
        :return: update Network attribute RsrpMatrix
        """

        sector_id, ue_id = sector_ue_key
        bs_id = self._SectorUeLink_container[sector_ue_key].base_station_ID
        bs_ue_key = (bs_id, ue_id)
        bs_sector_id = self._SectorUeLink_container[sector_ue_key].bs_sector_id
        sector = self.BSs[bs_id].sectors[bs_sector_id]
        key_fc = sector.frequency_ghz
        los = self._BsUeLink_container[bs_ue_key].los

        U = self.UEs[ue_id].number_of_antennas

        pathloss = self._BsUeLink_container[bs_ue_key].lspContainer[key_fc].Pathloss
        shadow_fading = self._BsUeLink_container[bs_ue_key].lspContainer[key_fc].SF

        tx_power_per_RE = sector.tx_power_dBm - 10 * np.log10(12 * sector.number_of_PRBs)
        # tx_power_per_RE = sector.tx_power_dBm

        # TODO: Remember that the channel coefficients should be weighted by the beamforming
        #         (antenna virtualization)
        Nu, Ns, N, M = self._SectorUeLink_container[sector_ue_key].channel.h_nlos_u_s_n_m.shape

        # w = np.zeros(Ns)
        # w[0] = 1
        w = (1/np.sqrt(Ns)) * np.ones(Ns)
        acc = 0.0

        if los is LOS.LOS:
            alpha_0_u_p = np.dot(self._SectorUeLink_container[sector_ue_key].channel.h_los_u_s, w)
            acc += np.real(np.dot(alpha_0_u_p, alpha_0_u_p.conj()))

        for n in range(N):
            for m in range(M):
                alpha_n_m_u_p = np.dot(self._SectorUeLink_container[sector_ue_key].channel.h_nlos_u_s_n_m[:, :, n, m],
                                       w)
                acc += np.real(np.dot(alpha_n_m_u_p, alpha_n_m_u_p.conj()))

        self._SectorUeLink_container[sector_ue_key].RSRP = tx_power_per_RE - pathloss - shadow_fading + 10 * np.log10(
            acc) - 10 * np.log10(U)

    def UE_attach(self):
        """
        Performs UE attachment -> finds the sector for which UE senses the highest RSRP and 'connect' to them
        :return: updates attributes 'serving_sector' and 'serving_base_station' of the UserEquipment object
        """
        for ue in self.UEs:

            # Find links related to the ue:
            SectorUeLinks = [sector_id for (sector_id, ue_id) in self.SectorUeLink_container if ue_id == ue.ID]
            ue_links_rsrp = np.array([self._SectorUeLink_container[(sector_id, ue.ID)].RSRP
                                      for sector_id in SectorUeLinks])

            highestRSRP_idx = np.argmax(ue_links_rsrp)
            highestRSRP_link = SectorUeLinks[highestRSRP_idx]

            if ue_links_rsrp[highestRSRP_idx] >= self.UE_attach_threshold:
                ue.serving_sector = highestRSRP_link
                ue.serving_base_station = self._SectorUeLink_container[(highestRSRP_idx, ue.ID)].base_station_ID
            else:
                ue.serving_sector = None
                ue.serving_base_station = None

    def computeSINR(self):
        """
        Computes the SINR for the UEs in the UE list; it assumes that the desired signal is received from the serving
        sector and the interference is received for all other sectors.
        :return:
        """

        self.SINR_Matrix = np.zeros(len(self.UEs))

        for ue in self.UEs:
            if (ue.serving_sector is not None) and (ue.serving_base_station is not None):
                # Find all links for the UE
                bs_links = [bs_id for (bs_id, ue_id) in self.BsUeLink_container.keys() if ue_id == ue.ID]
                sec_link = [sec.ID for bs_id in bs_links for sec in self.BSs[bs_id].sectors]

                signal_power_dB = self._SectorUeLink_container[(ue.serving_sector, ue.ID)].RSRP

                # interference_plus_noise = 10 ** (ue.noise_floor/ 10)

                # Adjust noise level to the number of PRBs
                # bs_sector_id = [x for x, s in enumerate(self.BSs[ue.serving_base_station].sectors)
                #                 if s.ID == ue.serving_sector][0]

                # interference_plus_noise = 10 ** ((ue.noise_floor + 10 * np.log10(
                #     12 * self.BSs[ue.serving_base_station].sectors[bs_sector_id].number_of_PRBs)) / 10)
                # interference_plus_noise = 10 ** ((ue.noise_floor) / 10)

                # Noise per RE:
                interference_plus_noise = 10 ** ((-174.0 + 10 * np.log10(15_000)) / 10)

                for sector_id in (np.delete(np.array(sec_link), ue.serving_sector)):
                    interference_plus_noise = interference_plus_noise + 10 ** (
                                self._SectorUeLink_container[(sector_id, ue.ID)].RSRP / 10)

                interference_plus_noise_dB = 10 * np.log10(interference_plus_noise)
                self.SINR_Matrix[ue.ID] = signal_power_dB - interference_plus_noise_dB
            else:
                self.SINR_Matrix[ue.ID] = -np.inf

    def rsrpServingCell(self):
        self._rsrpServingCell = np.zeros(len(self.UEs))

        for ue in self.UEs:
            if (ue.serving_sector is not None) and (ue.serving_base_station is not None):

                self._rsrpServingCell[ue.ID] = self._SectorUeLink_container[(ue.serving_sector, ue.ID)].RSRP

        return self._rsrpServingCell

    @property
    def number_of_ue(self):
        return len(self.UEs)

    @property
    def number_of_bs(self):
        return len(self.BSs)

    @property
    def number_of_sector(self):
        n = 0
        for bs in self.BSs:
            n += len(bs.sectors)
        return n

    @property
    def network_limits(self):
        min_x = np.inf
        min_y = np.inf
        max_x = -np.inf
        max_y = -np.inf

        for bs in self.BSs:
            if bs.pos_x < min_x:
                min_x = bs.pos_x
            if bs.pos_x > max_x:
                max_x = bs.pos_x
            if bs.pos_y < min_y:
                min_y = bs.pos_y
            if bs.pos_y > max_y:
                max_y = bs.pos_y

        r = self.ISD / 2.0  # Hexagonal cell's inner radius
        R = (2.0 / np.sqrt(3.0)) * r  # Hexagonal cell's outer radius
        return [(min_x - R, min_y - R), (max_x + R, max_y + R)]

    @property
    def rsrp_matrix(self):
        self._rsrp_matrix = np.zeros((self.number_of_sector, self.number_of_ue))

        for link in self.SectorUeLink_container:
            self._rsrp_matrix[link[0], link[1]] = self._SectorUeLink_container[link].RSRP

        return self._rsrp_matrix


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

    def get_correlated_value_at(self, point_x, point_y):
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

    def update_correlation_grid(self):
        # Drawn new random values for correlation grid
        self.grid_values = np.random.normal(0.0, 1.0, (self.Nx, self.Ny))


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
    def GCS2LCS_angle(cls, alpha: float, beta: float, gamma: float, theta: float, phi: float):
        rho_unit = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]).T
        z_i = np.array([0, 0, 1]).T
        theta_prime = np.arccos(
            np.matmul(z_i, cls._inverse_rotation(alpha=alpha, beta=beta, gamma=gamma, point=rho_unit)))

        z_i = cls._inverse_rotation(alpha=alpha, beta=beta, gamma=gamma, point=rho_unit)
        phi_prime = np.arctan2(z_i[1], z_i[0])

        return theta_prime, phi_prime

    @classmethod
    def get_angles(cls, point):
        rho_unit = point / np.sqrt(np.dot(point, point))
        z_i = np.array([0, 0, 1]).T
        theta = np.arccos(np.matmul(z_i, rho_unit))
        phi = np.arctan2(rho_unit[1], rho_unit[0])

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
    def _inverse_rotation(alpha: float, beta: float, gamma: float, point: float):
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

    @staticmethod
    def _spherical_unit_vector_phi(phi):
        return np.array([-np.sin(phi), np.cos(phi), 0]).T

    @classmethod
    def polarized_field_component_transformation_matrix(cls, alpha, beta, gamma, theta_lcs, phi_lcs, theta_gcs,
                                                        phi_gcs):
        unit_gcs_theta = cls._spherical_unit_vector_theta(theta_gcs, phi_gcs)
        unit_gcs_phi = cls._spherical_unit_vector_phi(phi_gcs)

        unit_lcs_theta = cls._spherical_unit_vector_theta(theta_lcs, phi_lcs)
        unit_lcs_phi = cls._spherical_unit_vector_phi(phi_lcs)

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
