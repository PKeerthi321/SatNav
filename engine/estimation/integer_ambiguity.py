import numpy as np


GPS_L1_WAVELENGTH = 0.190293672798365


def compute_geometric_range(sat_pos, rx_pos):
    """
    Compute geometric distance between satellite and receiver
    """

    sat = np.array(sat_pos)
    rx = np.array(rx_pos)

    rho = np.linalg.norm(sat - rx)

    return rho


def estimate_float_ambiguity(carrier_phase_cycles, wavelength, geometric_range):
    """
    Estimate float ambiguity
    """

    carrier_phase_m = carrier_phase_cycles * wavelength

    N_float = (carrier_phase_m - geometric_range) / wavelength

    return N_float


def fix_integer_ambiguity(N_float):
    """
    Fix ambiguity using simple rounding
    """

    return np.round(N_float)


def compute_residual(carrier_phase_cycles, wavelength, geometric_range, N_int):

    carrier_phase_m = carrier_phase_cycles * wavelength

    model = geometric_range + wavelength * N_int

    residual = carrier_phase_m - model

    return residual