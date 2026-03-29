import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from engine.estimation.integer_ambiguity import (
    compute_geometric_range,
    estimate_float_ambiguity,
    fix_integer_ambiguity,
    compute_residual,
    GPS_L1_WAVELENGTH
)


# Example receiver position (ECEF meters)
rx_pos = np.array([1113194.9, -4841692.8, 3985354.6])


# Example satellite position (ECEF meters)
sat_pos = np.array([15600000.0, 7540000.0, 20140000.0])


# Example carrier phase observation (cycles)
carrier_phase = 110234567.45


# Step 1 — geometric range
rho = compute_geometric_range(sat_pos, rx_pos)


# Step 2 — float ambiguity
N_float = estimate_float_ambiguity(
    carrier_phase,
    GPS_L1_WAVELENGTH,
    rho
)


# Step 3 — integer ambiguity
N_int = fix_integer_ambiguity(N_float)


# Step 4 — residual check
residual = compute_residual(
    carrier_phase,
    GPS_L1_WAVELENGTH,
    rho,
    N_int
)


print("\nFloat ambiguity:", N_float)
print("Integer ambiguity:", N_int)
print("Residual:", residual)