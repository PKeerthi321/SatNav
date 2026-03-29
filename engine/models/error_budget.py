import math
from engine.geometry.coordinates import ecef2aer_deg

# -----------------------
# Error budget per satellite (simple models)
# -----------------------
def compute_error_budget_for_sat(sat_row, rx_ecef):
    TEC = 10.0
    f = 1575.42e6
    iono_zenith = 40.3 * TEC / (f**2)
    az, el, _ = ecef2aer_deg([sat_row['X'], sat_row['Y'], sat_row['Z']], rx_ecef)
    el_rad = math.radians(max(el,1.0))

    iono_zenith = 40.3 * TEC / (f**2)
    iono_m = iono_zenith / math.sin(el_rad)

    ZHD = 2.3      # zenith hydrostatic delay (m)
    ZWD = 0.1      # zenith wet delay (m)

    mh = 1.0 / math.sin(el_rad)
    mw = 1.0 / math.sin(el_rad)

    tropo_m = ZHD * mh + ZWD * mw
    clock_m = 0.3
    sigma0 = 0.3  # nominal multipath std (m)
    multipath_m = sigma0 / math.sin(el_rad)

    total = math.sqrt(iono_m**2 + tropo_m**2 + clock_m**2 + multipath_m**2)
    return {
        'az': az, 'el': el,
        'iono_m': iono_m, 'tropo_m': tropo_m,
        'clock_m': clock_m, 'multipath_m': multipath_m,
        'total_m': total
    }