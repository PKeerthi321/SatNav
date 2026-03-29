import math
import numpy as np

def lla_to_ecef(lat_deg, lon_deg, alt_m):
    lat = math.radians(lat_deg); lon = math.radians(lon_deg); h = alt_m
    a = 6378137.0; f = 1.0/298.257223563; e2 = f*(2-f)
    N = a / math.sqrt(1 - e2 * math.sin(lat)**2)
    x = (N + h) * math.cos(lat) * math.cos(lon)
    y = (N + h) * math.cos(lat) * math.sin(lon)
    z = (N*(1-e2) + h) * math.sin(lat)
    return np.array([x, y, z], dtype=float)

def ecef_to_lla(x, y, z):
    a = 6378137.0; f = 1.0/298.257223563; e2 = f*(2-f)
    lon = math.atan2(y, x)
    p = math.sqrt(x*x + y*y)
    lat = math.atan2(z, p*(1 - e2))
    lat_prev = 1e9
    cnt = 0
    while abs(lat - lat_prev) > 1e-12 and cnt < 50:
        cnt += 1
        N = a / math.sqrt(1 - e2 * math.sin(lat)**2)
        alt = p / math.cos(lat) - N
        lat_prev = lat
        lat = math.atan2(z, p*(1 - e2 * (N/(N+alt))))
    N = a / math.sqrt(1 - e2 * math.sin(lat)**2)
    alt = p / math.cos(lat) - N
    return math.degrees(lat), math.degrees(lon), alt

def ecef2aer_deg(satXYZ, rxECEF):
    dx = satXYZ[0] - rxECEF[0]
    dy = satXYZ[1] - rxECEF[1]
    dz = satXYZ[2] - rxECEF[2]
    rng = math.sqrt(dx*dx + dy*dy + dz*dz)
    lat, lon, _ = ecef_to_lla(rxECEF[0], rxECEF[1], rxECEF[2])
    lat = math.radians(lat); lon = math.radians(lon)
    R = np.array([
        [-math.sin(lon), math.cos(lon), 0],
        [-math.sin(lat)*math.cos(lon), -math.sin(lat)*math.sin(lon), math.cos(lat)],
        [math.cos(lat)*math.cos(lon), math.cos(lat)*math.sin(lon), math.sin(lat)]
    ])
    enu = R.dot(np.array([dx,dy,dz]))
    east, north, up = enu[0], enu[1], enu[2]
    horiz = math.sqrt(east*east + north*north)
    az = math.degrees(math.atan2(east, north))
    if az < 0: az += 360
    el = math.degrees(math.atan2(up, horiz))
    return az, el, rng