import math
import numpy as np
import pandas as pd
from engine.io.rinex_nav import parse_rinex_nav
from engine.geometry.coordinates import lla_to_ecef

def satpos_placeholder_from_id(sid, epoch_time, ref_ecef):
    seed = sum(ord(ch) for ch in sid) + int(epoch_time.timestamp())%10000
    az = (seed * 37) % 360
    el = 20 + (seed % 60)  # 20..79 deg
    if sid.startswith('G'):
        R = 20200000.0
    elif sid.startswith('I'):
        R = 35786000.0
    else:
        R = 22000000.0
    lat0 = math.radians(20.0); lon0 = math.radians(78.0)
    azr = math.radians(az); elr = math.radians(el)
    east = math.cos(elr)*math.sin(azr)
    north = math.cos(elr)*math.cos(azr)
    up = math.sin(elr)
    enu = np.array([east, north, up]) * R
    Rmat = np.array([
        [-math.sin(lon0), math.cos(lon0), 0.0],
        [-math.sin(lat0)*math.cos(lon0), -math.sin(lat0)*math.sin(lon0), math.cos(lat0)],
        [ math.cos(lat0)*math.cos(lon0),  math.cos(lat0)*math.sin(lon0), math.sin(lat0)]
    ])
    ecef_vec = Rmat.T.dot(enu)
    ecef = ref_ecef + ecef_vec
    return float(ecef[0]), float(ecef[1]), float(ecef[2])

def compute_sat_positions_from_nav(navgps_files, navirnss_files, epoch_time, sat_list):
    eph = {}
    for f in (navgps_files or []):
        try:
            eph.update(parse_rinex_nav(f))
        except:
            pass
    for f in (navirnss_files or []):
        try:
            eph.update(parse_rinex_nav(f))
        except:
            pass
    Sat=[]; Sys=[]; X=[]; Y=[]; Z=[]
    ref = lla_to_ecef(20.0,78.0,0.0)
    for s in sat_list:
        Sat.append(s); Sys.append(s[0] if len(s)>0 else '?')
        key1 = s; key2 = s[1:] if len(s)>1 else s
        found = None
        for key in (key1, key2, key2.lstrip('0')):
            if key in eph:
                found = eph[key]; break
        if found is not None:
            try:
                x,y,z = satpos_from_eph(found, epoch_time)
                X.append(x); Y.append(y); Z.append(z)
                continue
            except Exception:
                pass
        x,y,z = satpos_placeholder_from_id(s, epoch_time, ref)
        X.append(x); Y.append(y); Z.append(z)
    return pd.DataFrame({'Sat':Sat,'Sys':Sys,'X':X,'Y':Y,'Z':Z})

def satpos_from_eph(e, epoch_time):
    mu = 3.986005e14
    OMEGA_E_DOT = 7.2921151467e-5
    A = float(e.get('A', 0.0))
    if A <= 0:
        raise ValueError('Invalid A')
    n0 = math.sqrt(mu/(A**3))
    deltaN = float(e.get('deltaN',0.0))
    toe = float(e.get('Toe',0.0))
    t = (epoch_time.timestamp() % 604800)
    tk = t - toe
    while tk > 302400: tk -= 604800
    while tk < -302400: tk += 604800
    n = n0 + deltaN
    M0 = float(e.get('M0',0.0))
    M = M0 + n*tk
    ecc = float(e.get('e',0.0))
    E = M + ecc * math.sin(M)
    for _ in range(50):
        f = E - ecc*math.sin(E) - M
        df = 1 - ecc*math.cos(E)
        dE = -f / df
        E += dE
        if abs(dE) < 1e-12:
            break
    v = math.atan2(math.sqrt(1-ecc**2)*math.sin(E), math.cos(E)-ecc)
    phi = v + float(e.get('w',0.0))
    Cuc = float(e.get('Cuc',0.0)); Cus = float(e.get('Cus',0.0))
    Crc = float(e.get('Crc',0.0)); Crs = float(e.get('Crs',0.0))
    Cic = float(e.get('Cic',0.0)); Cis = float(e.get('Cis',0.0))
    u = phi + Cuc*math.cos(2*phi) + Cus*math.sin(2*phi)
    r = A*(1 - ecc*math.cos(E)) + Crc*math.cos(2*phi) + Crs*math.sin(2*phi)
    i = float(e.get('i0',0.0)) + Cic*math.cos(2*phi) + Cis*math.sin(2*phi)
    x_orb = r*math.cos(u); y_orb = r*math.sin(u)
    Omega0 = float(e.get('omega0',0.0)); OmegaDot = float(e.get('OmegaDot',0.0))
    Omega = Omega0 + (OmegaDot - OMEGA_E_DOT)*tk - OMEGA_E_DOT*(toe + tk)
    x = x_orb*math.cos(Omega) - y_orb*math.cos(i)*math.sin(Omega)
    y = x_orb*math.sin(Omega) + y_orb*math.cos(i)*math.cos(Omega)
    z = y_orb*math.sin(i)
    return float(x), float(y), float(z)