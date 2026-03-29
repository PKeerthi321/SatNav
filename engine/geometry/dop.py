import numpy as np
import math
import pandas as pd


def compute_dop_table(satpos_df, rx_ecef):
    from engine.geometry.coordinates import ecef2aer_deg
    good   = ~satpos_df[['X','Y','Z']].isnull().any(axis=1)
    satpos = satpos_df[good].reset_index(drop=True)
    n = satpos.shape[0]
    if n < 4:
        raise ValueError(f'Need at least 4 satellites, got {n}')
    G = []
    elevations = []
    azimuths   = []
    for _, row in satpos.iterrows():
        dx = row['X'] - rx_ecef[0]
        dy = row['Y'] - rx_ecef[1]
        dz = row['Z'] - rx_ecef[2]
        rnorm = math.sqrt(dx*dx + dy*dy + dz*dz)
        if rnorm == 0:
            raise ValueError(f'Satellite {row["Sat"]} at same position as receiver')
        G.append([-dx/rnorm, -dy/rnorm, -dz/rnorm, 1.0])
        az, el, _ = ecef2aer_deg([row['X'], row['Y'], row['Z']], rx_ecef)
        elevations.append(round(el, 3))
        azimuths.append(round(az, 3))
    G = np.array(G)
    try:
        Q = np.linalg.inv(G.T @ G)
    except np.linalg.LinAlgError:
        Q = np.linalg.pinv(G.T @ G)
    pdop = math.sqrt(max(0, Q[0,0] + Q[1,1] + Q[2,2]))
    hdop = math.sqrt(max(0, Q[0,0] + Q[1,1]))
    vdop = math.sqrt(max(0, Q[2,2]))
    gdop = math.sqrt(max(0, np.trace(Q)))
    tdop = math.sqrt(max(0, Q[3,3]))
    importance = []
    for k in range(n):
        keep = [i for i in range(n) if i != k]
        if len(keep) < 4:
            importance.append(np.inf); continue
        Gsub = G[keep, :]
        try:
            Qsub = np.linalg.inv(Gsub.T @ Gsub)
        except np.linalg.LinAlgError:
            Qsub = np.linalg.pinv(Gsub.T @ Gsub)
        pdop_without = math.sqrt(max(0, Qsub[0,0] + Qsub[1,1] + Qsub[2,2]))
        importance.append(round(pdop_without - pdop, 4))
    out = satpos.copy()
    out['El']   = elevations
    out['Az']   = azimuths
    out['PDOP'] = round(pdop, 4)
    out['HDOP'] = round(hdop, 4)
    out['VDOP'] = round(vdop, 4)
    out['GDOP'] = round(gdop, 4)
    out['TDOP'] = round(tdop, 4)
    out['dPDOP_if_removed'] = importance
    return out
