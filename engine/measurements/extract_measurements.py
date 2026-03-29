"""
engine/measurements/extract_measurements.py

Extracts per-signal measurements from the obs_df produced by rinex_obs.py.

Signal priority for IISC station (RINEX 3, 30 s):
  GPS P1 : C1C → C1W → C1P (all NaN in this dataset — no L1 pseudorange)
  GPS P2 : C2L → C2W → C2P (C2L has 182 obs, best available)
  GPS P5 : C5Q → C5X → C5I (1373 obs — best usable pseudorange)
  GPS L1 : L1C → L1W → L1P
  GPS L2 : L2W → L2L → L2P
  GPS L5 : L5Q → L5X → L5I

Key insight for this dataset:
  - C1C/C1W are present as rows but all values are NaN (receiver config)
  - C2L (182 obs) and C5Q (1373 obs) are the only usable pseudoranges
  - MW combination uses (f1*L1 - f2*L2)/(f1-f2) - (f1*P1 + f2*P2)/(f1+f2)
    → not possible without P1
  - GF-only detection (L1C + L2W) is the primary method for this dataset
  - A P5-based wide-lane using L5Q/C5Q is tracked separately for future use
"""

import pandas as pd
import numpy as np

_KEEP_SYSTEMS = {'G', 'I'}

# GPS — ordered by quality/availability for IISC station
# C1C listed first per standard, but dataset has all NaN for C1C/C1W
_GPS_P1_TYPES = ['C1C', 'C1W', 'C1P', 'C1Y', 'C1M', 'C1L', 'C1Z', 'C1']
_GPS_P2_TYPES = ['C2L', 'C2W', 'C2P', 'C2C', 'C2S', 'C2']   # C2L first — 182 obs
_GPS_P5_TYPES = ['C5Q', 'C5X', 'C5I', 'C5P', 'C5']           # C5Q — 1373 obs
_GPS_L1_TYPES = ['L1C', 'L1W', 'L1P', 'L1Y', 'L1M', 'L1L', 'L1']
_GPS_L2_TYPES = ['L2W', 'L2L', 'L2P', 'L2C', 'L2S', 'L2']
_GPS_L5_TYPES = ['L5Q', 'L5X', 'L5I', 'L5P', 'L5']

# IRNSS
_IRNSS_P5_TYPES = ['C5A', 'C5B', 'C5C', 'C5']
_IRNSS_PS_TYPES = ['C9A', 'C9B', 'C9C', 'CS', 'C9']
_IRNSS_L5_TYPES = ['L5A', 'L5B', 'L5C', 'L5']
_IRNSS_LS_TYPES = ['L9A', 'L9B', 'L9C', 'LS', 'L9']


def _pick_best(sub_df, candidates, col):
    """
    Pick the best observation type from candidates list.
    'Best' = highest count of non-null, non-zero values.
    Returns a DataFrame with columns [UTC_Time, Sat, col].
    """
    available  = set(sub_df['ObsType'].unique())
    best_type  = None
    best_count = 0

    for obs_code in candidates:
        if obs_code not in available:
            continue
        mask  = sub_df['ObsType'] == obs_code
        vals  = sub_df.loc[mask, 'Value']
        # Count real observations: not NaN and not zero
        count = int(vals.notna().sum()) - int((vals == 0.0).sum())
        if count > best_count:
            best_count = count
            best_type  = obs_code

    if best_type is None or best_count == 0:
        return pd.DataFrame(columns=['UTC_Time', 'Sat', col])

    s = sub_df[sub_df['ObsType'] == best_type].copy()
    s = s[s['Value'].notna() & (s['Value'] != 0.0)]
    s = s.drop_duplicates(['UTC_Time', 'Sat'], keep='first')
    return s[['UTC_Time', 'Sat', 'Value']].rename(columns={'Value': col})


def _pick_type_name(sub_df, candidates):
    """Return (chosen_type, count) for logging."""
    available = set(sub_df['ObsType'].unique())
    for obs_code in candidates:
        if obs_code not in available:
            continue
        vals  = sub_df.loc[sub_df['ObsType'] == obs_code, 'Value']
        count = int(vals.notna().sum()) - int((vals == 0.0).sum())
        if count > 0:
            return obs_code, count
    # Still report the first available type even if count is 0 (for diagnostics)
    for obs_code in candidates:
        if obs_code in available:
            vals  = sub_df.loc[sub_df['ObsType'] == obs_code, 'Value']
            count = int(vals.notna().sum())
            return obs_code, count
    return None, 0


def extract_measurements(obs_df):
    """
    Extract per-satellite measurements from the flat obs_df.

    Returns a DataFrame with columns:
      epoch, sat, sys, P1, P2, P5, PS, L1, L2, L5, LS

    Notes for IISC DOY 008 2026 dataset:
      - P1 will be NaN for all epochs (C1C/C1W all NaN in source)
      - P2  = C2L (182 obs)
      - P5  = C5Q (1373 obs) — best available pseudorange
      - L1  = L1C (4625 obs)
      - L2  = L2W (4750 obs)
      - L5  = L5Q GPS + L5A IRNSS
    """
    valid = (
        obs_df['Sys'].isin(_KEEP_SYSTEMS) &
        obs_df['Sat'].str.match(r'^[GI]\d{2}$')
    )
    df = obs_df[valid].copy()
    if df.empty:
        return pd.DataFrame(
            columns=['epoch', 'sat', 'sys',
                     'P1', 'P2', 'P5', 'PS', 'L1', 'L2', 'L5', 'LS'])

    gps   = df[df['Sys'] == 'G']
    irnss = df[df['Sys'] == 'I']

    # ── Extract each signal ───────────────────────────────────────────────────
    P1  = _pick_best(gps,   _GPS_P1_TYPES,    'P1')
    P2  = _pick_best(gps,   _GPS_P2_TYPES,    'P2')
    P5g = _pick_best(gps,   _GPS_P5_TYPES,    'P5g')
    L1  = _pick_best(gps,   _GPS_L1_TYPES,    'L1')
    L2  = _pick_best(gps,   _GPS_L2_TYPES,    'L2')
    L5g = _pick_best(gps,   _GPS_L5_TYPES,    'L5g')
    P5i = _pick_best(irnss, _IRNSS_P5_TYPES,  'P5i')
    L5i = _pick_best(irnss, _IRNSS_L5_TYPES,  'L5i')
    PS  = _pick_best(irnss, _IRNSS_PS_TYPES,  'PS')
    LS  = _pick_best(irnss, _IRNSS_LS_TYPES,  'LS')

    # ── Build base with all unique (UTC_Time, Sat, Sys) combos ────────────────
    base = df[['UTC_Time', 'Sat', 'Sys']].drop_duplicates(
        ['UTC_Time', 'Sat']).copy()
    meas = base

    # Merge core signals
    for sig in (P1, P2, L1, L2):
        if not sig.empty:
            meas = pd.merge(meas, sig, on=['UTC_Time', 'Sat'], how='left')

    # GPS P5 / L5
    if not P5g.empty:
        meas = pd.merge(meas, P5g.rename(columns={'P5g': 'P5'}),
                        on=['UTC_Time', 'Sat'], how='left')
    if not L5g.empty:
        meas = pd.merge(meas, L5g.rename(columns={'L5g': 'L5'}),
                        on=['UTC_Time', 'Sat'], how='left')

    # IRNSS P5 — merge then combine with GPS P5
    if not P5i.empty:
        meas = pd.merge(meas, P5i.rename(columns={'P5i': '_P5i'}),
                        on=['UTC_Time', 'Sat'], how='left')
        if 'P5' not in meas.columns:
            meas['P5'] = meas['_P5i']
        else:
            meas['P5'] = meas['P5'].combine_first(meas['_P5i'])
        meas = meas.drop(columns=['_P5i'], errors='ignore')

    # IRNSS L5
    if not L5i.empty:
        meas = pd.merge(meas, L5i.rename(columns={'L5i': '_L5i'}),
                        on=['UTC_Time', 'Sat'], how='left')
        if 'L5' not in meas.columns:
            meas['L5'] = meas['_L5i']
        else:
            meas['L5'] = meas['L5'].combine_first(meas['_L5i'])
        meas = meas.drop(columns=['_L5i'], errors='ignore')

    # IRNSS secondary signals
    for sig in (PS, LS):
        if not sig.empty:
            meas = pd.merge(meas, sig, on=['UTC_Time', 'Sat'], how='left')

    # Ensure all columns exist
    for col in ('P1', 'P2', 'P5', 'PS', 'L1', 'L2', 'L5', 'LS'):
        if col not in meas.columns:
            meas[col] = np.nan

    # Add epoch index (integer 0..N-1 per sorted UTC_Time)
    times = sorted(meas['UTC_Time'].unique())
    t2ep  = {t: i for i, t in enumerate(times)}
    meas['epoch'] = meas['UTC_Time'].map(t2ep)

    meas = meas.rename(columns={'Sat': 'sat', 'Sys': 'sys'})
    meas = meas.drop(columns=['UTC_Time'])

    # ── Print extraction report ───────────────────────────────────────────────
    _log_extraction(gps, irnss)

    return meas.sort_values(['epoch', 'sat']).reset_index(drop=True)


def _log_extraction(gps, irnss):
    """Print detailed signal extraction report."""
    print()
    for sys_df, label, groups in [
        (gps, 'GPS', [
            ('P1', _GPS_P1_TYPES),
            ('P2', _GPS_P2_TYPES),
            ('P5', _GPS_P5_TYPES),
            ('L1', _GPS_L1_TYPES),
            ('L2', _GPS_L2_TYPES),
            ('L5', _GPS_L5_TYPES),
        ]),
        (irnss, 'IRNSS', [
            ('P5', _IRNSS_P5_TYPES),
            ('PS', _IRNSS_PS_TYPES),
            ('L5', _IRNSS_L5_TYPES),
            ('LS', _IRNSS_LS_TYPES),
        ]),
    ]:
        if sys_df.empty:
            continue
        available = set(sys_df['ObsType'].unique())
        for sig, candidates in groups:
            chosen, count = _pick_type_name(sys_df, candidates)
            if chosen:
                # Warn if count is 0 (type present but all NaN)
                flag = '  ⚠ ALL NaN' if count == 0 else ''
                print(f'[extract] {label:5s} {sig} → {chosen:<4s}  '
                      f'({count} obs){flag}')
            else:
                tried = candidates[:4]
                print(f'[extract] {label:5s} {sig} → NOT FOUND  '
                      f'(tried: {tried})')
    print()


def signal_availability_report(meas_df):
    """Return dict of signal → observation count."""
    return {
        col: int(meas_df[col].notna().sum())
        for col in ('P1', 'P2', 'P5', 'PS', 'L1', 'L2', 'L5', 'LS')
        if col in meas_df.columns
    }


def pivot_epoch(meas_df, epoch):
    """Return {sat: {signal: value}} dict for one epoch."""
    sub  = meas_df[meas_df['epoch'] == epoch]
    cols = ['P1', 'P2', 'P5', 'PS', 'L1', 'L2', 'L5', 'LS']
    return {
        row['sat']: {
            c: (float(row[c]) if c in row.index and pd.notna(row[c]) else None)
            for c in cols
        }
        for _, row in sub.iterrows()
    }