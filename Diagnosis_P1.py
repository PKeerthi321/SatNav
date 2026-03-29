"""
diagnose_p1.py
Run this from your gnss_project folder to diagnose why C1C returns 0 obs.
Usage: python diagnose_p1.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.io.rinex_obs import read_rinex_obs

OBS_FILE = r"F:\KEERTHI KA BIO\IIT Tirupathi Internship\RINEX_Files\IISC00IND_R_20260080000_01D_30S_MO.rnx"

print("Loading RINEX...")
obs_df = read_rinex_obs(OBS_FILE)

print(f"\nTotal rows  : {len(obs_df)}")
print(f"Columns     : {list(obs_df.columns)}")
print(f"\nUnique Sys  : {sorted(obs_df['Sys'].unique())}")

# ── Check what obs types are present for GPS satellites ──────────────────────
gps = obs_df[obs_df['Sys'] == 'G']
print(f"\nGPS rows total : {len(gps)}")
print(f"GPS ObsTypes   : {sorted(gps['ObsType'].unique())}")

# ── Count each pseudorange-like signal for GPS ────────────────────────────────
print("\n--- GPS Pseudorange ObsType counts ---")
pr_types = ['C1C', 'C1W', 'C1X', 'C1P', 'C1L', 'C2W', 'C2L', 'C2P', 'C5Q']
for t in pr_types:
    sub = gps[gps['ObsType'] == t]
    non_null = sub['Value'].notna().sum() if 'Value' in sub.columns else len(sub)
    print(f"  {t:6s} : {len(sub):6d} rows   non-null={non_null}")

# ── Show a sample of C1C rows ─────────────────────────────────────────────────
c1c = gps[gps['ObsType'] == 'C1C'].head(5)
print(f"\n--- Sample C1C rows (first 5) ---")
print(c1c.to_string())

# ── Check if 'Value' column exists and its dtype ─────────────────────────────
print(f"\n--- DataFrame dtypes ---")
print(obs_df.dtypes)

# ── Check what extract_measurements sees ─────────────────────────────────────
print("\n--- Testing pivot for G01 epoch 0 ---")
ep0 = obs_df[obs_df['Sys'] == 'G'].copy()
epochs = sorted(ep0['UTC_Time'].unique())
if epochs:
    ep_data = ep0[ep0['UTC_Time'] == epochs[0]]
    sats = ep_data['Sat'].unique()
    print(f"  Epoch 0 time : {epochs[0]}")
    print(f"  Satellites   : {list(sats)[:5]}")
    if len(sats) > 0:
        sat0 = ep_data[ep_data['Sat'] == sats[0]]
        print(f"\n  {sats[0]} observations:")
        print(sat0[['ObsType','Value']].to_string())