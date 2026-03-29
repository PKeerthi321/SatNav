"""
engine/cycle_slip/mw_detector.py

Combined GF + MW cycle slip detection.
Reference: Wang & Huang (2023), KSCE Journal of Civil Engineering,
           27(12):5329-5337. DOI: 10.1007/s12205-023-0388-2

Detection logic
---------------
Primary  : GF Dynamic Test (paper Sections 2-4)
             Δφ = ΔN1 − (λ2/λ1)·ΔN2
             T0  = {i : |Δφ[i]| > tol}
             T12 = {i : |ΔN1[i]| > 3m1  AND  |ΔN2[i]| > 3m2}
             T   = T0 ∪ T12
             Classify: CONFIRMED | GF_SLIP | GF_ONLY | CLEAN | INIT

Secondary : MW arc test (Blewitt 1990 / TurboEdit)
             MW  = (f1·L1 − f2·L2)/(f1−f2) − (f1·P1 + f2·P2)/(f1+f2)
             flag if |MW[i] − mean| > k·sigma  OR  |ΔMW| > 0.5 cyc
             Status upgrade: GF_SLIP → CONFIRMED, CLEAN → MW_ONLY

Tertiary  : L5 wide-lane fallback (when P1 absent, P5=C5Q available)
             WL5 = L5 − P5/λ5
             flag if |ΔWL5| > 0.5 cyc
             Status upgrade: same as MW

DPHI_TOL scaling
----------------
  Paper: 0.07 cyc for ≤15 s data
  30 s : 0.07 × √(30/15) = 0.099 → use 0.10 cycles

Status codes
------------
  INIT      — first epoch of arc
  CLEAN     — no slip from any method
  GF_SLIP   — GF Δφ out of range, ΔN1/ΔN2 in range
  GF_ONLY   — ΔN1/ΔN2 outliers, Δφ in range (special slip type)
  CONFIRMED — GF Δφ AND ΔN1/ΔN2 both flagged
  MW_ONLY   — MW/WL5 flagged, GF clean (code-noise or small slip)
  MW_CONFIRMED — both GF and MW/WL5 flagged (highest confidence)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional

from engine.cycle_slip.mw_combination import (
    LAMBDA1, LAMBDA2, LAMBDA5, LAMBDA_WL,
    F1, F2, F5, RATIO_21,
    mw_combination, wl5_combination,
    mw_arc_stats, mw_detect_arc,
    delta_N1_N2,
)

# ── Thresholds ────────────────────────────────────────────────────────────────
DPHI_TOL_BASE    = 0.07    # paper value for ≤15 s
INTERVAL_REF_SEC = 15.0    # reference interval from paper
DPHI_TOL         = 0.10    # effective at 30 s (exported for GUI)
MW_TOL           = 0.5     # MW epoch-difference threshold (cycles)
WL5_TOL          = 0.5     # WL5 epoch-difference threshold (cycles)
MW_K             = 5.0     # k-sigma for arc-level MW test


def _scale_threshold(interval_sec: float) -> float:
    """Scale Δφ threshold for sampling interval."""
    return DPHI_TOL_BASE * np.sqrt(interval_sec / INTERVAL_REF_SEC)


# ── Single-epoch GF classify ──────────────────────────────────────────────────

def _gf_detect(dN1: float, dN2: float, dphi: float,
               m1: float, m2: float, tol: float) -> str:
    """
    Classify one epoch via GF Dynamic Test.
    Returns: CLEAN | GF_SLIP | GF_ONLY | CONFIRMED
    """
    dphi_out    = abs(dphi) > tol
    dN1_out     = (abs(dN1) > 3.0 * m1) if m1 > 0 else False
    dN2_out     = (abs(dN2) > 3.0 * m2) if m2 > 0 else False
    both_dN_out = dN1_out and dN2_out

    if   dphi_out and both_dN_out: return 'CONFIRMED'
    elif dphi_out:                 return 'GF_SLIP'
    elif both_dN_out:              return 'GF_ONLY'
    return 'CLEAN'


def _upgrade_status(gf_status: str, mw_flag: bool) -> str:
    """
    Upgrade GF status using MW/WL5 flag.
      CLEAN    + MW flag → MW_ONLY
      GF_SLIP  + MW flag → MW_CONFIRMED
      CONFIRMED+ MW flag → MW_CONFIRMED
      GF_ONLY  + MW flag → MW_CONFIRMED
    """
    if not mw_flag:
        return gf_status
    if gf_status == 'CLEAN':
        return 'MW_ONLY'
    return 'MW_CONFIRMED'


# ── Full arc detection ────────────────────────────────────────────────────────

def detect_arc(L1:          np.ndarray,
               L2:          np.ndarray,
               P1:          Optional[np.ndarray] = None,
               P2:          Optional[np.ndarray] = None,
               L5:          Optional[np.ndarray] = None,
               P5:          Optional[np.ndarray] = None,
               interval_sec: float = 30.0,
               m1_seq:      Optional[np.ndarray] = None,
               m2_seq:      Optional[np.ndarray] = None
               ) -> Tuple[np.ndarray, list]:
    """
    Run combined GF + MW detection over one satellite arc.

    Parameters
    ----------
    L1, L2       : phase arrays in cycles, length N  (required)
    P1, P2       : pseudorange arrays in metres (optional, for standard MW)
    L5, P5       : L5 phase/pseudorange (optional, for WL5 fallback)
    interval_sec : sampling interval in seconds
    m1_seq, m2_seq : external Welford std dev arrays (optional)

    Returns
    -------
    statuses : np.ndarray of str, length N
    infos    : list of dicts, length N
    """
    L1 = np.asarray(L1, dtype=float)
    L2 = np.asarray(L2, dtype=float)
    N  = len(L1)
    tol = _scale_threshold(interval_sec)

    statuses = np.full(N, 'INIT', dtype=object)
    infos    = [{}] * N

    if m1_seq is None: m1_seq = np.full(N, 1.5)
    if m2_seq is None: m2_seq = np.full(N, 1.2)

    # ── Compute MW sequence for whole arc (NaN where P1/P2 absent) ────────────
    have_mw  = (P1 is not None and P2 is not None)
    have_wl5 = (L5 is not None and P5 is not None)

    mw_seq  = np.full(N, np.nan)
    wl5_seq = np.full(N, np.nan)

    if have_mw:
        P1a = np.asarray(P1, dtype=float)
        P2a = np.asarray(P2, dtype=float)
        valid_mw = ~np.isnan(P1a) & ~np.isnan(P2a)
        if valid_mw.any():
            mw_all = mw_combination(L1 * LAMBDA1, L2 * LAMBDA2, P1a, P2a)
            mw_seq = np.where(valid_mw, mw_all, np.nan)

    if have_wl5:
        L5a = np.asarray(L5, dtype=float)
        P5a = np.asarray(P5, dtype=float)
        valid_wl5 = ~np.isnan(L5a) & ~np.isnan(P5a)
        if valid_wl5.any():
            wl5_all = wl5_combination(L5a, P5a)
            wl5_seq = np.where(valid_wl5, wl5_all, np.nan)

    # Arc-level MW statistics (running mean/std)
    mw_mean, mw_std = mw_arc_stats(mw_seq)

    # ── Welford online stats for ΔN1, ΔN2 (GF) ───────────────────────────────
    wN1_mean = wN1_M2 = 0.0;  wN1_n = 0
    wN2_mean = wN2_M2 = 0.0;  wN2_n = 0

    for i in range(1, N):
        dN1  = float(L1[i] - L1[i-1])
        dN2  = float(L2[i] - L2[i-1])
        dphi = dN1 - RATIO_21 * dN2

        # Update Welford only on clean epochs
        if statuses[i-1] in ('CLEAN', 'INIT'):
            wN1_n += 1
            d = dN1 - wN1_mean
            wN1_mean += d / wN1_n
            wN1_M2   += d * (dN1 - wN1_mean)

            wN2_n += 1
            d2 = dN2 - wN2_mean
            wN2_mean += d2 / wN2_n
            wN2_M2   += d2 * (dN2 - wN2_mean)

        m1 = max(float(np.sqrt(wN1_M2/wN1_n)) if wN1_n > 1 else 1.5, 1.5)
        m2 = max(float(np.sqrt(wN2_M2/wN2_n)) if wN2_n > 1 else 1.2, 1.2)

        # ── GF classification ─────────────────────────────────────────────────
        gf_status = _gf_detect(dN1, dN2, dphi, m1, m2, tol)

        # ── MW flag ───────────────────────────────────────────────────────────
        mw_flag = False
        mw_val  = mw_seq[i]
        wl5_val = wl5_seq[i]

        # Standard MW: k-sigma arc test OR epoch-diff test
        if not np.isnan(mw_val):
            prev_mw = mw_seq[i-1] if i > 0 else np.nan
            if not np.isnan(mw_mean[i-1]) and not np.isnan(mw_std[i-1]):
                if mw_std[i-1] > 0:
                    mw_flag = abs(mw_val - mw_mean[i-1]) > MW_K * mw_std[i-1]
            if not mw_flag and not np.isnan(prev_mw):
                mw_flag = abs(mw_val - prev_mw) > MW_TOL

        # WL5 fallback (when standard MW unavailable)
        if not mw_flag and not np.isnan(wl5_val):
            prev_wl5 = wl5_seq[i-1] if i > 0 else np.nan
            if not np.isnan(prev_wl5):
                mw_flag = abs(wl5_val - prev_wl5) > WL5_TOL

        # ── Combine GF + MW ───────────────────────────────────────────────────
        status      = _upgrade_status(gf_status, mw_flag)
        statuses[i] = status
        infos[i]    = {
            'dN1':    dN1,
            'dN2':    dN2,
            'dphi':   dphi,
            'tol':    tol,
            'm1':     m1,
            'm2':     m2,
            'mw':     float(mw_val)  if not np.isnan(mw_val)  else None,
            'wl5':    float(wl5_val) if not np.isnan(wl5_val) else None,
            'mw_flag': mw_flag,
        }

    return statuses, infos


# ── Single epoch interface ────────────────────────────────────────────────────

def detect_epoch(L1_prev: float, L2_prev: float,
                 L1_curr: float, L2_curr: float,
                 m1: float = 1.5, m2: float = 1.2,
                 n_epochs: int = 2,
                 P1: float = np.nan, P2: float = np.nan,
                 interval_sec: float = 30.0) -> Tuple[str, dict]:
    """Single epoch wrapper — used by compatibility shim."""
    if n_epochs < 2:
        return 'INIT', {}

    dN1  = L1_curr - L1_prev
    dN2  = L2_curr - L2_prev
    dphi = dN1 - RATIO_21 * dN2
    tol  = _scale_threshold(interval_sec)

    m1_use = max(m1, 1.5)
    m2_use = max(m2, 1.2)
    gf_status = _gf_detect(dN1, dN2, dphi, m1_use, m2_use, tol)

    mw_val = np.nan
    if not (np.isnan(P1) or np.isnan(P2)):
        mw_val = float(mw_combination(
            np.array([L1_curr * LAMBDA1]),
            np.array([L2_curr * LAMBDA2]),
            np.array([P1]), np.array([P2]))[0])

    info = {
        'dN1': dN1, 'dN2': dN2, 'dphi': dphi,
        'tol': tol, 'm1': m1_use, 'm2': m2_use,
        'mw': mw_val if not np.isnan(mw_val) else None,
    }
    return gf_status, info


# ── Backward-compatibility shim ───────────────────────────────────────────────

def results_to_dataframe(statuses_dict, infos_dict):
    """Convert multi-arc detect_arc results to a flat DataFrame."""
    rows = []
    for prn, statuses in statuses_dict.items():
        infos = infos_dict.get(prn, [{}] * len(statuses))
        for i, (status, info) in enumerate(zip(statuses, infos)):
            row = {'sat': prn, 'epoch_idx': i, 'status': status, 'flag': status}
            row.update({k: v for k, v in (info or {}).items()})
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=[
            'sat', 'epoch_idx', 'status', 'flag',
            'dN1', 'dN2', 'dphi', 'tol', 'm1', 'm2', 'mw', 'wl5'])
    return pd.DataFrame(rows)


def detect_all_arcs(meas_df, interval_sec: float = 30.0):
    """Run detect_arc across all satellites in a measurements DataFrame."""
    statuses_dict = {}
    infos_dict    = {}
    for sat in meas_df['sat'].unique():
        sub = meas_df[meas_df['sat'] == sat].sort_values('epoch')
        L1  = sub['L1'].values.astype(float) if 'L1' in sub.columns else None
        L2  = sub['L2'].values.astype(float) if 'L2' in sub.columns else None
        if L1 is None or L2 is None:
            continue
        mask = ~np.isnan(L1) & ~np.isnan(L2)
        if mask.sum() < 2:
            continue
        P1  = sub['P1'].values.astype(float) if 'P1' in sub.columns else None
        P2  = sub['P2'].values.astype(float) if 'P2' in sub.columns else None
        L5  = sub['L5'].values.astype(float) if 'L5' in sub.columns else None
        P5  = sub['P5'].values.astype(float) if 'P5' in sub.columns else None
        statuses, infos = detect_arc(
            L1[mask], L2[mask],
            P1=P1[mask] if P1 is not None else None,
            P2=P2[mask] if P2 is not None else None,
            L5=L5[mask] if L5 is not None else None,
            P5=P5[mask] if P5 is not None else None,
            interval_sec=interval_sec,
        )
        statuses_dict[sat] = statuses
        infos_dict[sat]    = infos
    return results_to_dataframe(statuses_dict, infos_dict)


class CycleSlipDetector:
    """Compatibility wrapper around detect_arc()."""
    def __init__(self, interval_sec=30.0, *args, **kwargs):
        self.interval_sec = interval_sec

    def detect(self, L1, L2, P1=None, P2=None, L5=None, P5=None):
        return detect_arc(
            np.asarray(L1, dtype=float),
            np.asarray(L2, dtype=float),
            P1=np.asarray(P1, dtype=float) if P1 is not None else None,
            P2=np.asarray(P2, dtype=float) if P2 is not None else None,
            L5=np.asarray(L5, dtype=float) if L5 is not None else None,
            P5=np.asarray(P5, dtype=float) if P5 is not None else None,
            interval_sec=self.interval_sec,
        )

    def detect_epoch(self, L1_prev, L2_prev, L1_curr, L2_curr,
                     m1=1.5, m2=1.2, n_epochs=2, P1=np.nan, P2=np.nan):
        return detect_epoch(L1_prev, L2_prev, L1_curr, L2_curr,
                            m1, m2, n_epochs, P1, P2, self.interval_sec)

    def results_to_dataframe(self, sd, id_):
        return results_to_dataframe(sd, id_)