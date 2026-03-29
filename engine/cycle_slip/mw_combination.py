"""
engine/cycle_slip/mw_combination.py

Melbourne-Wubbena (MW) and Geometry-Free (GF) combination module.

Reference: Wang & Huang (2023), KSCE Journal of Civil Engineering,
           27(12):5329-5337. DOI: 10.1007/s12205-023-0388-2

Combinations implemented
------------------------
1. GF (Geometry-Free) — Eq.(3) of paper
       GF_m   = L1_m - L2_m                            [metres]
       delta_phi = dN1 - (lam2/lam1)*dN2               [cycles, epoch-diff]

2. Standard MW (Melbourne-Wubbena) — requires L1, L2, P1, P2
       MW = (f1*L1 - f2*L2)/(f1-f2) - (f1*P1 + f2*P2)/(f1+f2)  [wide-lane cycles]
       Slip detection: |delta_MW| > MW_TOL (typically 0.5 cycles)

3. L5-based wide-lane — uses L5 + P5 (C5Q available: 1373 obs in IISC dataset)
       WL_L5 = L5_cyc - P5_m/lam5                      [L5 cycles]
       Slip detection: |delta_WL_L5| > WL5_TOL

Dataset note (IISC DOY 008 2026):
   P1 (C1C/C1W): all NaN  ->  standard MW unavailable
   P2 (C2L)    : 182 obs  ->  standard MW computable for these epochs only
   P5 (C5Q)    : 1373 obs ->  L5-based wide-lane is the primary fallback
   L5 (L5Q)    : 2145 obs
   L1 (L1C)    : 4625 obs ->  GF primary method
   L2 (L2W)    : 4750 obs ->  GF primary method

MW arc-level statistics (Blewitt 1990 / TurboEdit approach)
------------------------------------------------------------
For each satellite arc compute running mean and std dev of MW sequence.
A slip is flagged when |MW[i] - mean| > k*sigma (k=5 default).
This is integrated with the GF Dynamic Test for combined detection.
"""

import numpy as np
from typing import Optional, Tuple


# ── Physical constants ────────────────────────────────────────────────────────
C_LIGHT = 299792458.0          # speed of light (m/s)

F1 = 1575.42e6                 # GPS L1 frequency (Hz)
F2 = 1227.60e6                 # GPS L2 frequency (Hz)
F5 = 1176.45e6                 # GPS L5 frequency (Hz)

LAMBDA1   = C_LIGHT / F1       # 0.190293672798 m
LAMBDA2   = C_LIGHT / F2       # 0.244210213425 m
LAMBDA5   = C_LIGHT / F5       # 0.254828049    m
LAMBDA_WL = C_LIGHT / (F1-F2)  # 0.861918400    m  wide-lane L1-L2
LAMBDA_NL = C_LIGHT / (F1+F2)  # 0.106953378    m  narrow-lane
LAMBDA_WL5 = C_LIGHT / (F1-F5) # 0.751020422    m  L1-L5 wide-lane

RATIO_21  = LAMBDA2 / LAMBDA1  # f1/f2 = lam2/lam1 ≈ 77/60 = 1.28333
RATIO_51  = LAMBDA5 / LAMBDA1  # lam5/lam1 ≈ 77/60... no: f1/f5

# Coefficient for ionosphere-free (IF) combination
ALPHA_IF  = F1**2 / (F1**2 - F2**2)
BETA_IF   = F2**2 / (F1**2 - F2**2)

# MW detection thresholds
MW_TOL    = 0.5    # cycles — standard MW slip threshold (Blewitt 1990)
WL5_TOL   = 0.5    # cycles — L5-based wide-lane threshold
MW_K      = 5.0    # k-sigma multiplier for arc-level MW test


# ── 1. Geometry-free (GF) combination ────────────────────────────────────────

def gf_combination(L1_m: np.ndarray,
                   L2_m: np.ndarray) -> np.ndarray:
    """
    GF phase combination in metres.
        GF = L1_m - L2_m
    Eliminates geometry; retains ionosphere + ambiguities.
    """
    return np.asarray(L1_m) - np.asarray(L2_m)


def gf_combination_cycles(L1_cyc: np.ndarray,
                           L2_cyc: np.ndarray) -> np.ndarray:
    """
    GF combination in cycles (non-differenced).
        phi_GF = L1 - (lam2/lam1)*L2  = N1 - (lam2/lam1)*N2  + iono term
    """
    return np.asarray(L1_cyc) - RATIO_21 * np.asarray(L2_cyc)


def gf_epoch_diff(L1_cyc: np.ndarray,
                  L2_cyc: np.ndarray) -> np.ndarray:
    """
    Epoch-differenced GF sequence (Eq.3, Wang & Huang 2023).
        delta_phi[i] = dN1[i] - (lam2/lam1)*dN2[i]
    First element is 0.0 (no previous epoch).
    """
    L1 = np.asarray(L1_cyc, dtype=float)
    L2 = np.asarray(L2_cyc, dtype=float)
    N  = len(L1)
    dphi = np.zeros(N)
    if N >= 2:
        dN1       = np.diff(L1)
        dN2       = np.diff(L2)
        dphi[1:]  = dN1 - RATIO_21 * dN2
    return dphi


# ── 2. Standard Melbourne-Wubbena combination ─────────────────────────────────

def mw_combination(L1_m: np.ndarray,
                   L2_m: np.ndarray,
                   P1_m: np.ndarray,
                   P2_m: np.ndarray) -> np.ndarray:
    """
    Standard Melbourne-Wubbena (MW) wide-lane combination.

        MW = (f1*L1 - f2*L2) / (f1-f2)  -  (f1*P1 + f2*P2) / (f1+f2)
           [result in metres, divide by lam_WL for cycles]

    Parameters
    ----------
    L1_m, L2_m : phase in metres (L1_cyc * lam1, L2_cyc * lam2)
    P1_m, P2_m : pseudorange in metres

    Returns
    -------
    mw : wide-lane combination in cycles (NaN where P1/P2 are NaN)
    """
    L1 = np.asarray(L1_m, dtype=float)
    L2 = np.asarray(L2_m, dtype=float)
    P1 = np.asarray(P1_m, dtype=float)
    P2 = np.asarray(P2_m, dtype=float)

    phase_part = (F1 * L1 - F2 * L2) / (F1 - F2)
    code_part  = (F1 * P1 + F2 * P2) / (F1 + F2)
    return (phase_part - code_part) / LAMBDA_WL


def mw_from_cycles(L1_cyc: np.ndarray,
                   L2_cyc: np.ndarray,
                   P1_m:   np.ndarray,
                   P2_m:   np.ndarray) -> np.ndarray:
    """
    MW combination taking phase in cycles and pseudorange in metres.
    Converts phase to metres internally.
    """
    return mw_combination(
        np.asarray(L1_cyc) * LAMBDA1,
        np.asarray(L2_cyc) * LAMBDA2,
        P1_m, P2_m
    )


# ── 3. L5-based wide-lane combination ────────────────────────────────────────

def wl5_combination(L5_cyc: np.ndarray,
                    P5_m:   np.ndarray) -> np.ndarray:
    """
    L5-based code-minus-phase wide-lane combination.
    Useful when P1 is absent (as in IISC dataset).

        WL5 = L5_cyc  -  P5_m / lam5

    This gives a wide-lane-like observable that is sensitive to cycle
    slips on L5. The mean of WL5 over a clean arc equals the L5 integer
    ambiguity; a jump indicates a slip.

    Parameters
    ----------
    L5_cyc : L5 phase in cycles
    P5_m   : L5 pseudorange in metres (C5Q)

    Returns
    -------
    wl5 : code-minus-phase in L5 cycles (NaN where P5 is NaN)
    """
    L5 = np.asarray(L5_cyc, dtype=float)
    P5 = np.asarray(P5_m,   dtype=float)
    return L5 - P5 / LAMBDA5


def wl5_from_meas(meas_df, sat: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract L5_cyc and P5_m for one satellite and compute WL5.

    Returns
    -------
    epochs : np.ndarray of epoch indices
    wl5    : np.ndarray of WL5 values (NaN where L5 or P5 absent)
    """
    import pandas as pd
    sub = meas_df[meas_df['sat'] == sat].sort_values('epoch')
    epochs = sub['epoch'].values.astype(float)
    L5 = sub['L5'].values.astype(float) if 'L5' in sub.columns else np.full(len(sub), np.nan)
    P5 = sub['P5'].values.astype(float) if 'P5' in sub.columns else np.full(len(sub), np.nan)
    wl5 = wl5_combination(L5, P5)
    return epochs, wl5


# ── 4. Arc-level MW statistics (Blewitt / TurboEdit approach) ─────────────────

def mw_arc_stats(mw_seq: np.ndarray,
                 window: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute running mean and std dev of MW sequence over an arc.
    Uses an expanding window (Welford online) up to `window` epochs,
    then a sliding window of fixed size.

    Parameters
    ----------
    mw_seq : MW combination values in cycles, length N (may contain NaN)
    window : sliding window size (default 50 epochs = 25 min at 30 s)

    Returns
    -------
    mw_mean : running mean,    length N
    mw_std  : running std dev, length N  (NaN for first 2 valid epochs)
    """
    N        = len(mw_seq)
    mw_mean  = np.full(N, np.nan)
    mw_std   = np.full(N, np.nan)

    valid_idx = np.where(~np.isnan(mw_seq))[0]
    if len(valid_idx) < 2:
        return mw_mean, mw_std

    # Welford online for expanding window, then sliding
    buf_vals = []
    w_mean   = 0.0
    w_M2     = 0.0
    w_n      = 0

    for i in range(N):
        v = mw_seq[i]
        if np.isnan(v):
            continue

        if w_n < window:
            # Expanding window — Welford update
            w_n  += 1
            delta = v - w_mean
            w_mean += delta / w_n
            w_M2   += delta * (v - w_mean)
            buf_vals.append(v)
        else:
            # Sliding window — remove oldest, add new
            oldest  = buf_vals.pop(0)
            buf_vals.append(v)
            w_mean  = float(np.mean(buf_vals))
            w_M2    = float(np.sum((np.array(buf_vals) - w_mean)**2))

        mw_mean[i] = w_mean
        if w_n > 1:
            mw_std[i]  = float(np.sqrt(w_M2 / (min(w_n, window) - 1)))

    return mw_mean, mw_std


def mw_detect_arc(mw_seq:  np.ndarray,
                  k:       float = MW_K,
                  tol:     float = MW_TOL) -> np.ndarray:
    """
    Detect cycle slips in a MW sequence using arc statistics.

    Two-criteria detection:
      1. |MW[i] - mean| > k * sigma   (arc-level k-sigma test)
      2. |delta_MW[i]|  > tol         (epoch-difference absolute test)
    A slip is flagged if EITHER criterion is met.

    Parameters
    ----------
    mw_seq : MW sequence in cycles, length N
    k      : k-sigma multiplier (default 5)
    tol    : absolute epoch-difference threshold in cycles (default 0.5)

    Returns
    -------
    flags : np.ndarray of bool, length N  (True = slip detected)
    """
    N     = len(mw_seq)
    flags = np.zeros(N, dtype=bool)

    mw_mean, mw_std = mw_arc_stats(mw_seq)

    for i in range(1, N):
        v = mw_seq[i]
        if np.isnan(v):
            continue

        # Criterion 1: k-sigma arc test
        if not np.isnan(mw_mean[i-1]) and not np.isnan(mw_std[i-1]):
            if mw_std[i-1] > 0 and abs(v - mw_mean[i-1]) > k * mw_std[i-1]:
                flags[i] = True
                continue

        # Criterion 2: absolute epoch difference
        prev = mw_seq[i-1]
        if not np.isnan(prev) and abs(v - prev) > tol:
            flags[i] = True

    return flags


# ── 5. Combined GF + MW delta sequences ───────────────────────────────────────

def delta_N1_N2(L1_cyc: np.ndarray,
                L2_cyc: np.ndarray,
                P1_m:   Optional[np.ndarray] = None,
                P2_m:   Optional[np.ndarray] = None,
                P5_m:   Optional[np.ndarray] = None,
                L5_cyc: Optional[np.ndarray] = None):
    """
    Compute all epoch-differenced sequences needed for detection.

    Parameters
    ----------
    L1_cyc, L2_cyc : phase in cycles (required)
    P1_m, P2_m     : L1/L2 pseudorange in metres (optional)
    P5_m, L5_cyc   : L5 pseudorange/phase (optional, for WL5 fallback)

    Returns
    -------
    dN1   : epoch diff of L1 (cycles), length N
    dN2   : epoch diff of L2 (cycles), length N
    dphi  : GF epoch diff = dN1 - ratio*dN2, length N
    mw    : MW sequence in wide-lane cycles, length N (NaN if P1/P2 absent)
    wl5   : WL5 sequence in L5 cycles, length N (NaN if L5/P5 absent)
    """
    L1 = np.asarray(L1_cyc, dtype=float)
    L2 = np.asarray(L2_cyc, dtype=float)
    N  = len(L1)

    dN1  = np.zeros(N)
    dN2  = np.zeros(N)
    dphi = np.zeros(N)
    mw   = np.full(N, np.nan)
    wl5  = np.full(N, np.nan)

    if N >= 2:
        dN1[1:]  = np.diff(L1)
        dN2[1:]  = np.diff(L2)
        dphi[1:] = dN1[1:] - RATIO_21 * dN2[1:]

    # Standard MW — only where P1 and P2 are both valid
    if P1_m is not None and P2_m is not None:
        P1 = np.asarray(P1_m, dtype=float)
        P2 = np.asarray(P2_m, dtype=float)
        valid = ~np.isnan(P1) & ~np.isnan(P2)
        if valid.any():
            mw_all = mw_combination(L1 * LAMBDA1, L2 * LAMBDA2, P1, P2)
            mw     = np.where(valid, mw_all, np.nan)

    # L5-based wide-lane — fallback when standard MW unavailable
    if L5_cyc is not None and P5_m is not None:
        L5 = np.asarray(L5_cyc, dtype=float)
        P5 = np.asarray(P5_m,   dtype=float)
        valid5 = ~np.isnan(L5) & ~np.isnan(P5)
        if valid5.any():
            wl5_all = wl5_combination(L5, P5)
            wl5     = np.where(valid5, wl5_all, np.nan)

    return dN1, dN2, dphi, mw, wl5


# ── 6. Ionosphere-free (IF) combination ──────────────────────────────────────

def if_combination_phase(L1_m: np.ndarray,
                          L2_m: np.ndarray) -> np.ndarray:
    """
    Ionosphere-free phase combination (for PPP, future use).
        L_IF = (f1^2 * L1 - f2^2 * L2) / (f1^2 - f2^2)
    Result in metres.
    """
    return ALPHA_IF * np.asarray(L1_m) - BETA_IF * np.asarray(L2_m)


def if_combination_code(P1_m: np.ndarray,
                         P2_m: np.ndarray) -> np.ndarray:
    """
    Ionosphere-free pseudorange combination (for PPP, future use).
        P_IF = (f1^2 * P1 - f2^2 * P2) / (f1^2 - f2^2)
    Result in metres.
    """
    return ALPHA_IF * np.asarray(P1_m) - BETA_IF * np.asarray(P2_m)


# ── 7. Convenience accessors ──────────────────────────────────────────────────

def wide_lane_wavelength()  -> float: return LAMBDA_WL
def narrow_lane_wavelength() -> float: return LAMBDA_NL
def l5_widlane_wavelength() -> float: return LAMBDA_WL5