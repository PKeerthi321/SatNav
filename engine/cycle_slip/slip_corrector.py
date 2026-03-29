"""
engine/cycle_slip/slip_corrector.py

GF-based cycle slip correction using Dynamic Test method.
Reference: Wang & Huang (2023), KSCE Journal of Civil Engineering,
           27(12):5329-5337. DOI: 10.1007/s12205-023-0388-2

Algorithm:
  Given geometry-free combination:
      Δφ = ΔN1 - (λ2/λ1) * ΔN2                         ... Eq.(3)
  
  For each detected slip epoch ti:
      Solve C1, C2 (integer cycle slip values) such that:
          |C1 - (77/60)*C2 - Δφ(ti)| < 0.07             ... Eq.(6)
          C1 ∈ [ΔN1(ti) - 3*m1, ΔN1(ti) + 3*m1]
          C2 ∈ [ΔN2(ti) - 3*m2, ΔN2(ti) + 3*m2]
  
  Special types handled:
      (9,7) type:  |9*C1 - (77/60)*7*C2| < 0.07         ... Eq.(7)
      (77,60) type: |77*C1 - (77/60)*60*C2| = 0         ... Eq.(8)
"""

import numpy as np
from typing import Optional, Tuple


# GPS L1/L2 wavelengths (metres)
LAMBDA1 = 0.190293672798  # L1 = c / f1, f1 = 1575.42 MHz
LAMBDA2 = 0.244210213425  # L2 = c / f2, f2 = 1227.60 MHz
RATIO   = LAMBDA2 / LAMBDA1  # ≈ 77/60 = 1.28333...

# Detection tolerance on Δφ (cycles) — 3σ for mφ = ±0.01 cycles
DPHI_TOL = 0.07


def Q(a: float, b: float) -> list:
    """
    Directed rounding function Q[a, b] from Wang & Huang (2023) Section 4.
    Returns all integers in [a, b] with directed rounding at boundaries.

    If a >= 0: a rounds toward +inf, b rounds toward 0
    If a < 0 and b <= 0: a rounds toward 0, b rounds toward -inf
    If a < 0 and b > 0: both round toward 0
    """
    if a >= 0:
        lo = int(np.ceil(a))
        hi = int(np.floor(b)) if b >= 0 else int(np.ceil(b))
    else:
        if b <= 0:
            lo = int(np.ceil(a))
            hi = int(np.floor(b))
        else:
            lo = int(np.ceil(a))
            hi = int(np.floor(b))

    return list(range(lo, hi + 1))


def _solve_normal(dN1: float, dN2: float, dphi: float,
                  m1: float, m2: float) -> Optional[Tuple[int, int]]:
    """
    Solve normal cycle slip type using Eq.(6).
    Returns (C1, C2) integer pair or None if no unique solution.
    """
    c1_range = range(int(np.floor(dN1 - 3*m1)),
                     int(np.ceil(dN1 + 3*m1)) + 1)
    c2_range = range(int(np.floor(dN2 - 3*m2)),
                     int(np.ceil(dN2 + 3*m2)) + 1)

    candidates = []
    for c1 in c1_range:
        for c2 in c2_range:
            residual = abs(c1 - RATIO * c2 - dphi)
            if residual < DPHI_TOL:
                candidates.append((c1, c2))

    if len(candidates) == 1:
        return candidates[0]
    return None


def _solve_97_type(dN1: float, dN2: float,
                   m1: float, m2: float) -> Optional[Tuple[int, int]]:
    """
    Solve (9,7) special cycle slip type using Eq.(7).
    These have Δφ ≈ 0 but ΔN1/ΔN2 are abnormal.
    Returns (C1*9, C2*7) correction or None.
    """
    c1_candidates = Q((dN1 - 3*m1) / 9, (dN1 + 3*m1) / 9)
    c2_candidates = Q((dN2 - 3*m2) / 7, (dN2 + 3*m2) / 7)

    solutions = []
    for c1 in c1_candidates:
        for c2 in c2_candidates:
            if abs(9*c1 - RATIO * 7*c2) < DPHI_TOL:
                solutions.append((9*c1, 7*c2))

    if len(solutions) == 1:
        return solutions[0]
    return None


def _solve_7760_type(dN1: float, dN2: float,
                     m1: float, m2: float) -> Optional[Tuple[int, int]]:
    """
    Solve (77,60) special cycle slip type using Eq.(8).
    Returns (C1*77, C2*60) correction or None.
    """
    c1_candidates = Q((dN1 - 3*m1) / 77, (dN1 + 3*m1) / 77)
    c2_candidates = Q((dN2 - 3*m2) / 60, (dN2 + 3*m2) / 60)

    solutions = []
    for c1 in c1_candidates:
        for c2 in c2_candidates:
            if abs(77*c1 - RATIO * 60*c2) < 1e-6:
                solutions.append((77*c1, 60*c2))

    if len(solutions) == 1:
        return solutions[0]
    return None


def correct_slip(L1: np.ndarray,
                 L2: np.ndarray,
                 slip_epoch: int,
                 m1: float,
                 m2: float) -> Tuple[np.ndarray, np.ndarray, bool, str]:
    """
    Attempt to correct a cycle slip at slip_epoch using GF dynamic test.

    Parameters
    ----------
    L1 : np.ndarray
        L1 phase observations in cycles (full arc, length N).
    L2 : np.ndarray
        L2 phase observations in cycles (full arc, length N).
    slip_epoch : int
        Index within L1/L2 where the slip occurred.
    m1 : float
        Running std dev of ΔN1 sequence (from arc_manager Welford).
    m2 : float
        Running std dev of ΔN2 sequence (from arc_manager Welford).

    Returns
    -------
    L1_corrected : np.ndarray
        L1 array with slip corrected from slip_epoch onward.
    L2_corrected : np.ndarray
        L2 array with slip corrected from slip_epoch onward.
    success : bool
        True if correction was found and applied.
    slip_type : str
        One of 'NORMAL', '(9,7)', '(77,60)', 'FAILED'.
    """
    if slip_epoch < 1 or slip_epoch >= len(L1):
        return L1.copy(), L2.copy(), False, 'FAILED'

    # --- Step 1: Compute ΔN1, ΔN2, Δφ at slip epoch ---
    dN1  = L1[slip_epoch] - L1[slip_epoch - 1]
    dN2  = L2[slip_epoch] - L2[slip_epoch - 1]
    dphi = dN1 - RATIO * dN2

    L1_out = L1.copy()
    L2_out = L2.copy()

    # --- Step 2: Check if Δφ is out of normal range ---
    dphi_out_of_range = abs(dphi) > DPHI_TOL

    # --- Step 3a: Normal type (Δφ abnormal) ---
    if dphi_out_of_range:
        result = _solve_normal(dN1, dN2, dphi, m1, m2)
        if result is not None:
            C1, C2 = result
            L1_out[slip_epoch:] -= C1
            L2_out[slip_epoch:] -= C2
            return L1_out, L2_out, True, 'NORMAL'

    # --- Step 3b: Special (9,7) type (Δφ near zero but ΔN1/ΔN2 abnormal) ---
    result_97 = _solve_97_type(dN1, dN2, m1, m2)
    if result_97 is not None:
        C1, C2 = result_97
        L1_out[slip_epoch:] -= C1
        L2_out[slip_epoch:] -= C2
        return L1_out, L2_out, True, '(9,7)'

    # --- Step 3c: Special (77,60) type ---
    result_7760 = _solve_7760_type(dN1, dN2, m1, m2)
    if result_7760 is not None:
        C1, C2 = result_7760
        L1_out[slip_epoch:] -= C1
        L2_out[slip_epoch:] -= C2
        return L1_out, L2_out, True, '(77,60)'

    # --- Step 4: Correction failed — caller should re-initialise arc ---
    return L1.copy(), L2.copy(), False, 'FAILED'


def correct_arc(L1: np.ndarray,
                L2: np.ndarray,
                slip_epochs: list,
                m1_seq: list,
                m2_seq: list) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Correct all detected slip epochs in an arc sequentially.
    Applies corrections in order; each correction updates the array
    before processing the next slip, matching paper Section 4 Step 5.

    Parameters
    ----------
    L1, L2       : phase arrays for one arc (cycles)
    slip_epochs  : list of epoch indices where slips were detected
    m1_seq       : list of m1 values at each slip epoch
    m2_seq       : list of m2 values at each slip epoch

    Returns
    -------
    L1_corrected, L2_corrected : corrected phase arrays
    report : dict with keys = epoch indices, values = slip_type string
    """
    L1_c = L1.copy()
    L2_c = L2.copy()
    report = {}

    for i, ep in enumerate(sorted(slip_epochs)):
        m1 = m1_seq[i] if i < len(m1_seq) else 1.5
        m2 = m2_seq[i] if i < len(m2_seq) else 1.2

        L1_c, L2_c, success, stype = correct_slip(L1_c, L2_c, ep, m1, m2)
        report[ep] = stype if success else 'FAILED'

    return L1_c, L2_c, report