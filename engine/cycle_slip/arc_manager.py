"""
engine/cycle_slip/arc_manager.py

Arc tracking with Welford running statistics for ΔN1 and ΔN2 sequences.
Feeds m1, m2 values to slip_corrector.py for each detected slip epoch.

Reference: Wang & Huang (2023), KSCE Journal of Civil Engineering,
           27(12):5329-5337. DOI: 10.1007/s12205-023-0388-2
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class WelfordState:
    """Online mean and variance using Welford's algorithm."""
    n: int = 0
    mean: float = 0.0
    M2: float = 0.0  # sum of squared deviations

    def update(self, value: float):
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self) -> float:
        return self.M2 / self.n if self.n > 1 else 1.0

    @property
    def std(self) -> float:
        return float(np.sqrt(self.variance))

    def reset(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0


@dataclass
class ArcState:
    """State for one continuous satellite arc."""
    prn: str
    start_epoch: int
    epochs: List[int] = field(default_factory=list)

    # Phase observations stored per epoch
    L1: List[float] = field(default_factory=list)
    L2: List[float] = field(default_factory=list)
    L5: List[float] = field(default_factory=list)

    # Running stats for GF sequences
    welford_N1: WelfordState = field(default_factory=WelfordState)
    welford_N2: WelfordState = field(default_factory=WelfordState)
    welford_gf: WelfordState = field(default_factory=WelfordState)

    # Slip history within this arc: {epoch_idx: slip_type}
    slip_history: Dict[int, str] = field(default_factory=dict)

    # m1, m2 at each slip epoch (for corrector)
    slip_m1: List[float] = field(default_factory=list)
    slip_m2: List[float] = field(default_factory=list)
    slip_epochs_list: List[int] = field(default_factory=list)

    # GF combination values per epoch
    gf_values: List[float] = field(default_factory=list)

    def add_epoch(self, epoch: int, L1: float, L2: float,
                  L5: Optional[float] = None):
        """Add a new epoch to this arc and update running statistics."""
        self.epochs.append(epoch)
        self.L1.append(L1)
        self.L2.append(L2)
        self.L5.append(L5 if L5 is not None else np.nan)

        n = len(self.L1)
        if n >= 2:
            # Wavelengths (cycles to metres already handled upstream)
            LAMBDA1 = 0.190293672798
            LAMBDA2 = 0.244210213425

            dN1 = self.L1[-1] - self.L1[-2]
            dN2 = self.L2[-1] - self.L2[-2]
            dphi = dN1 - (LAMBDA2 / LAMBDA1) * dN2

            self.welford_N1.update(dN1)
            self.welford_N2.update(dN2)
            self.welford_gf.update(dphi)
            self.gf_values.append(dphi)
        else:
            self.gf_values.append(0.0)

    def record_slip(self, epoch_idx: int, slip_type: str):
        """Record a detected slip at epoch_idx within this arc."""
        self.slip_history[epoch_idx] = slip_type
        self.slip_epochs_list.append(epoch_idx)
        self.slip_m1.append(self.welford_N1.std)
        self.slip_m2.append(self.welford_N2.std)

    @property
    def length(self) -> int:
        return len(self.epochs)

    @property
    def m1(self) -> float:
        """Current std dev of ΔN1 sequence."""
        return self.welford_N1.std

    @property
    def m2(self) -> float:
        """Current std dev of ΔN2 sequence."""
        return self.welford_N2.std

    def L1_array(self) -> np.ndarray:
        return np.array(self.L1)

    def L2_array(self) -> np.ndarray:
        return np.array(self.L2)


class ArcManager:
    """
    Manages all satellite arcs across all epochs.
    Tracks data gaps, initialises new arcs, and feeds
    slip information to slip_corrector.
    """

    # Data gap threshold in seconds
    GAP_THRESHOLD_SEC: float = 40.0

    def __init__(self, interval_sec: float = 30.0):
        self.interval_sec = interval_sec
        # Active arcs: {prn: ArcState}
        self._active: Dict[str, ArcState] = {}
        # Completed arcs: {prn: [ArcState, ...]}
        self._completed: Dict[str, List[ArcState]] = {}
        # Last seen epoch per PRN
        self._last_epoch: Dict[str, int] = {}

    def process_epoch(self, epoch: int, prn: str,
                      L1: float, L2: float,
                      L5: Optional[float] = None) -> str:
        """
        Process one observation for one satellite.

        Returns
        -------
        str : 'NEW_ARC' | 'CONTINUED' | 'GAP_RESET'
        """
        status = 'CONTINUED'

        if prn not in self._active:
            # First observation for this PRN
            self._active[prn] = ArcState(prn=prn, start_epoch=epoch)
            status = 'NEW_ARC'
        else:
            last = self._last_epoch.get(prn, epoch)
            gap_sec = (epoch - last) * self.interval_sec

            if gap_sec > self.GAP_THRESHOLD_SEC:
                # Data gap — close current arc, start new one
                self._close_arc(prn)
                self._active[prn] = ArcState(prn=prn, start_epoch=epoch)
                status = 'GAP_RESET'

        self._active[prn].add_epoch(epoch, L1, L2, L5)
        self._last_epoch[prn] = epoch
        return status

    def record_slip(self, prn: str, epoch_idx: int, slip_type: str):
        """Record a detected slip in the active arc for prn."""
        if prn in self._active:
            self._active[prn].record_slip(epoch_idx, slip_type)

    def get_arc(self, prn: str) -> Optional[ArcState]:
        """Get the currently active arc for a PRN."""
        return self._active.get(prn)

    def get_slip_correction_inputs(self, prn: str):
        """
        Return inputs needed by slip_corrector.correct_arc().
        Returns (L1_array, L2_array, slip_epochs, m1_seq, m2_seq)
        or None if no active arc.
        """
        arc = self._active.get(prn)
        if arc is None or arc.length < 2:
            return None

        return (
            arc.L1_array(),
            arc.L2_array(),
            arc.slip_epochs_list.copy(),
            arc.slip_m1.copy(),
            arc.slip_m2.copy()
        )

    def _close_arc(self, prn: str):
        """Move active arc to completed list."""
        if prn in self._active:
            arc = self._active.pop(prn)
            if prn not in self._completed:
                self._completed[prn] = []
            self._completed[prn].append(arc)

    def close_all(self):
        """Close all active arcs at end of processing."""
        for prn in list(self._active.keys()):
            self._close_arc(prn)

    def summary(self) -> Dict:
        """Return arc count summary per PRN."""
        result = {}
        for prn, arcs in self._completed.items():
            result[prn] = {
                'num_arcs': len(arcs),
                'total_epochs': sum(a.length for a in arcs),
                'total_slips': sum(len(a.slip_history) for a in arcs)
            }
        return result