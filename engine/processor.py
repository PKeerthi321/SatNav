import numpy as np
from engine.geometry.coordinates import lla_to_ecef

class GNSSProcessor:
    def __init__(self):
        self.obs_df=None; self.nav_gps=[]; self.nav_irnss=[]
        self.sat_positions=None; self.dop_table=None; self.error_table=None
        self.meas_df=None; self.slip_df=None; self.slip_detector=None; self.iar_results={}

    def load_obs(self, path):
        from engine.io.rinex_obs import read_rinex_obs
        self.obs_df = read_rinex_obs(path)

    def load_nav(self, gps_files, irnss_files):
        self.nav_gps=gps_files; self.nav_irnss=irnss_files

    def compute_geometry(self):
        from engine.geometry.orbit import compute_sat_positions_from_nav
        epoch = self.obs_df['UTC_Time'].iloc[0]
        sats  = sorted(self.obs_df['Sat'].unique())
        self.sat_positions = compute_sat_positions_from_nav(self.nav_gps,self.nav_irnss,epoch,sats)

    def compute_dop(self):
        from engine.geometry.dop import compute_dop_table
        self.dop_table = compute_dop_table(self.sat_positions, lla_to_ecef(20.,78.,0.))

    def compute_errors(self):
        from engine.models.error_budget import compute_error_budget_for_sat
        rx=lla_to_ecef(20.,78.,0.)
        self.error_table=[compute_error_budget_for_sat(r,rx) for _,r in self.dop_table.iterrows()]

    def extract_measurements(self):
        if self.obs_df is None: raise RuntimeError('Load OBS first.')
        from engine.measurements.extract_measurements import extract_measurements
        self.meas_df = extract_measurements(self.obs_df)
        return self.meas_df

    def compute_cycle_slips(self, mw_threshold_cyc=0.5, gf_threshold_m=0.05,
                            init_epochs=4, interval_sec=30.0):
        """
        Detect cycle slips using GF Dynamic Test (Wang & Huang 2023).

        Processes all satellites arc-by-arc using detect_arc().
        Gaps > 40 s trigger a new arc automatically via ArcManager.

        Parameters
        ----------
        mw_threshold_cyc : kept for API compatibility (not used — P1 absent)
        gf_threshold_m   : kept for API compatibility (threshold now auto-scaled)
        init_epochs      : kept for API compatibility
        interval_sec     : sampling interval in seconds (default 30 s)

        Returns
        -------
        slip_df : DataFrame with columns
            sat, epoch_idx, arc_start, status, dN1, dN2, dphi, tol, m1, m2
        """
        if self.meas_df is None:
            raise RuntimeError('Call extract_measurements() first.')

        import pandas as pd
        from engine.cycle_slip.mw_detector import detect_arc, DPHI_TOL, MW_TOL
        from engine.cycle_slip.arc_manager import ArcManager

        n_ep = self.meas_df['epoch'].nunique()
        n_st = self.meas_df['sat'].nunique()
        print(f'[processor] compute_cycle_slips: meas_df={self.meas_df.shape} '
              f'epochs={n_ep} sats={n_st} interval={interval_sec}s')

        # ── Build arcs via ArcManager ─────────────────────────────────────────
        arc_mgr = ArcManager(interval_sec=interval_sec)
        epochs  = sorted(self.meas_df['epoch'].unique())

        for idx in epochs:
            sub = self.meas_df[self.meas_df['epoch'] == idx]
            for _, row in sub.iterrows():
                L1 = row.get('L1') if 'L1' in row.index else None
                L2 = row.get('L2') if 'L2' in row.index else None
                L5 = row.get('L5') if 'L5' in row.index else None
                if L1 is None or pd.isna(L1) or L2 is None or pd.isna(L2):
                    continue
                arc_mgr.process_epoch(
                    idx, row['sat'], float(L1), float(L2),
                    float(L5) if (L5 is not None and not pd.isna(L5)) else None
                )

        arc_mgr.close_all()

        # ── Run detect_arc on each completed arc ──────────────────────────────
        rows = []
        for prn, arcs in arc_mgr._completed.items():
            for arc in arcs:
                if arc.length < 2:
                    continue
                L1_arr = arc.L1_array()
                L2_arr = arc.L2_array()

                # Pull P1, P2, L5, P5 for this arc's epochs
                sat_sub = self.meas_df[self.meas_df['sat'] == prn]
                arc_ep  = arc.epochs  # list of epoch indices

                def _arr(col):
                    if col not in self.meas_df.columns:
                        return None
                    vals = sat_sub[sat_sub['epoch'].isin(arc_ep)].sort_values('epoch')[col].values
                    a = vals.astype(float)
                    return a if not np.all(np.isnan(a)) else None

                P1_arr = _arr('P1')
                P2_arr = _arr('P2')
                L5_arr = _arr('L5')
                P5_arr = _arr('P5')

                # Align lengths (arc may be shorter if some epochs had NaN L1/L2)
                n = len(L1_arr)
                def _trim(a): return a[:n] if a is not None and len(a) >= n else None

                statuses, infos = detect_arc(
                    L1_arr, L2_arr,
                    P1=_trim(P1_arr), P2=_trim(P2_arr),
                    L5=_trim(L5_arr), P5=_trim(P5_arr),
                    interval_sec=interval_sec,
                )

                for i, (status, info) in enumerate(zip(statuses, infos)):
                    ep_idx = arc.epochs[i] if i < len(arc.epochs) else i

                    # Record slip into arc for corrector
                    if status in ('GF_SLIP', 'GF_ONLY', 'CONFIRMED'):
                        arc.slip_epochs_list.append(i)
                        arc.slip_m1.append(info.get('m1', 1.5))
                        arc.slip_m2.append(info.get('m2', 1.2))
                        arc.slip_history[i] = status

                    rows.append({
                        'sat':       prn,
                        'epoch_idx': ep_idx,
                        'arc_start': arc.start_epoch,
                        'status':    status,
                        'flag':      status,   # alias — slip_summary() uses 'flag'
                        'dN1':  info.get('dN1',  np.nan),
                        'dN2':  info.get('dN2',  np.nan),
                        'dphi': info.get('dphi', np.nan),
                        'tol':  info.get('tol',  np.nan),
                        'm1':   info.get('m1',   np.nan),
                        'm2':   info.get('m2',   np.nan),
                    })

        slip_df = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=['sat', 'epoch_idx', 'arc_start', 'status', 'flag',
                     'dN1', 'dN2', 'dphi', 'tol', 'm1', 'm2'])

        if not slip_df.empty:
            counts = slip_df['status'].value_counts()
            for flag, cnt in counts.items():
                print(f'[processor]   {flag:<12s}: {cnt}')

        # Store arc_mgr so corrector can access it later
        self.arc_mgr   = arc_mgr
        self.slip_df   = slip_df
        self.slip_detector = arc_mgr   # alias for backward compat
        return self.slip_df

    def run_lambda_iar(self, a_float, Q_aa, ratio_threshold=3.0, prn_labels=None):
        from engine.estimation.lambda_iar import lambda_search
        result=lambda_search(a_float,Q_aa,ratio_threshold)
        if prn_labels:
            for p in prn_labels: self.iar_results[p]=result
        return result

    def slip_summary(self):
        if self.slip_df is None or self.slip_df.empty: return {}
        return self.slip_df['flag'].value_counts().to_dict()

    def confirmed_slips(self):
        if self.slip_df is None: return None
        return self.slip_df[self.slip_df['flag']=='CONFIRMED'].reset_index(drop=True)