# ═══════════════════════════════════════════════════════════════════════════════
#  GNSS Post-Processing Analysis — run_analysis.py
#  Runs the full pipeline, then opens a rich PyQt5 dashboard with all graphs.
# ═══════════════════════════════════════════════════════════════════════════════

import sys, os

# ── PATH FIX ──────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_candidates = [
    _HERE,
    os.path.join(_HERE, 'gnss_project'),
    os.path.dirname(_HERE),
    os.path.join(os.path.dirname(_HERE), 'gnss_project'),
]
for _candidate in _candidates:
    if os.path.isdir(os.path.join(_candidate, 'engine')):
        if _candidate not in sys.path:
            sys.path.insert(0, _candidate)
        break
else:
    sys.path.insert(0, _HERE)

# ═══════════════════════════════════════════════════════════════════════════════
#  ★  EDIT THESE PATHS  ★
# ═══════════════════════════════════════════════════════════════════════════════
OBS_FILE        = r"F:\KEERTHI KA BIO\IIT Tirupathi Internship\RINEX_Files\IISC00IND_R_20260080000_01D_30S_MO.rnx"
GPS_NAV_FILES   = [r"F:\KEERTHI KA BIO\IIT Tirupathi Internship\RINEX_Files\IISC00IND_R_20260080000_01D_GN.rnx"]
IRNSS_NAV_FILES = [r"F:\KEERTHI KA BIO\IIT Tirupathi Internship\RINEX_Files\IISC00IND_R_20260080000_01D_MN.rnx"]
OUTPUT_DIR      = "output"
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from datetime import datetime

# ── Engine imports — matching your ACTUAL uploaded files ──────────────────────
from engine.io.rinex_obs import read_rinex_obs
from engine.measurements.extract_measurements import (
    extract_measurements, signal_availability_report, pivot_epoch
)
# mw_combination.py: gf_combination, mw_combination, delta_N1_N2, LAMBDA1, LAMBDA2, ...
from engine.cycle_slip.mw_combination import (
    gf_combination, mw_combination, delta_N1_N2,
    LAMBDA1, LAMBDA2, LAMBDA_WL, F1, F2,
)
# mw_detector.py: detect_arc, detect_epoch, DPHI_TOL
from engine.cycle_slip.mw_detector import detect_arc, detect_epoch, DPHI_TOL
# arc_manager.py: ArcManager
from engine.cycle_slip.arc_manager import ArcManager
# slip_corrector.py: correct_arc
from engine.cycle_slip.slip_corrector import correct_arc

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── colour palette ────────────────────────────────────────────────────────────
_LIGHT = {
    'bg': '#F5F7FA',        # window background
    'panel': '#FFFFFF',     # cards / panels
    'border': '#D0D7DE',
    'text': '#1F2328',
    'subtext': '#57606A',

    'accent1': '#0969DA',
    'accent2': '#2DA44E',
    'accent3': '#CF222E',
    'accent4': '#8250DF',
    'accent5': '#BC4C00',
    'accent6': '#1B7F83',
}
_DARK= _LIGHT
_SIG_C = {
    'P1': '#58A6FF', 'P2': '#79C0FF', 'P5': '#A5D6FF',
    'L1': '#3FB950', 'L2': '#56D364', 'L5': '#7EE787',
    'PS': '#FFA657', 'LS': '#FFD700',
    'raw': '#F78166', 'cor': '#3FB950', 'slip': '#FFA657',
    'mw': '#58A6FF', 'mean': '#FFA657',
}
_FLAG_C = {
    'INIT': '#21262D', 'CLEAN': '#1A3A2A',
    'MW_ONLY': '#7C4A00', 'GF_ONLY': '#4A007C',
    'CONFIRMED': '#CC2200', 'GF_SLIP': '#DD4400', 'FAILED': '#880000',
}


def _mpl_light():
    plt.rcParams.update({
        'figure.facecolor': '#F5F7FA',
        'axes.facecolor': '#FFFFFF',
        'axes.edgecolor': '#D0D7DE',
        'axes.labelcolor': '#57606A',
        'axes.titlecolor': '#1F2328',

        'xtick.color': '#57606A',
        'ytick.color': '#57606A',

        'grid.color': '#D0D7DE',
        'grid.linestyle': '--',
        'grid.alpha': 0.6,

        'legend.facecolor': '#FFFFFF',
        'legend.edgecolor': '#D0D7DE',
        'legend.labelcolor': '#1F2328',

        'text.color': '#1F2328',
        'font.size': 10,
    })


_mpl_light()


# ═══════════════════════════════════════════════════════════════════════════════
#  PIPELINE STEPS
# ═══════════════════════════════════════════════════════════════════════════════

def step_load_obs(path):
    print(f"\n{'─'*60}\n  STEP 1 — Loading RINEX OBS\n{'─'*60}")
    obs_df = read_rinex_obs(path)
    print(f"  File       : {os.path.basename(path)}")
    print(f"  Total rows : {len(obs_df)}")
    print(f"  Epochs     : {obs_df['UTC_Time'].nunique()}")
    print(f"  Satellites : {obs_df['Sat'].nunique()}")
    print(f"  Systems    : {sorted(obs_df['Sys'].unique().tolist())}")
    print(f"  Obs types  : {sorted(obs_df['ObsType'].unique().tolist())}")
    return obs_df


def step_extract_measurements(obs_df):
    print(f"\n{'─'*60}\n  STEP 2 — Extracting measurements\n{'─'*60}")
    meas_df = extract_measurements(obs_df)
    report  = signal_availability_report(meas_df)
    print(f"  Epochs : {meas_df['epoch'].nunique()}   Sats : {meas_df['sat'].nunique()}")
    max_v = max(report.values(), default=1)
    for sig, cnt in report.items():
        bar = '█' * int(30 * cnt / max_v)
        print(f"    {sig:4s}  {cnt:6d}  {bar}")
    return meas_df, report


def step_mw_combinations(meas_df):
    """
    Uses gf_combination(L1_m, L2_m) and mw_combination(L1_m, L2_m, P1_m, P2_m)
    from your mw_combination.py.
    """
    print(f"\n{'─'*60}\n  STEP 3 — GF / MW combinations\n{'─'*60}")
    rows = []
    for _, r in meas_df.iterrows():
        L1 = r.get('L1') if 'L1' in r.index else None
        L2 = r.get('L2') if 'L2' in r.index else None
        P1 = r.get('P1') if 'P1' in r.index else None
        P2 = r.get('P2') if 'P2' in r.index else None

        if L1 is None or pd.isna(L1) or L2 is None or pd.isna(L2):
            continue

        L1_m = float(L1) * LAMBDA1
        L2_m = float(L2) * LAMBDA2

        gf_val = float(gf_combination(np.array([L1_m]), np.array([L2_m]))[0])

        mw_val = np.nan
        if P1 is not None and not pd.isna(P1) and P2 is not None and not pd.isna(P2):
            mw_val = float(mw_combination(
                np.array([L1_m]), np.array([L2_m]),
                np.array([float(P1)]), np.array([float(P2)])
            )[0])

        rows.append({
            'epoch': r['epoch'], 'sat': r['sat'], 'sys': r['sys'],
            'gf_m': gf_val, 'mw_cycles': mw_val,
        })

    mw_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if not mw_df.empty:
        gf_s = mw_df.dropna(subset=['gf_m'])
        mw_s = mw_df.dropna(subset=['mw_cycles'])
        if not gf_s.empty:
            print(f"  GF rows : {len(gf_s)}   mean={gf_s['gf_m'].mean():.4f} m   "
                  f"std={gf_s['gf_m'].std():.4f} m")
        if not mw_s.empty:
            print(f"  MW rows : {len(mw_s)}   mean={mw_s['mw_cycles'].mean():.4f} cyc  "
                  f"std={mw_s['mw_cycles'].std():.4f} cyc")
    return mw_df


def step_cycle_slip_detection(meas_df):
    print(f"\n{'─'*60}\n  STEP 4 — Cycle slip detection (GF Dynamic Test)\n{'─'*60}")

    arc_mgr = ArcManager(interval_sec=30.0)
    epochs  = sorted(meas_df['epoch'].unique())

    for idx, ep in enumerate(epochs):
        sub = meas_df[meas_df['epoch'] == ep]
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

    rows = []
    for prn, arcs in arc_mgr._completed.items():
        for arc in arcs:
            if arc.length < 2:
                continue
            L1_arr = arc.L1_array()
            L2_arr = arc.L2_array()
            statuses, infos = detect_arc(L1_arr, L2_arr)

            for i, (status, info) in enumerate(zip(statuses, infos)):
                ep_idx = arc.epochs[i] if i < len(arc.epochs) else i
                rows.append({
                    'sat': prn, 'arc_start': arc.start_epoch,
                    'epoch_idx': ep_idx, 'status': status,
                    'dN1':  info.get('dN1',  np.nan),
                    'dN2':  info.get('dN2',  np.nan),
                    'dphi': info.get('dphi', np.nan),
                    'm1':   info.get('m1',   np.nan),
                    'm2':   info.get('m2',   np.nan),
                })

                # ── FIX: record slip directly into the arc object ──
                if status in ('GF_SLIP', 'GF_ONLY', 'CONFIRMED'):
                    arc.slip_epochs_list.append(i)
                    arc.slip_m1.append(info.get('m1', 1.5))
                    arc.slip_m2.append(info.get('m2', 1.2))
                    arc.slip_history[i] = status

    slip_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if not slip_df.empty:
        for flag, cnt in slip_df['status'].value_counts().items():
            print(f"    {flag:12s} : {cnt}")
    return slip_df, arc_mgr

def step_slip_correction(arc_mgr):
    """Apply correct_arc() to each arc that has recorded slips."""
    print(f"\n{'─'*60}\n  STEP 5 — Slip correction\n{'─'*60}")
    correction_report = {}
    total_ok = total_fail = 0

    for prn, arcs in arc_mgr._completed.items():
        correction_report[prn] = {}
        for arc in arcs:
            if not arc.slip_epochs_list:
                continue
            _, _, report = correct_arc(
                arc.L1_array(), arc.L2_array(),
                arc.slip_epochs_list,
                arc.slip_m1, arc.slip_m2,
            )
            correction_report[prn].update(report)
            for stype in report.values():
                if stype != 'FAILED':
                    total_ok += 1
                else:
                    total_fail += 1

    print(f"  Corrected : {total_ok}")
    print(f"  Failed    : {total_fail}")
    return correction_report


def step_phase_residuals(meas_df, slip_df):
    print(f"\n{'─'*60}\n  STEP 6 — Phase residuals\n{'─'*60}")
    slip_set = {}
    if not slip_df.empty:
        for sat in slip_df['sat'].unique():
            slip_set[sat] = set(
                slip_df[
                    (slip_df['sat'] == sat) &
                    (slip_df['status'].isin(['GF_SLIP', 'CONFIRMED', 'GF_ONLY']))
                ]['epoch_idx'].tolist()
            )

    rows = []
    for sat in meas_df['sat'].unique():
        sub = meas_df[meas_df['sat'] == sat].sort_values('epoch').copy()
        eps = slip_set.get(sat, set())
        for sig in ('L1', 'L2', 'L5', 'LS'):
            if sig not in sub.columns:
                continue
            arc_mean = arc_mean_c = None
            for i, (_, row) in enumerate(sub.iterrows()):
                v = row[sig]
                if pd.isna(v):
                    continue
                v = float(v)
                if arc_mean is None:
                    arc_mean = arc_mean_c = v
                raw_res = v - arc_mean
                if i in eps:
                    arc_mean_c = v
                rows.append({
                    'epoch': row['epoch'], 'epoch_idx': i,
                    'sat': sat, 'signal': sig,
                    'raw_residual': raw_res,
                    'corrected_residual': v - arc_mean_c,
                    'slipped': i in eps,
                })

    res_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if not res_df.empty:
        for sig in ('L1', 'L2', 'L5', 'LS'):
            sub = res_df[res_df['signal'] == sig]
            if not sub.empty:
                print(f"  {sig}  raw_std={sub['raw_residual'].std():.4f}  "
                      f"cor_std={sub['corrected_residual'].std():.4f}  "
                      f"slip_epochs={int(sub['slipped'].sum())}")
    return res_df


# ═══════════════════════════════════════════════════════════════════════════════
#  GUI DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
#  GUI DASHBOARD  — fixed layout: scroll areas, proper chart heights, nav buttons
# ═══════════════════════════════════════════════════════════════════════════════

def launch_dashboard(obs_df, meas_df, sig_report, mw_df,
                     slip_df, arc_mgr, correction_report, res_df):
    matplotlib.use('Qt5Agg')
    _mpl_light()

    from PyQt5.QtWidgets import (
        QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout,
        QTabWidget, QLabel, QFrame, QScrollArea, QHeaderView,
        QTableWidget, QTableWidgetItem, QPushButton, QSizePolicy,
        QSplitter, QSpacerItem,
    )
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QColor, QPalette
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar

    # ══════════════════════════════════════════════════════════════════════════
    #  SHARED HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def make_fig(*args, **kw):
        return plt.Figure(*args, facecolor=_DARK['bg'], **kw)

    def ax_style(ax, title='', xlabel='', ylabel=''):
        ax.set_facecolor(_DARK['panel'])
        for sp in ax.spines.values(): sp.set_color(_DARK['border'])
        ax.tick_params(colors=_DARK['subtext'])
        ax.xaxis.label.set_color(_DARK['subtext'])
        ax.yaxis.label.set_color(_DARK['subtext'])
        ax.title.set_color(_DARK['text'])
        if title:  ax.set_title(title, fontsize=11, pad=8)
        if xlabel: ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)
        ax.grid(True, color=_DARK['border'], linestyle='--', alpha=0.5, lw=0.6)

    def stat_card(label, value, color):
        f = QFrame()
        f.setStyleSheet(f"QFrame{{background:{_DARK['panel']};border:1px solid "
                        f"{_DARK['border']};border-radius:8px;padding:8px;}}")
        f.setFixedWidth(160)
        vb = QVBoxLayout(f); vb.setSpacing(2)
        lb = QLabel(str(label))
        lb.setStyleSheet(f"color:{_DARK['subtext']};font-size:11px;border:none;background:transparent;")
        vl = QLabel(str(value))
        vl.setStyleSheet(f"color:{color};font-size:22px;font-weight:700;border:none;background:transparent;")
        vb.addWidget(lb); vb.addWidget(vl)
        return f

    def section_label(text):
        lb = QLabel(text)
        lb.setStyleSheet(f"color:{_DARK['accent1']};font-size:13px;font-weight:700;"
                         f"border-bottom:1px solid {_DARK['border']};padding-bottom:4px;"
                         f"margin-top:6px;")
        return lb

    def nav_btn(text, color=None):
        c = color or _DARK['accent1']
        b = QPushButton(text)
        b.setStyleSheet(
            f"QPushButton{{background:{_DARK['panel']};color:{c};"
            f"border:1px solid {c};border-radius:5px;"
            f"padding:5px 18px;font-size:12px;font-weight:600;}}"
            f"QPushButton:hover{{background:{c}22;}}"
            f"QPushButton:disabled{{color:{_DARK['border']};border-color:{_DARK['border']};}}"
        )
        return b

    TAB_STYLE = (
        f"QTabWidget::pane{{border:1px solid {_DARK['border']};background:{_DARK['bg']};}}"
        f"QTabBar::tab{{background:{_DARK['panel']};color:{_DARK['subtext']};"
        f"padding:6px 16px;border:1px solid {_DARK['border']};}}"
        f"QTabBar::tab:selected{{background:{_DARK['border']};color:{_DARK['text']};}}"
    )

    def make_table(headers, data, max_h=220):
        tbl = QTableWidget()
        tbl.setStyleSheet(
            f"QTableWidget{{background:{_DARK['panel']};color:{_DARK['text']};"
            f"gridline-color:{_DARK['border']};border:none;font-size:11px;}}"
            f"QHeaderView::section{{background:{_DARK['border']};color:{_DARK['text']};"
            f"padding:5px;border:none;font-weight:700;}}"
            f"QTableWidget::item:selected{{background:{_DARK['accent1']};color:{_DARK['bg']};}}"
            f"QTableWidget::item:alternate{{background:{_DARK['bg']};}}"
        )
        tbl.setAlternatingRowColors(True)
        tbl.setColumnCount(len(headers)); tbl.setRowCount(len(data))
        tbl.setHorizontalHeaderLabels(headers)
        tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tbl.verticalHeader().setVisible(False)
        tbl.setMaximumHeight(max_h)
        for r, row in enumerate(data):
            for c, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignCenter)
                tbl.setItem(r, c, item)
        return tbl

    def scrolled(inner_widget):
        """Wrap any widget in a dark-styled QScrollArea."""
        sa = QScrollArea()
        sa.setWidget(inner_widget)
        sa.setWidgetResizable(True)
        sa.setStyleSheet(
            f"QScrollArea{{background:{_DARK['bg']};border:none;}}"
            f"QScrollBar:vertical{{background:{_DARK['panel']};width:10px;border-radius:5px;}}"
            f"QScrollBar::handle:vertical{{background:{_DARK['border']};border-radius:5px;min-height:20px;}}"
            f"QScrollBar:horizontal{{background:{_DARK['panel']};height:10px;border-radius:5px;}}"
            f"QScrollBar::handle:horizontal{{background:{_DARK['border']};border-radius:5px;min-width:20px;}}"
        )
        return sa

    def chart_card(fig, height_px=420, toolbar=True):
        """
        Wraps a Figure into a fixed-height card widget with optional toolbar.
        Charts are sized to always fully render — no clipping.
        """
        canvas = FigureCanvas(fig)
        canvas.setStyleSheet(f"background:{_DARK['bg']};")
        card = QWidget()
        card.setStyleSheet(
            f"background:{_DARK['panel']};border:1px solid {_DARK['border']};"
            f"border-radius:6px;"
        )
        vb = QVBoxLayout(card); vb.setContentsMargins(4, 4, 4, 4); vb.setSpacing(0)
        if toolbar:
            tb = NavToolbar(canvas, card)
            tb.setStyleSheet(
                f"background:{_DARK['panel']};color:{_DARK['text']};"
                f"border:none;border-bottom:1px solid {_DARK['border']};"
            )
            vb.addWidget(tb)
        canvas.setMinimumHeight(height_px)
        vb.addWidget(canvas)
        return card

    def stacked_charts(figures_and_heights, parent_vb, toolbar=True):
        """
        Add multiple chart_cards stacked in parent_vb.
        figures_and_heights: list of (fig, height_px) tuples.
        """
        for fig, h in figures_and_heights:
            parent_vb.addWidget(chart_card(fig, height_px=h, toolbar=toolbar))

    # ══════════════════════════════════════════════════════════════════════════
    #  SLIDE NAVIGATOR  — used for tabs with many per-satellite charts
    # ══════════════════════════════════════════════════════════════════════════

    class SlideNavigator(QWidget):
        """
        Shows one chart at a time with ◀ Prev / Next ▶ buttons.
        charts: list of (title_str, fig, height_px)
        """
        def __init__(self, charts, parent=None):
            super().__init__(parent)
            self._charts = charts   # [(title, fig, height)]
            self._idx    = 0
            self._layout = QVBoxLayout(self)
            self._layout.setContentsMargins(0, 0, 0, 0)
            self._layout.setSpacing(6)

            # Header row
            hdr = QHBoxLayout()
            self._btn_prev = nav_btn("◀  Prev")
            self._btn_next = nav_btn("Next  ▶")
            self._lbl_pos  = QLabel()
            self._lbl_pos.setStyleSheet(f"color:{_DARK['subtext']};font-size:11px;")
            self._lbl_pos.setAlignment(Qt.AlignCenter)
            self._lbl_title = QLabel()
            self._lbl_title.setStyleSheet(
                f"color:{_DARK['text']};font-size:13px;font-weight:700;")
            self._lbl_title.setAlignment(Qt.AlignCenter)
            hdr.addWidget(self._btn_prev)
            hdr.addStretch()
            hdr.addWidget(self._lbl_title)
            hdr.addStretch()
            hdr.addWidget(self._lbl_pos)
            hdr.addWidget(self._btn_next)
            self._layout.addLayout(hdr)

            # Chart area (replaced on navigation)
            self._chart_holder = QVBoxLayout()
            self._layout.addLayout(self._chart_holder)

            self._btn_prev.clicked.connect(self._go_prev)
            self._btn_next.clicked.connect(self._go_next)
            self._render()

        def _clear(self):
            while self._chart_holder.count():
                item = self._chart_holder.takeAt(0)
                if item.widget():
                    item.widget().setParent(None)

        def _render(self):
            self._clear()
            n = len(self._charts)
            if n == 0: return
            title, fig, h = self._charts[self._idx]
            self._lbl_title.setText(title)
            self._lbl_pos.setText(f"{self._idx+1} / {n}")
            self._chart_holder.addWidget(chart_card(fig, height_px=h, toolbar=True))
            self._btn_prev.setEnabled(self._idx > 0)
            self._btn_next.setEnabled(self._idx < n - 1)

        def _go_prev(self):
            if self._idx > 0: self._idx -= 1; self._render()

        def _go_next(self):
            if self._idx < len(self._charts) - 1: self._idx += 1; self._render()

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 1 — Overview
    # ══════════════════════════════════════════════════════════════════════════

    def build_overview(parent):
        inner = QWidget()
        inner.setStyleSheet(f"background:{_DARK['bg']};")
        vb = QVBoxLayout(inner)
        vb.setContentsMargins(16, 16, 16, 16); vb.setSpacing(12)

        # Stat cards row
        n_slips = int((slip_df['status'].isin(['GF_SLIP','CONFIRMED','GF_ONLY'])
                       ).sum()) if not slip_df.empty else 0
        n_corr  = sum(1 for rep in correction_report.values()
                      for s in rep.values() if s != 'FAILED')

        cards_row = QHBoxLayout()
        cards_row.setSpacing(8)
        for lbl, val, col in [
            ('Epochs',          obs_df['UTC_Time'].nunique(), _DARK['accent1']),
            ('Total Sats',      obs_df['Sat'].nunique(),      _DARK['accent2']),
            ('GPS Sats',        obs_df[obs_df['Sys']=='G']['Sat'].nunique(), _DARK['accent6']),
            ('IRNSS Sats',      obs_df[obs_df['Sys']=='I']['Sat'].nunique(), _DARK['accent5']),
            ('Slips Detected',  n_slips,    _DARK['accent3']),
            ('Slips Corrected', n_corr,     _DARK['accent2']),
        ]:
            cards_row.addWidget(stat_card(lbl, val, col))
        cards_row.addStretch()
        vb.addLayout(cards_row)

        # Signal availability bar chart
        vb.addWidget(section_label("Signal Availability"))
        sigs   = [s for s, c in sig_report.items() if c > 0]
        counts = [sig_report[s] for s in sigs]
        colors_bar = [_SIG_C.get(s, '#888') for s in sigs]
        fig1 = make_fig(figsize=(10, 3.2), tight_layout=True)
        ax1  = fig1.add_subplot(111)
        bars = ax1.bar(sigs, counts, color=colors_bar, width=0.55, zorder=3)
        ax1.bar_label(bars, fmt='%d', color=_DARK['text'], fontsize=9, padding=3)
        ax_style(ax1, 'Observations per Signal', 'Signal', 'Count')
        vb.addWidget(chart_card(fig1, height_px=320, toolbar=False))

        # System pie + obs-type bar
        vb.addWidget(section_label("Constellation & Observation Type Breakdown"))
        sys_counts = obs_df.groupby('Sys')['Sat'].nunique()
        top_obs    = obs_df.groupby('ObsType').size().nlargest(14)
        fig2 = make_fig(figsize=(14, 3.5), tight_layout=True)
        gs   = gridspec.GridSpec(1, 2, figure=fig2, wspace=0.35)
        ax_pie = fig2.add_subplot(gs[0, 0])
        pie_c  = [_DARK['accent1'],_DARK['accent2'],_DARK['accent5'],
                  _DARK['accent4'],_DARK['accent6'],_DARK['accent3']]
        wedges, texts, auto = ax_pie.pie(
            sys_counts.values, labels=sys_counts.index,
            autopct='%1.0f%%', colors=pie_c[:len(sys_counts)],
            textprops={'color': _DARK['text']},
            wedgeprops={'linewidth': 1, 'edgecolor': _DARK['bg']}
        )
        for at in auto: at.set_color(_DARK['bg']); at.set_fontsize(9)
        ax_pie.set_title('Satellite Systems', color=_DARK['text'], fontsize=11)
        ax_pie.set_facecolor(_DARK['panel'])
        ax_obs = fig2.add_subplot(gs[0, 1])
        obs_colors = [_DARK['accent1'] if o[0]=='C' else
                      _DARK['accent2'] if o[0]=='L' else _DARK['accent5']
                      for o in top_obs.index]
        ax_obs.barh(top_obs.index, top_obs.values, color=obs_colors, height=0.65)
        ax_style(ax_obs, 'Top Observation Types', 'Count', '')
        ax_obs.invert_yaxis()
        vb.addWidget(chart_card(fig2, height_px=360, toolbar=True))

        vb.addItem(QSpacerItem(0, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        return scrolled(inner)

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 2 — GF / MW
    # ══════════════════════════════════════════════════════════════════════════

    def build_mw_tab(parent):
        inner = QWidget()
        inner.setStyleSheet(f"background:{_DARK['bg']};")
        vb = QVBoxLayout(inner)
        vb.setContentsMargins(16, 16, 16, 16); vb.setSpacing(10)

        if mw_df is None or mw_df.empty:
            vb.addWidget(QLabel("No GF/MW combinations — check L1/L2/P1/P2 availability."))
            return scrolled(inner)

        # ── GF section ────────────────────────────────────────────────────────
        gf_sub = mw_df.dropna(subset=['gf_m'])
        if not gf_sub.empty:
            vb.addWidget(section_label("Geometry-Free (GF) Combination  —  L1 − L2"))
            top_sats = gf_sub.groupby('sat').size().nlargest(8).index.tolist()

            # Time series + histogram side by side
            fig_gf = make_fig(figsize=(14, 4.0), tight_layout=True)
            gs1 = gridspec.GridSpec(1, 2, figure=fig_gf, wspace=0.28)
            ax_ts  = fig_gf.add_subplot(gs1[0, 0])
            ax_his = fig_gf.add_subplot(gs1[0, 1])
            cmap = plt.cm.get_cmap('cool', len(top_sats))
            for i, sat in enumerate(top_sats):
                s = gf_sub[gf_sub['sat'] == sat].sort_values('epoch')
                ax_ts.plot(range(len(s)), s['gf_m'].values,
                           lw=1.0, alpha=0.8, color=cmap(i), label=sat)
            mean_gf = gf_sub['gf_m'].mean(); std_gf = gf_sub['gf_m'].std()
            ax_ts.axhline(mean_gf, color=_DARK['accent5'], lw=1.0, ls='--',
                          label=f'Mean {mean_gf:.3f}')
            ax_ts.axhline(mean_gf + 3*std_gf, color=_DARK['accent3'], lw=0.7, ls=':')
            ax_ts.axhline(mean_gf - 3*std_gf, color=_DARK['accent3'], lw=0.7, ls=':', label='±3σ')
            ax_style(ax_ts, 'GF Time Series (top 8 sats)', 'Epoch', 'GF (m)')
            ax_ts.legend(fontsize=8, ncol=4)
            ax_his.hist(gf_sub['gf_m'].dropna(), bins=60,
                        color=_DARK['accent1'], edgecolor=_DARK['bg'], lw=0.4)
            ax_style(ax_his, 'GF Distribution', 'GF (m)', 'Count')
            vb.addWidget(chart_card(fig_gf, height_px=380, toolbar=True))

            # Per-satellite GF slide navigator
            vb.addWidget(section_label("GF per Satellite  —  use ◀ ▶ to navigate"))
            gf_charts = []
            for sat in gf_sub.groupby('sat').size().nlargest(20).index.tolist():
                s    = gf_sub[gf_sub['sat'] == sat].sort_values('epoch')
                x    = np.arange(len(s)); vals = s['gf_m'].values
                mean_v = np.nanmean(vals); res = vals - mean_v
                fig_s = make_fig(figsize=(13, 3.0), tight_layout=True)
                ax_s  = fig_s.add_subplot(111)
                ax_s.fill_between(x, res, 0, where=(res >= 0),
                                  color=_DARK['accent1'], alpha=0.45)
                ax_s.fill_between(x, res, 0, where=(res < 0),
                                  color=_DARK['accent3'], alpha=0.45)
                ax_s.plot(x, res, color=_DARK['accent1'], lw=0.9, alpha=0.9)
                ax_s.axhline(0, color=_DARK['border'], lw=0.8)
                if not slip_df.empty:
                    sf = slip_df[(slip_df['sat'] == sat) &
                                 (slip_df['status'].isin(['GF_SLIP','CONFIRMED','GF_ONLY']))]
                    for ep in sf['epoch_idx'].values:
                        ax_s.axvline(ep, color=_DARK['accent3'], lw=1.2, ls='--', alpha=0.7)
                ax_style(ax_s, f'{sat}  —  GF residual from arc mean', 'Epoch', 'GF residual (m)')
                gf_charts.append((sat, fig_s, 320))
            vb.addWidget(SlideNavigator(gf_charts))

        # ── MW section ────────────────────────────────────────────────────────
        mw_sub = mw_df.dropna(subset=['mw_cycles'])
        if not mw_sub.empty:
            vb.addWidget(section_label("Melbourne-Wübbena (MW) Wide-lane Combination"))
            top_sats2 = mw_sub.groupby('sat').size().nlargest(8).index.tolist()
            fig_mw = make_fig(figsize=(14, 4.0), tight_layout=True)
            gs2 = gridspec.GridSpec(1, 2, figure=fig_mw, wspace=0.28)
            ax2  = fig_mw.add_subplot(gs2[0, 0])
            ah2  = fig_mw.add_subplot(gs2[0, 1])
            cmap2 = plt.cm.get_cmap('plasma', len(top_sats2))
            for i, sat in enumerate(top_sats2):
                s = mw_sub[mw_sub['sat'] == sat].sort_values('epoch')
                ax2.plot(range(len(s)), s['mw_cycles'].values,
                         lw=1.0, alpha=0.8, color=cmap2(i), label=sat)
            mean_mw = mw_sub['mw_cycles'].mean(); std_mw = mw_sub['mw_cycles'].std()
            ax2.axhline(mean_mw, color=_DARK['accent5'], lw=1.0, ls='--',
                        label=f'Mean {mean_mw:.3f}')
            ax2.axhline(mean_mw + 3*std_mw, color=_DARK['accent3'], lw=0.7, ls=':')
            ax2.axhline(mean_mw - 3*std_mw, color=_DARK['accent3'], lw=0.7, ls=':', label='±3σ')
            ax_style(ax2, 'MW Wide-lane Time Series (top 8 sats)', 'Epoch', 'MW (cyc)')
            ax2.legend(fontsize=8, ncol=4)
            ah2.hist(mw_sub['mw_cycles'].dropna(), bins=60,
                     color=_DARK['accent4'], edgecolor=_DARK['bg'], lw=0.4)
            ax_style(ah2, 'MW Distribution', 'MW (cyc)', 'Count')
            vb.addWidget(chart_card(fig_mw, height_px=380, toolbar=True))

        # Summary table
        rows_data = []
        if not gf_sub.empty:
            rows_data.append(['GF (L1−L2)', len(gf_sub),
                              f"{gf_sub['gf_m'].mean():.4f}",
                              f"{gf_sub['gf_m'].std():.4f}",
                              f"{gf_sub['gf_m'].min():.4f}",
                              f"{gf_sub['gf_m'].max():.4f}"])
        if not mw_sub.empty:
            rows_data.append(['MW WL', len(mw_sub),
                              f"{mw_sub['mw_cycles'].mean():.4f}",
                              f"{mw_sub['mw_cycles'].std():.4f}",
                              f"{mw_sub['mw_cycles'].min():.4f}",
                              f"{mw_sub['mw_cycles'].max():.4f}"])
        if rows_data:
            vb.addWidget(section_label("Combination Statistics"))
            vb.addWidget(make_table(['Combination','N','Mean','Std','Min','Max'],
                                    rows_data, max_h=120))

        vb.addItem(QSpacerItem(0, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        return scrolled(inner)

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 3 — Cycle Slip Detection
    # ══════════════════════════════════════════════════════════════════════════

    def build_slip_tab(parent):
        inner = QWidget()
        inner.setStyleSheet(f"background:{_DARK['bg']};")
        vb = QVBoxLayout(inner)
        vb.setContentsMargins(16, 16, 16, 16); vb.setSpacing(10)

        if slip_df.empty:
            vb.addWidget(QLabel("No slip detection results.")); return scrolled(inner)

        # Stat cards
        counts = slip_df['status'].value_counts()
        fc = {'CLEAN': _DARK['accent2'], 'INIT': _DARK['subtext'],
              'GF_SLIP': _DARK['accent3'], 'CONFIRMED': _DARK['accent3'],
              'GF_ONLY': _DARK['accent4'], 'FAILED': '#880000'}
        cards_row = QHBoxLayout(); cards_row.setSpacing(8)
        for flag, cnt in counts.items():
            cards_row.addWidget(stat_card(flag, cnt, fc.get(flag, _DARK['text'])))
        cards_row.addStretch()
        vb.addLayout(cards_row)

        # Detection timeline (full width, tall enough for all sats)
        vb.addWidget(section_label("Detection Timeline  —  all satellites × epochs"))
        sats    = sorted(slip_df['sat'].unique())
        sat_idx = {s: i for i, s in enumerate(sats)}
        n_sats  = len(sats)
        fig_tl  = make_fig(figsize=(14, max(6, n_sats * 0.30)), tight_layout=True)
        ax_tl   = fig_tl.add_subplot(111)
        for _, row in slip_df.iterrows():
            ax_tl.scatter(row['epoch_idx'], sat_idx[row['sat']],
                          c=_FLAG_C.get(row['status'], '#333'),
                          s=5, marker='s', linewidths=0)
        ax_tl.set_yticks(range(n_sats))
        ax_tl.set_yticklabels(sats, fontsize=7)
        handles = [Patch(facecolor=c, label=f, edgecolor=_DARK['border'])
                   for f, c in _FLAG_C.items() if f in counts.index]
        ax_tl.legend(handles=handles, loc='upper right', fontsize=8, ncol=2)
        ax_style(ax_tl, 'Cycle Slip Detection Timeline', 'Epoch index', 'Satellite')
        ax_tl.yaxis.grid(False)
        tl_h = max(380, n_sats * 18)
        vb.addWidget(chart_card(fig_tl, height_px=tl_h, toolbar=True))

        # Δφ per-satellite slide navigator
        slip_sats = (slip_df[slip_df['status'].isin(['GF_SLIP','CONFIRMED','GF_ONLY'])]
                     ['sat'].value_counts().head(20).index.tolist())
        if slip_sats:
            vb.addWidget(section_label("Δφ Series per Satellite  —  use ◀ ▶ to navigate"))
            dphi_charts = []
            for sat in slip_sats:
                sub = slip_df[slip_df['sat'] == sat].sort_values('epoch_idx')
                x   = sub['epoch_idx'].values
                phi = sub['dphi'].values
                fig_d = make_fig(figsize=(13, 3.0), tight_layout=True)
                ax_d  = fig_d.add_subplot(111)
                ax_d.plot(x, phi, color=_DARK['accent1'], lw=0.9, alpha=0.9)
                ax_d.axhline( DPHI_TOL, color=_DARK['accent3'], lw=0.9, ls='--',
                              label=f'+tol ({DPHI_TOL})')
                ax_d.axhline(-DPHI_TOL, color=_DARK['accent3'], lw=0.9, ls='--',
                              label=f'−tol ({DPHI_TOL})')
                slip_mask = sub['status'].isin(['GF_SLIP','CONFIRMED','GF_ONLY']).values
                if slip_mask.any():
                    ax_d.scatter(x[slip_mask], phi[slip_mask],
                                 color=_DARK['accent3'], s=50, zorder=5, marker='^',
                                 label='Slip')
                ax_style(ax_d, f'{sat}  —  Δφ (epoch-to-epoch GF)', 'Epoch', 'Δφ (cyc)')
                ax_d.legend(fontsize=8)
                dphi_charts.append((sat, fig_d, 320))
            vb.addWidget(SlideNavigator(dphi_charts))

        # Detected slips table
        slip_rows = slip_df[slip_df['status'].isin(
            ['GF_SLIP','CONFIRMED','GF_ONLY'])].head(300)
        if not slip_rows.empty:
            vb.addWidget(section_label(f"Detected Slip Events  ({len(slip_rows)} shown)"))
            cols = ['sat','epoch_idx','status','dN1','dN2','dphi','m1','m2']
            cols = [c for c in cols if c in slip_rows.columns]
            data = [[f"{v:.4f}" if isinstance(v, float) else str(v)
                     for v in row[cols]] for _, row in slip_rows.iterrows()]
            vb.addWidget(make_table(cols, data, max_h=240))

        vb.addItem(QSpacerItem(0, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        return scrolled(inner)

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 4 — Correction
    # ══════════════════════════════════════════════════════════════════════════

    def build_correction_tab(parent):
        inner = QWidget()
        inner.setStyleSheet(f"background:{_DARK['bg']};")
        vb = QVBoxLayout(inner)
        vb.setContentsMargins(16, 16, 16, 16); vb.setSpacing(10)

        all_types = {}
        rows_data = []
        for prn, rep in correction_report.items():
            for ep, stype in rep.items():
                rows_data.append([prn, ep, stype])
                all_types[stype] = all_types.get(stype, 0) + 1

        if not rows_data:
            vb.addWidget(QLabel("No slips corrected.")); return scrolled(inner)

        tc = {'NORMAL': _DARK['accent2'], '(9,7)': _DARK['accent4'],
              '(77,60)': _DARK['accent5'], 'FAILED': _DARK['accent3']}
        cards_row = QHBoxLayout(); cards_row.setSpacing(8)
        for stype, cnt in all_types.items():
            cards_row.addWidget(stat_card(stype, cnt, tc.get(stype, _DARK['text'])))
        cards_row.addStretch()
        vb.addLayout(cards_row)

        # Per-satellite correction type bar chart
        prns   = list(correction_report.keys())
        stypes = list(set(v for rep in correction_report.values() for v in rep.values()))
        if prns and stypes:
            vb.addWidget(section_label("Correction Types per Satellite"))
            data_by = {st: [sum(1 for v in correction_report[p].values() if v == st)
                            for p in prns] for st in stypes}
            fig_c = make_fig(figsize=(13, 3.8), tight_layout=True)
            ax_c  = fig_c.add_subplot(111)
            x     = np.arange(len(prns))
            wb    = 0.8 / max(len(stypes), 1)
            for i, st in enumerate(stypes):
                ax_c.bar(x + i*wb, data_by[st], width=wb, label=st,
                         color=tc.get(st, _DARK['subtext']), zorder=3)
            ax_c.set_xticks(x + wb*(len(stypes)-1)/2)
            ax_c.set_xticklabels(prns, rotation=45, ha='right', fontsize=8)
            ax_style(ax_c, 'Correction Types per Satellite', 'Satellite', 'Count')
            ax_c.legend(fontsize=9)
            vb.addWidget(chart_card(fig_c, height_px=380, toolbar=True))

        vb.addWidget(section_label(f"Correction Log  ({min(300,len(rows_data))} rows shown)"))
        vb.addWidget(make_table(['Satellite','Epoch idx','Correction Type'],
                                rows_data[:300], max_h=280))

        vb.addItem(QSpacerItem(0, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        return scrolled(inner)

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 5 — Phase Residuals
    # ══════════════════════════════════════════════════════════════════════════

    def build_residuals_tab(parent):
        inner = QWidget()
        inner.setStyleSheet(f"background:{_DARK['bg']};")
        vb = QVBoxLayout(inner)
        vb.setContentsMargins(16, 16, 16, 16); vb.setSpacing(10)

        if res_df.empty:
            vb.addWidget(QLabel("No phase residuals computed.")); return scrolled(inner)

        # Inner tab per signal, slide navigator per satellite inside each
        inner_tabs = QTabWidget(); inner_tabs.setStyleSheet(TAB_STYLE)
        inner_tabs.setMinimumHeight(700)

        for sig in ('L1', 'L2', 'L5', 'LS'):
            sub_sig = res_df[res_df['signal'] == sig]
            if sub_sig.empty: continue

            sig_widget = QWidget()
            sig_widget.setStyleSheet(f"background:{_DARK['bg']};")
            sig_vb = QVBoxLayout(sig_widget)
            sig_vb.setContentsMargins(8, 8, 8, 8); sig_vb.setSpacing(8)

            top_sats = sub_sig.groupby('sat').size().nlargest(20).index.tolist()
            charts   = []
            for sat in top_sats:
                sub = sub_sig[sub_sig['sat'] == sat].sort_values('epoch_idx')
                if sub.empty: continue
                idx = sub['epoch_idx'].values

                fig_r = make_fig(figsize=(13, 3.4), tight_layout=True)
                gs_r  = gridspec.GridSpec(1, 2, figure=fig_r, wspace=0.28)
                ax_ts  = fig_r.add_subplot(gs_r[0, 0])
                ax_his = fig_r.add_subplot(gs_r[0, 1])

                ax_ts.plot(idx, sub['raw_residual'].values,
                           color=_DARK['accent3'], lw=0.9, alpha=0.7, label='Raw')
                ax_ts.plot(idx, sub['corrected_residual'].values,
                           color=_DARK['accent2'], lw=1.2, label='Corrected')
                si = sub[sub['slipped']]['epoch_idx'].values
                if len(si):
                    ax_ts.scatter(si, sub[sub['slipped']]['raw_residual'].values,
                                  color=_DARK['accent5'], s=50, zorder=5,
                                  marker='^', label='Slip')
                ax_ts.axhline(0, color=_DARK['border'], lw=0.8, ls='--')
                ax_style(ax_ts, f'{sat}  {sig}  —  Time series', 'Epoch', 'Residual (cyc)')
                ax_ts.legend(fontsize=8)

                raw_vals = sub['raw_residual'].dropna()
                cor_vals = sub['corrected_residual'].dropna()
                if len(raw_vals) > 1:
                    ax_his.hist(raw_vals, bins=40, color=_DARK['accent3'],
                                alpha=0.6, label='Raw', density=True)
                if len(cor_vals) > 1:
                    ax_his.hist(cor_vals, bins=40, color=_DARK['accent2'],
                                alpha=0.6, label='Corrected', density=True)
                ax_style(ax_his, f'{sat} {sig}  —  Distribution', 'Residual (cyc)', 'Density')
                ax_his.legend(fontsize=8)

                charts.append((f"{sat}  [{sig}]", fig_r, 360))

            if charts:
                sig_vb.addWidget(SlideNavigator(charts))

            # Summary stats table for this signal
            rows_s = []
            for sat in top_sats:
                sub = sub_sig[sub_sig['sat'] == sat]
                if sub.empty: continue
                rows_s.append([sat,
                               f"{sub['raw_residual'].std():.4f}",
                               f"{sub['corrected_residual'].std():.4f}",
                               int(sub['slipped'].sum())])
            if rows_s:
                sig_vb.addWidget(section_label(f"{sig}  —  Per-satellite residual statistics"))
                sig_vb.addWidget(make_table(
                    ['Sat', 'Raw Std (cyc)', 'Corr Std (cyc)', 'Slip Epochs'],
                    rows_s, max_h=220))

            inner_tabs.addTab(scrolled(sig_widget), f"  {sig}  ")

        vb.addWidget(inner_tabs)
        vb.addItem(QSpacerItem(0, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        return scrolled(inner)

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 6 — Arc Analysis
    # ══════════════════════════════════════════════════════════════════════════

    def build_arc_tab(parent):
        inner = QWidget()
        inner.setStyleSheet(f"background:{_DARK['bg']};")
        vb = QVBoxLayout(inner)
        vb.setContentsMargins(16, 16, 16, 16); vb.setSpacing(10)

        gf_sub = (mw_df.dropna(subset=['gf_m'])
                  if (mw_df is not None and not mw_df.empty) else pd.DataFrame())
        if gf_sub.empty:
            vb.addWidget(QLabel("No GF data for arc analysis.")); return scrolled(inner)

        top_sats = gf_sub.groupby('sat').size().nlargest(20).index.tolist()
        vb.addWidget(section_label("GF Arc Residuals per Satellite  —  use ◀ ▶ to navigate"))

        arc_charts = []
        for sat in top_sats:
            s      = gf_sub[gf_sub['sat'] == sat].sort_values('epoch')
            x      = np.arange(len(s)); vals = s['gf_m'].values
            mean_v = np.nanmean(vals); res = vals - mean_v
            fig_a  = make_fig(figsize=(13, 3.0), tight_layout=True)
            ax_a   = fig_a.add_subplot(111)
            ax_a.fill_between(x, res, 0, where=(res >= 0),
                              color=_DARK['accent1'], alpha=0.5)
            ax_a.fill_between(x, res, 0, where=(res < 0),
                              color=_DARK['accent3'], alpha=0.5)
            ax_a.plot(x, res, color=_DARK['accent1'], lw=0.8, alpha=0.9)
            ax_a.axhline(0, color=_DARK['border'], lw=0.8)
            if not slip_df.empty:
                sf = slip_df[(slip_df['sat'] == sat) &
                             (slip_df['status'].isin(['GF_SLIP','CONFIRMED','GF_ONLY']))]
                for ep in sf['epoch_idx'].values:
                    ax_a.axvline(ep, color=_DARK['accent3'], lw=1.2, ls='--', alpha=0.7)
            ax_style(ax_a, f'{sat}  —  GF residual from arc mean', 'Epoch', 'Residual (m)')
            arc_charts.append((sat, fig_a, 320))

        vb.addWidget(SlideNavigator(arc_charts))
        vb.addItem(QSpacerItem(0, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        return scrolled(inner)

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 7 — Statistics
    # ══════════════════════════════════════════════════════════════════════════

    def build_stats_tab(parent):
        # ── card helpers local to this tab ─────────────────────────────────────
        def full_table(headers, data):
            """Table auto-sized to show ALL rows — no internal scrollbar."""
            tbl = QTableWidget()
            tbl.setStyleSheet(
                f"QTableWidget{{background:{_DARK['panel']};color:{_DARK['text']};"
                f"gridline-color:{_DARK['border']};border:none;font-size:11px;}}"
                f"QHeaderView::section{{background:{_DARK['border']};color:{_DARK['text']};"
                f"padding:5px;border:none;font-weight:700;font-size:11px;}}"
                f"QTableWidget::item:selected{{background:{_DARK['accent1']};"
                f"color:{_DARK['bg']};}}"
                f"QTableWidget::item:alternate{{background:{_DARK['bg']};}}"
            )
            tbl.setAlternatingRowColors(True)
            tbl.setColumnCount(len(headers))
            tbl.setRowCount(len(data))
            tbl.setHorizontalHeaderLabels(headers)
            tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            tbl.verticalHeader().setVisible(False)
            tbl.setEditTriggers(QTableWidget.NoEditTriggers)
            tbl.setSelectionBehavior(QTableWidget.SelectRows)
            for r, row in enumerate(data):
                for c, val in enumerate(row):
                    item = QTableWidgetItem(str(val))
                    item.setTextAlignment(Qt.AlignCenter)
                    tbl.setItem(r, c, item)
            # exact height so every row shows without clipping
            tbl.setFixedHeight(36 + len(data) * 28 + 4)
            return tbl

        def card(title, accent, headers, data):
            """Bordered card: coloured dot + title + row-count badge + full table."""
            f = QFrame()
            f.setStyleSheet(
                f"QFrame{{background:{_DARK['panel']};"
                f"border:1px solid {_DARK['border']};border-radius:8px;}}"
            )
            vb = QVBoxLayout(f)
            vb.setContentsMargins(12, 10, 12, 12)
            vb.setSpacing(8)

            # title row
            hdr_row = QHBoxLayout()
            dot = QLabel("●")
            dot.setStyleSheet(
                f"color:{accent};font-size:10px;border:none;background:transparent;")
            lbl = QLabel(title)
            lbl.setStyleSheet(
                f"color:{_DARK['text']};font-size:13px;font-weight:700;"
                f"border:none;background:transparent;")
            badge = QLabel(f"{len(data)} rows")
            badge.setStyleSheet(
                f"color:{_DARK['subtext']};font-size:10px;"
                f"border:none;background:transparent;")
            hdr_row.addWidget(dot)
            hdr_row.addWidget(lbl)
            hdr_row.addStretch()
            hdr_row.addWidget(badge)
            vb.addLayout(hdr_row)

            # thin divider
            div = QFrame()
            div.setFrameShape(QFrame.HLine)
            div.setFixedHeight(1)
            div.setStyleSheet(
                f"background:{_DARK['border']};border:none;max-height:1px;")
            vb.addWidget(div)

            vb.addWidget(full_table(headers, data))
            return f

        # ── collect all sections ────────────────────────────────────────────────
        sections = []

        # 1 — Observation File
        sections.append(('Observation File', _DARK['accent1'],
            ['Metric', 'Value'],
            [['Total rows',  len(obs_df)],
             ['Epochs',      obs_df['UTC_Time'].nunique()],
             ['Satellites',  obs_df['Sat'].nunique()],
             ['GPS sats',    obs_df[obs_df['Sys']=='G']['Sat'].nunique()],
             ['IRNSS sats',  obs_df[obs_df['Sys']=='I']['Sat'].nunique()],
             ['Obs types',   obs_df['ObsType'].nunique()]]
        ))

        # 2 — Signal Availability
        n_ep = max(obs_df['UTC_Time'].nunique(), 1)
        sections.append(('Signal Availability', _DARK['accent2'],
            ['Signal', 'Count', '% of epochs'],
            [[sig, cnt, f"{100*cnt/n_ep:.1f}%"] for sig, cnt in sig_report.items()]
        ))

        # 3 — GF / MW Statistics
        if mw_df is not None and not mw_df.empty:
            rows_mw = []
            for col, name in [('gf_m', 'GF (m)'), ('mw_cycles', 'MW (cyc)')]:
                sub = mw_df.dropna(subset=[col])
                if not sub.empty:
                    rows_mw.append([name, len(sub),
                                    f"{sub[col].mean():.4f}",
                                    f"{sub[col].std():.4f}",
                                    f"{sub[col].min():.4f}",
                                    f"{sub[col].max():.4f}"])
            if rows_mw:
                sections.append(('GF / MW Statistics', _DARK['accent6'],
                    ['Combination', 'N', 'Mean', 'Std', 'Min', 'Max'], rows_mw))

        # 4 — Cycle Slip Detection
        if not slip_df.empty:
            counts_s = slip_df['status'].value_counts()
            sections.append(('Cycle Slip Detection', _DARK['accent3'],
                ['Status', 'Count', '%'],
                [[f, c, f"{100*c/len(slip_df):.2f}%"] for f, c in counts_s.items()]
            ))

        # 5 — Correction Summary
        if correction_report:
            all_types = {}
            for rep in correction_report.values():
                for s in rep.values():
                    all_types[s] = all_types.get(s, 0) + 1
            sections.append(('Correction Summary', _DARK['accent5'],
                ['Correction Type', 'Count'],
                [[t, c] for t, c in all_types.items()]
            ))

        # 6 — Phase Residuals
        if not res_df.empty:
            rows_res = []
            for sig in ('L1', 'L2', 'L5', 'LS'):
                sub = res_df[res_df['signal'] == sig]
                if sub.empty: continue
                rows_res.append([sig,
                                 f"{sub['raw_residual'].std():.4f}",
                                 f"{sub['corrected_residual'].std():.4f}",
                                 int(sub['slipped'].sum())])
            if rows_res:
                sections.append(('Phase Residuals Summary', _DARK['accent2'],
                    ['Signal', 'Raw Std (cyc)', 'Corr Std (cyc)', 'Slip Epochs'],
                    rows_res
                ))

        # 7 — Slips per Satellite
        if not slip_df.empty:
            by_sat = (slip_df[slip_df['status']
                      .isin(['GF_SLIP', 'CONFIRMED', 'GF_ONLY'])]
                      .groupby('sat').size().reset_index(name='slips')
                      .sort_values('slips', ascending=False))
            if not by_sat.empty:
                sections.append(('Slips per Satellite', _DARK['accent4'],
                    ['Satellite', 'Slip Count'],
                    by_sat.values.tolist()
                ))

        # 8 — Top Obs Types
        top_obs = (obs_df.groupby('ObsType').size()
                   .reset_index(name='count')
                   .sort_values('count', ascending=False)
                   .head(20))
        sections.append(('Observation Type Counts (top 20)', _DARK['accent6'],
            ['Obs Type', 'Count'],
            top_obs.values.tolist()
        ))

        # ── 2-column grid layout ────────────────────────────────────────────────
        sw = QWidget()
        sw.setStyleSheet(f"background:{_DARK['bg']};")
        scroll = QScrollArea(parent)
        scroll.setWidget(sw)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            f"QScrollArea{{background:{_DARK['bg']};border:none;}}"
            f"QScrollBar:vertical{{background:{_DARK['panel']};width:10px;"
            f"border-radius:5px;}}"
            f"QScrollBar::handle:vertical{{background:{_DARK['border']};"
            f"border-radius:5px;min-height:20px;}}"
        )

        outer = QVBoxLayout(sw)
        outer.setContentsMargins(16, 16, 16, 24)
        outer.setSpacing(12)

        # page title
        ttl = QLabel("📋  Statistics Summary")
        ttl.setStyleSheet(
            f"color:{_DARK['text']};font-size:16px;font-weight:700;"
            f"font-family:'Courier New',monospace;padding:4px 0 8px 0;")
        outer.addWidget(ttl)

        # place cards 2 per row
        for i in range(0, len(sections), 2):
            row_hb = QHBoxLayout()
            row_hb.setSpacing(12)
            title_l, acc_l, hdrs_l, data_l = sections[i]
            row_hb.addWidget(card(title_l, acc_l, hdrs_l, data_l), stretch=1)
            if i + 1 < len(sections):
                title_r, acc_r, hdrs_r, data_r = sections[i + 1]
                row_hb.addWidget(card(title_r, acc_r, hdrs_r, data_r), stretch=1)
            else:
                row_hb.addStretch(1)   # keep last odd card at half-width
            outer.addLayout(row_hb)

        outer.addStretch()
        return scroll

    # ══════════════════════════════════════════════════════════════════════════
    #  MAIN WINDOW
    # ══════════════════════════════════════════════════════════════════════════

    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle('Fusion')
    pal = QPalette()
    for role, color in [
        (QPalette.Window,          _DARK['bg']),
        (QPalette.WindowText,      _DARK['text']),
        (QPalette.Base,            _DARK['panel']),
        (QPalette.Text,            _DARK['text']),
        (QPalette.Button,          _DARK['panel']),
        (QPalette.ButtonText,      _DARK['text']),
        (QPalette.Highlight,       _DARK['accent1']),
        (QPalette.HighlightedText, _DARK['bg']),
    ]:
        pal.setColor(role, QColor(color))
    app.setPalette(pal)

    win = QMainWindow()
    win.setWindowTitle("GNSS Dashboard  —  IISC Post-Processing")
    win.resize(1480, 940)

    central = QWidget(); win.setCentralWidget(central)
    main_vb = QVBoxLayout(central)
    main_vb.setContentsMargins(0,0,0,0); main_vb.setSpacing(0)

    # Header bar
    hdr = QFrame(); hdr.setFixedHeight(52)
    hdr.setStyleSheet(f"background:{_DARK['panel']};border-bottom:1px solid {_DARK['border']};")
    hdr_hb = QHBoxLayout(hdr); hdr_hb.setContentsMargins(20,0,20,0)
    t_lbl = QLabel("⬡  GNSS Post-Processing Dashboard")
    t_lbl.setStyleSheet(f"color:{_DARK['text']};font-size:15px;font-weight:700;"
                        f"font-family:'Courier New',monospace;")
    f_lbl = QLabel(os.path.basename(OBS_FILE))
    f_lbl.setStyleSheet(f"color:{_DARK['subtext']};font-size:11px;")
    hdr_hb.addWidget(t_lbl); hdr_hb.addStretch(); hdr_hb.addWidget(f_lbl)
    main_vb.addWidget(hdr)

    # Tabs
    tabs = QTabWidget()
    tabs.setStyleSheet(
        f"QTabWidget::pane{{border:none;background:{_DARK['bg']};}}"
        f"QTabBar::tab{{background:{_DARK['panel']};color:{_DARK['subtext']};"
        f"padding:8px 20px;font-size:12px;border-right:1px solid {_DARK['border']};}}"
        f"QTabBar::tab:selected{{background:{_DARK['bg']};color:{_DARK['text']};"
        f"border-top:2px solid {_DARK['accent1']};}}"
        f"QTabBar::tab:hover{{color:{_DARK['text']};}}"
    )
    tabs.addTab(build_overview(tabs),       "📊  Overview")
    tabs.addTab(build_mw_tab(tabs),         "〰  GF / MW")
    tabs.addTab(build_slip_tab(tabs),       "⚡  Cycle Slips")
    tabs.addTab(build_correction_tab(tabs), "🔧  Correction")
    tabs.addTab(build_residuals_tab(tabs),  "📈  Residuals")
    tabs.addTab(build_arc_tab(tabs),        "🔍  Arc Analysis")
    tabs.addTab(build_stats_tab(tabs),      "📋  Statistics")
    main_vb.addWidget(tabs)

    win.show()
    app.exec_()

def main():
    t0 = datetime.now()
    print(f"\n{'═'*60}")
    print(f"  GNSS Pipeline  —  {t0:%Y-%m-%d %H:%M:%S}")
    print(f"{'═'*60}")

    if not os.path.exists(OBS_FILE):
        print(f"\n  ERROR: OBS file not found:\n    {OBS_FILE}")
        sys.exit(1)

    obs_df              = step_load_obs(OBS_FILE)
    meas_df, sig_report = step_extract_measurements(obs_df)
    mw_df               = step_mw_combinations(meas_df)
    slip_df, arc_mgr    = step_cycle_slip_detection(meas_df)
    correction_report   = step_slip_correction(arc_mgr)
    res_df              = step_phase_residuals(meas_df, slip_df)

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\n{'═'*60}")
    print(f"  Pipeline done ({elapsed:.1f}s)  —  launching dashboard ...")
    print(f"{'═'*60}\n")

    launch_dashboard(obs_df, meas_df, sig_report, mw_df,
                     slip_df, arc_mgr, correction_report, res_df)


if __name__ == '__main__':
    main()