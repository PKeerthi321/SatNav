"""
gui/main_window.py
SatNav GNSS Post-Processing Tool — Main Window
4-tab structure: Input & Status | Parameters | Visualisation | Export
"""

import sys
import os
import math
import numpy as np
import pandas as pd

from datetime import datetime
from functools import partial

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QTableWidget, QTableWidgetItem,
    QMessageBox, QTextEdit, QHeaderView, QTabWidget,
    QGroupBox, QGridLayout, QLineEdit, QFrame, QSizePolicy
)
from PyQt5.QtGui import QColor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from engine.processor import GNSSProcessor
from engine.models.error_budget import compute_error_budget_for_sat
from engine.geometry.dop import compute_dop_table
from engine.io.rinex_nav import parse_rinex_nav
from engine.io.rinex_obs import read_rinex_obs
from engine.geometry.orbit import compute_sat_positions_from_nav
from engine.geometry.coordinates import lla_to_ecef, ecef2aer_deg
from engine.measurements.extract_measurements import signal_availability_report


# ── Colour constants ───────────────────────────────────────────────────────────
_BG        = '#F0F9FF'
_CARD_BG   = '#E6F4FF'
_BORDER    = '#7FBFD6'
_HDR_BG    = '#BEE6FF'
_TBL_BG    = '#F7FDFF'
_DOT_IDLE  = '#B0BEC5'
_DOT_OK    = '#4CAF50'
_DOT_WARN  = '#FF9800'
_DOT_ERR   = '#F44336'
_DOT_RUN   = '#2196F3'


# ── Reusable widgets ───────────────────────────────────────────────────────────

class StatusDot(QLabel):
    """Small coloured circle indicating processing state."""
    def __init__(self):
        super().__init__()
        self.setFixedSize(14, 14)
        self._set(_DOT_IDLE)

    def _set(self, colour):
        self.setStyleSheet(
            f'background:{colour};border-radius:7px;'
            f'border:1px solid rgba(0,0,0,0.12);'
        )

    def set_idle(self):    self._set(_DOT_IDLE)
    def set_running(self): self._set(_DOT_RUN)
    def set_ok(self):      self._set(_DOT_OK)
    def set_warning(self): self._set(_DOT_WARN)
    def set_error(self):   self._set(_DOT_ERR)


class StatusRow(QWidget):
    """One labelled row: dot + text."""
    def __init__(self, text):
        super().__init__()
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 2, 0, 2)
        h.setSpacing(8)
        self.dot  = StatusDot()
        self._base = text
        self.lbl  = QLabel(text)
        self.lbl.setStyleSheet('font-size:12px;')
        h.addWidget(self.dot)
        h.addWidget(self.lbl)
        h.addStretch()

    def _update(self, setter, extra):
        setter()
        if extra:
            self.lbl.setText(self._base + '  —  ' + extra)

    def set_idle(self):              self._update(self.dot.set_idle,    '')
    def set_running(self):           self._update(self.dot.set_running, 'running...')
    def set_ok(self, msg=''):        self._update(self.dot.set_ok,      msg)
    def set_warning(self, msg=''):   self._update(self.dot.set_warning, msg)
    def set_error(self, msg=''):     self._update(self.dot.set_error,   msg)


class MetricCard(QWidget):
    """Summary number card: big value + small label."""
    def __init__(self, label, value='—', colour='#1565C0'):
        super().__init__()
        self.setStyleSheet(
            f'background:{_CARD_BG};border-radius:8px;'
            f'border:1px solid {_BORDER};'
        )
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(2)
        self._num = QLabel(str(value))
        self._num.setAlignment(Qt.AlignCenter)
        self._num.setStyleSheet(
            f'font-size:22px;font-weight:700;color:{colour};'
            f'background:transparent;border:none;'
        )
        self._lbl = QLabel(label)
        self._lbl.setAlignment(Qt.AlignCenter)
        self._lbl.setStyleSheet(
            'font-size:11px;color:#546E7A;'
            'background:transparent;border:none;'
        )
        v.addWidget(self._num)
        v.addWidget(self._lbl)

    def update_value(self, value, colour=None):
        self._num.setText(str(value))
        if colour:
            self._num.setStyleSheet(
                f'font-size:22px;font-weight:700;color:{colour};'
                f'background:transparent;border:none;'
            )


# ── Main Window ────────────────────────────────────────────────────────────────

class DOPGui(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('SatNav — GNSS Post-Processing Tool')
        self.resize(1280, 820)

        self.processor = GNSSProcessor()

        self.data = {
            'obs_table':     pd.DataFrame(),
            'nav_gps':       [],
            'nav_irnss':     [],
            'sat_positions': pd.DataFrame(),
            'dop_table':     pd.DataFrame(),
            'selected_sats': [],
            'slip_df':       pd.DataFrame(),
        }

        self._build_ui()
        self._apply_styles()

    # ══════════════════════════════════════════════════════════════════════════
    # UI CONSTRUCTION
    # ══════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        root.addWidget(self.tabs)

        self.tabs.addTab(self._build_tab1(),                           '  1 · Input & Status  ')
        self.tabs.addTab(self._placeholder('Parameters'),              '  2 · Parameters  ')
        self.tabs.addTab(self._build_tab3(),                           '  3 · Visualisation  ')
        self.tabs.addTab(self._placeholder('Export'),                  '  4 · Export  ')

    # ── Tab 1 — Input & Status ─────────────────────────────────────────────

    def _build_tab1(self):
        tab = QWidget()
        root = QVBoxLayout(tab)
        root.setContentsMargins(14, 14, 14, 8)
        root.setSpacing(10)

        top = QHBoxLayout()
        top.setSpacing(12)
        top.addLayout(self._build_input_panel(),  stretch=5)
        top.addLayout(self._build_status_panel(), stretch=4)
        root.addLayout(top)

        root.addWidget(self._build_pipeline_panel())

        root.addWidget(QLabel('Status log:'))
        self.status = QTextEdit()
        self.status.setReadOnly(True)
        self.status.setFixedHeight(110)
        self.status.setStyleSheet(
            'background:#F0F9FF;border:1px solid #B0D8EF;'
            'font-family:Consolas,monospace;font-size:11px;'
        )
        root.addWidget(self.status)
        return tab

    def _build_input_panel(self):
        col = QVBoxLayout()
        col.setSpacing(8)

        # File loading
        grp = QGroupBox('Input files')
        grp.setStyleSheet(self._grp())
        g = QVBoxLayout(grp)
        g.setSpacing(6)

        self.btn_load_obs       = self._btn('Load RINEX OBS',   '#1565C0')
        self.btn_load_nav_gps   = self._btn('Load GPS NAV',     '#2E7D32')
        self.btn_load_nav_irnss = self._btn('Load IRNSS NAV',   '#6A1B9A')

        self.lbl_obs_file   = QLabel('No file loaded')
        self.lbl_nav_gps    = QLabel('No file loaded')
        self.lbl_nav_irnss  = QLabel('No file loaded')
        for lbl in (self.lbl_obs_file, self.lbl_nav_gps, self.lbl_nav_irnss):
            lbl.setStyleSheet('font-size:11px;color:#546E7A;')
            lbl.setWordWrap(True)

        for btn, lbl in (
            (self.btn_load_obs,       self.lbl_obs_file),
            (self.btn_load_nav_gps,   self.lbl_nav_gps),
            (self.btn_load_nav_irnss, self.lbl_nav_irnss),
        ):
            row = QHBoxLayout()
            row.addWidget(btn)
            row.addWidget(lbl, 1)
            g.addLayout(row)

        col.addWidget(grp)

        # Receiver position
        pos_grp = QGroupBox('Receiver position (approximate)')
        pos_grp.setStyleSheet(self._grp())
        pg = QGridLayout(pos_grp)
        pg.setSpacing(6)
        pg.addWidget(QLabel('Latitude  (°N):'), 0, 0)
        pg.addWidget(QLabel('Longitude (°E):'), 1, 0)
        pg.addWidget(QLabel('Altitude    (m):'), 2, 0)
        self.inp_lat = self._lineedit('20.0')
        self.inp_lon = self._lineedit('78.0')
        self.inp_alt = self._lineedit('0.0')
        pg.addWidget(self.inp_lat, 0, 1)
        pg.addWidget(self.inp_lon, 1, 1)
        pg.addWidget(self.inp_alt, 2, 1)
        col.addWidget(pos_grp)
        col.addStretch()
        return col

    def _build_status_panel(self):
        col = QVBoxLayout()
        col.setSpacing(8)

        # Status rows
        grp = QGroupBox('Processing status')
        grp.setStyleSheet(self._grp())
        g = QVBoxLayout(grp)
        g.setSpacing(1)

        self.st_obs     = StatusRow('OBS file')
        self.st_nav     = StatusRow('NAV files')
        self.st_satpos  = StatusRow('Satellite positions')
        self.st_dop     = StatusRow('DOP computation')
        self.st_errors  = StatusRow('Error budget')
        self.st_meas    = StatusRow('Measurement extraction')
        self.st_mw      = StatusRow('MW combination')
        self.st_slips   = StatusRow('Cycle slip detection')
        self.st_correct = StatusRow('Cycle slip correction')
        self.st_filter  = StatusRow('Kalman filter (PPP)')
        self.st_lambda  = StatusRow('LAMBDA IAR')

        for w in (self.st_obs, self.st_nav, self.st_satpos, self.st_dop,
                  self.st_errors, self.st_meas, self.st_mw, self.st_slips,
                  self.st_correct, self.st_filter, self.st_lambda):
            g.addWidget(w)

        col.addWidget(grp)

        # Metric cards
        mg_grp = QGroupBox('Summary')
        mg_grp.setStyleSheet(self._grp())
        mg = QGridLayout(mg_grp)
        mg.setSpacing(6)
        self.card_epochs = MetricCard('Epochs',     '—')
        self.card_sats   = MetricCard('Satellites', '—')
        self.card_slips  = MetricCard('Slips',      '—', colour=_DOT_WARN)
        self.card_pdop   = MetricCard('Avg PDOP',   '—')
        mg.addWidget(self.card_epochs, 0, 0)
        mg.addWidget(self.card_sats,   0, 1)
        mg.addWidget(self.card_slips,  1, 0)
        mg.addWidget(self.card_pdop,   1, 1)
        col.addWidget(mg_grp)
        col.addStretch()
        return col

    def _build_pipeline_panel(self):
        grp = QGroupBox('Processing pipeline  (run steps in order, or click Run all)')
        grp.setStyleSheet(self._grp())
        h = QHBoxLayout(grp)
        h.setSpacing(8)

        self.btn_extract  = self._btn('1 · Extract measurements', '#00695C')
        self.btn_comp_dop = self._btn('2 · Compute DOP',          '#1565C0')
        self.btn_comp_err = self._btn('3 · Compute errors',       '#4527A0')
        self.btn_run_mw   = self._btn('4 · Run MW detection',     '#E65100')
        self.btn_run_corr = self._btn('5 · Run correction',       '#BF360C')

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet('color:#B0D8EF;')

        self.btn_run_all = self._btn('Run all  ▶', '#B71C1C', bold=True)

        for w in (self.btn_extract, self.btn_comp_dop, self.btn_comp_err,
                  self.btn_run_mw, self.btn_run_corr, sep, self.btn_run_all):
            h.addWidget(w)

        # Wire buttons
        self.btn_load_obs.clicked.connect(self.on_load_obs)
        self.btn_load_nav_gps.clicked.connect(partial(self.on_load_nav, 'G'))
        self.btn_load_nav_irnss.clicked.connect(partial(self.on_load_nav, 'I'))
        self.btn_extract.clicked.connect(self.on_extract_measurements)
        self.btn_comp_dop.clicked.connect(self.on_compute_dop)
        self.btn_comp_err.clicked.connect(self.on_compute_errors)
        self.btn_run_mw.clicked.connect(self.on_detect_slips)
        self.btn_run_corr.clicked.connect(self.on_run_correction)
        self.btn_run_all.clicked.connect(self.on_run_all)

        return grp

    # ── Tab 3 — Visualisation ──────────────────────────────────────────────

    def _build_tab3(self):
        """
        Tab 3 contains two sub-tabs:
          A) Geometry  — DOP table, error budget, skyplot (existing)
          B) Cycle Slips — detection timeline, Δφ per satellite,
                           correction breakdown, signal availability bar
        """
        tab  = QWidget()
        root = QVBoxLayout(tab)
        root.setContentsMargins(0, 0, 0, 0)

        self.viz_tabs = QTabWidget()
        self.viz_tabs.setStyleSheet(
            'QTabBar::tab{padding:6px 18px;font-size:12px;font-weight:600;}'
        )
        self.viz_tabs.addTab(self._build_geometry_subtab(), '  🌐  Geometry & DOP  ')
        self.viz_tabs.addTab(self._build_slips_subtab(),    '  ⚡  Cycle Slips  ')
        root.addWidget(self.viz_tabs)
        return tab

    # ── Sub-tab A: Geometry (original content) ─────────────────────────────

    def _build_geometry_subtab(self):
        tab  = QWidget()
        root = QHBoxLayout(tab)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        tbl_style = (
            f'QHeaderView::section{{background:{_HDR_BG};padding:4px;'
            f'border:1px solid #9ECFE6;font-weight:700;}}'
            f'QTableWidget{{background:{_TBL_BG};}}'
        )

        # Left: DOP table + error budget
        left = QVBoxLayout()
        left.addWidget(QLabel('DOP Table'))
        self.tbl_dop = QTableWidget(0, 9)
        self.tbl_dop.setHorizontalHeaderLabels(
            ['#', 'Sat', 'Sys', 'El(°)', 'Az(°)', 'PDOP', 'HDOP', 'VDOP', 'dPDOP'])
        self.tbl_dop.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_dop.setStyleSheet(tbl_style)
        self.tbl_dop.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tbl_dop.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        left.addWidget(self.tbl_dop)

        left.addWidget(QLabel('Error Budget'))
        self.tbl_err = QTableWidget(0, 8)
        self.tbl_err.setHorizontalHeaderLabels(
            ['#', 'Sat', 'El(°)', 'Iono(m)', 'Tropo(m)', 'Clock(m)', 'MP(m)', 'Total(m)'])
        self.tbl_err.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_err.setStyleSheet(tbl_style)
        self.tbl_err.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tbl_err.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        left.addWidget(self.tbl_err)
        root.addLayout(left, stretch=3)

        # Right: satellite selection + skyplot
        right = QVBoxLayout()
        right.addWidget(QLabel('Satellite Selection'))
        self.tbl_sel = QTableWidget(0, 3)
        self.tbl_sel.setHorizontalHeaderLabels(['Select', 'Sat', 'Sys'])
        self.tbl_sel.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tbl_sel.horizontalHeader().setStretchLastSection(True)
        self.tbl_sel.setFixedHeight(200)
        self.tbl_sel.setStyleSheet(tbl_style)
        self.tbl_sel.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tbl_sel.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        right.addWidget(self.tbl_sel)

        self.btn_plot_selected = self._btn('Plot Selected Satellites', '#1565C0')
        self.btn_plot_selected.clicked.connect(self.on_plot_selected)
        right.addWidget(self.btn_plot_selected)

        right.addWidget(QLabel('Skyplot'))
        self.fig = Figure(figsize=(5, 5))
        self.fig.patch.set_facecolor('#E9F8FF')
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, polar=True)
        self._configure_sky_axes()
        right.addWidget(self.canvas)
        root.addLayout(right, stretch=2)

        return tab

    # ── Sub-tab B: Cycle Slips ─────────────────────────────────────────────

    def _build_slips_subtab(self):
        """
        Four plots arranged in a 2×2 grid:
          [0,0] Detection timeline  — epoch × satellite scatter, colour=status
          [0,1] Signal availability — bar chart of obs counts per signal
          [1,0] Δφ per satellite    — line plot with ±tol threshold lines
          [1,1] Correction breakdown— horizontal bar chart per satellite
        """
        tab  = QWidget()
        root = QVBoxLayout(tab)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # Refresh button
        btn_row = QHBoxLayout()
        self.btn_refresh_slips = self._btn('Refresh Slip Plots  ↻', '#E65100')
        self.btn_refresh_slips.clicked.connect(self.refresh_slip_plots)
        btn_row.addWidget(self.btn_refresh_slips)
        btn_row.addStretch()

        # Satellite selector for Δφ plot
        btn_row.addWidget(QLabel('Sat for Δφ:'))
        self.cmb_slip_sat = QtWidgets.QComboBox()
        self.cmb_slip_sat.setFixedWidth(90)
        self.cmb_slip_sat.currentTextChanged.connect(self._update_dphi_plot)
        btn_row.addWidget(self.cmb_slip_sat)
        root.addLayout(btn_row)

        # 2×2 matplotlib figure
        self.slip_fig = Figure(figsize=(13, 7))
        self.slip_fig.patch.set_facecolor(_BG)
        self.slip_fig.subplots_adjust(
            left=0.06, right=0.97, top=0.92, bottom=0.08,
            wspace=0.32, hspace=0.42)
        self.slip_canvas = FigureCanvas(self.slip_fig)
        self.slip_canvas.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)

        gs = self.slip_fig.add_gridspec(2, 2)
        self.ax_timeline  = self.slip_fig.add_subplot(gs[0, 0])
        self.ax_sigbar    = self.slip_fig.add_subplot(gs[0, 1])
        self.ax_dphi      = self.slip_fig.add_subplot(gs[1, 0])
        self.ax_corr      = self.slip_fig.add_subplot(gs[1, 1])

        self._init_slip_axes()
        root.addWidget(self.slip_canvas)
        return tab

    # ── Slip plot helpers ──────────────────────────────────────────────────

    _STATUS_COLOURS = {
        'CONFIRMED': '#D32F2F',
        'GF_SLIP':   '#FF7043',
        'GF_ONLY':   '#7B1FA2',
        'MW_ONLY':   '#1565C0',
        'CLEAN':     '#A5D6A7',
        'INIT':      '#CFD8DC',
    }

    def _ax_style(self, ax, title, xlabel='', ylabel=''):
        ax.set_facecolor(_CARD_BG)
        ax.set_title(title, fontsize=10, fontweight='bold', color='#0D47A1', pad=6)
        if xlabel: ax.set_xlabel(xlabel, fontsize=8, color='#546E7A')
        if ylabel: ax.set_ylabel(ylabel, fontsize=8, color='#546E7A')
        ax.tick_params(labelsize=7, colors='#546E7A')
        ax.grid(True, color='#B0D8EF', linestyle='--', alpha=0.5, linewidth=0.6)
        for sp in ax.spines.values():
            sp.set_color(_BORDER)

    def _init_slip_axes(self):
        for ax, title in [
            (self.ax_timeline, 'Detection Timeline'),
            (self.ax_sigbar,   'Signal Availability'),
            (self.ax_dphi,     'Δφ (epoch-to-epoch GF diff)'),
            (self.ax_corr,     'Correction Breakdown per Satellite'),
        ]:
            self._ax_style(ax, title)
            ax.text(0.5, 0.5, 'Run detection first',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=9, color='#90A4AE')
        self.slip_canvas.draw_idle()

    def refresh_slip_plots(self):
        """Redraw all four cycle slip plots from current processor data."""
        slip_df = self.data.get('slip_df', pd.DataFrame())
        meas_df = self.processor.meas_df

        if slip_df is None or (hasattr(slip_df, 'empty') and slip_df.empty):
            QMessageBox.warning(self, 'No data',
                                'Run cycle slip detection first.')
            return

        self._plot_timeline(slip_df)
        self._plot_sigbar(meas_df)
        self._populate_sat_combo(slip_df)
        self._update_dphi_plot()
        self._plot_corr_breakdown(slip_df)
        self.slip_canvas.draw_idle()
        # Switch to the slip sub-tab automatically
        self.viz_tabs.setCurrentIndex(1)

    def _plot_timeline(self, slip_df):
        ax = self.ax_timeline
        ax.clear()
        self._ax_style(ax, 'Detection Timeline  —  all satellites',
                       'Epoch index', 'Satellite')

        sats    = sorted(slip_df['sat'].unique())
        sat_idx = {s: i for i, s in enumerate(sats)}

        plotted = set()
        for _, row in slip_df.iterrows():
            status = row.get('status', 'CLEAN')
            colour = self._STATUS_COLOURS.get(status, '#90A4AE')
            label  = status if status not in plotted else '_nolegend_'
            plotted.add(status)
            ax.scatter(row['epoch_idx'], sat_idx[row['sat']],
                       c=colour, s=6, marker='s',
                       linewidths=0, label=label, zorder=2)

        ax.set_yticks(range(len(sats)))
        ax.set_yticklabels(sats, fontsize=6)
        ax.yaxis.grid(False)

        # Compact legend
        handles = [
            __import__('matplotlib.patches', fromlist=['Patch']).Patch(
                facecolor=self._STATUS_COLOURS.get(s, '#ccc'), label=s)
            for s in ('CONFIRMED', 'GF_SLIP', 'GF_ONLY', 'CLEAN', 'INIT')
            if s in slip_df['status'].values
        ]
        if handles:
            ax.legend(handles=handles, loc='upper right',
                      fontsize=6, ncol=2, framealpha=0.8)

    def _plot_sigbar(self, meas_df):
        ax = self.ax_sigbar
        ax.clear()
        self._ax_style(ax, 'Signal Availability', 'Signal', 'Observations')

        if meas_df is None:
            ax.text(0.5, 0.5, 'No measurements', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9, color='#90A4AE')
            return

        sig_cols = ['P1', 'P2', 'P5', 'PS', 'L1', 'L2', 'L5', 'LS']
        sigs     = [c for c in sig_cols if c in meas_df.columns]
        counts   = [int(meas_df[c].notna().sum()) for c in sigs]
        colours  = ['#64B5F6' if c.startswith('P') else
                    '#81C784' if c.startswith('L') else '#FFB74D'
                    for c in sigs]

        bars = ax.bar(sigs, counts, color=colours, width=0.6, zorder=3)
        ax.bar_label(bars, fmt='%d', fontsize=7, padding=2, color='#0D47A1')
        ax.set_ylim(0, max(counts) * 1.18 if counts else 10)

    def _populate_sat_combo(self, slip_df):
        """Fill the satellite combo with satellites that have slips."""
        slip_sats = sorted(
            slip_df[slip_df['status'].isin(
                ['CONFIRMED', 'GF_SLIP', 'GF_ONLY'])
            ]['sat'].unique()
        )
        all_sats = sorted(slip_df['sat'].unique())
        options  = slip_sats if slip_sats else all_sats

        self.cmb_slip_sat.blockSignals(True)
        self.cmb_slip_sat.clear()
        self.cmb_slip_sat.addItems(options)
        self.cmb_slip_sat.blockSignals(False)

    def _update_dphi_plot(self):
        """Redraw the Δφ line plot for the currently selected satellite."""
        ax      = self.ax_dphi
        sat     = self.cmb_slip_sat.currentText()
        slip_df = self.data.get('slip_df', pd.DataFrame())

        ax.clear()
        self._ax_style(ax, f'Δφ — {sat}' if sat else 'Δφ', 'Epoch', 'Δφ (cycles)')

        if slip_df is None or (hasattr(slip_df, 'empty') and slip_df.empty) or not sat:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9, color='#90A4AE')
            self.slip_canvas.draw_idle()
            return

        sub = slip_df[slip_df['sat'] == sat].sort_values('epoch_idx')
        if sub.empty:
            return

        x    = sub['epoch_idx'].values
        phi  = sub['dphi'].values
        tol  = sub['tol'].iloc[0] if 'tol' in sub.columns else 0.10

        # Clip extreme outliers for readability
        phi_disp = np.clip(phi, -5.0, 5.0)
        clipped  = int(np.sum(np.abs(phi) > 5.0))

        ax.fill_between(x, phi_disp, 0,
                        where=(phi_disp >= 0),
                        color='#90CAF9', alpha=0.4)
        ax.fill_between(x, phi_disp, 0,
                        where=(phi_disp < 0),
                        color='#EF9A9A', alpha=0.4)
        ax.plot(x, phi_disp, color='#1565C0', lw=0.9, alpha=0.9)
        ax.axhline( tol, color='#D32F2F', lw=1.0, ls='--',
                    label=f'+tol ({tol:.3f})')
        ax.axhline(-tol, color='#D32F2F', lw=1.0, ls='--',
                    label=f'-tol ({tol:.3f})')
        ax.axhline(0, color='#90A4AE', lw=0.6)

        # Mark detected slip epochs
        slip_mask = sub['status'].isin(['CONFIRMED', 'GF_SLIP', 'GF_ONLY']).values
        if slip_mask.any():
            ax.scatter(x[slip_mask],
                       np.clip(phi[slip_mask], -5.0, 5.0),
                       color='#FF6F00', s=40, zorder=5, marker='^',
                       label='Slip')

        ax.set_ylim(-5.5, 5.5)
        title = f'Δφ — {sat}'
        if clipped:
            title += f'  [{clipped} pts >±5 cyc clipped]'
        ax.set_title(title, fontsize=10, fontweight='bold',
                     color='#0D47A1', pad=6)
        ax.legend(fontsize=7, loc='upper right')
        self.slip_canvas.draw_idle()

    def _plot_corr_breakdown(self, slip_df):
        """Horizontal stacked bar: correction types per satellite."""
        ax = self.ax_corr
        ax.clear()
        self._ax_style(ax, 'Slip Status per Satellite', 'Count', 'Satellite')

        slip_types = ['CONFIRMED', 'GF_ONLY', 'GF_SLIP', 'CLEAN']
        colours    = {
            'CONFIRMED': '#D32F2F',
            'GF_ONLY':   '#7B1FA2',
            'GF_SLIP':   '#FF7043',
            'CLEAN':     '#81C784',
        }

        # Only show satellites with at least one non-INIT, non-CLEAN status
        active = slip_df[slip_df['status'].isin(
            ['CONFIRMED', 'GF_ONLY', 'GF_SLIP'])]['sat'].unique()
        if len(active) == 0:
            active = slip_df['sat'].unique()

        # Limit to top 20 by total slip count
        top20 = (slip_df[slip_df['sat'].isin(active) &
                         slip_df['status'].isin(['CONFIRMED','GF_ONLY','GF_SLIP'])]
                 .groupby('sat').size()
                 .nlargest(20).index.tolist())
        sats = top20 if top20 else sorted(active)[:20]

        lefts = np.zeros(len(sats))
        for st in slip_types:
            vals = np.array([
                int((slip_df[(slip_df['sat'] == s) &
                             (slip_df['status'] == st)].shape[0]))
                for s in sats
            ])
            if vals.sum() == 0:
                continue
            ax.barh(sats, vals, left=lefts,
                    color=colours.get(st, '#CFD8DC'),
                    label=st, height=0.6, zorder=3)
            lefts += vals

        ax.set_yticks(range(len(sats)))
        ax.set_yticklabels(sats, fontsize=7)
        ax.legend(fontsize=7, loc='lower right')

    def _placeholder(self, name):
        w = QWidget()
        v = QVBoxLayout(w)
        lbl = QLabel(f'{name} — coming in Week 3')
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet('font-size:16px;color:#90A4AE;')
        v.addWidget(lbl)
        return w

    # ══════════════════════════════════════════════════════════════════════════
    # STYLE HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _grp(self):
        return (
            'QGroupBox{font-size:12px;font-weight:600;color:#1565C0;'
            'border:1px solid #B0D8EF;border-radius:8px;margin-top:8px;'
            'padding-top:4px;background:#FAFEFF;}'
            'QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 4px;}'
        )

    def _btn(self, text, colour='#1565C0', bold=False):
        w = 'font-weight:700;' if bold else 'font-weight:600;'
        return self._make_btn(text, colour, w)

    def _make_btn(self, text, colour, weight_css):
        b = QPushButton(text)
        b.setStyleSheet(
            f'QPushButton{{background:{_CARD_BG};border:1px solid {_BORDER};'
            f'border-radius:6px;padding:7px 12px;font-size:12px;{weight_css}'
            f'color:{colour};}}'
            f'QPushButton:hover{{background:{_HDR_BG};}}'
            f'QPushButton:pressed{{background:{_BORDER};}}'
            f'QPushButton:disabled{{background:#ECEFF1;color:#B0BEC5;border-color:#CFD8DC;}}'
        )
        return b

    def _lineedit(self, default=''):
        le = QLineEdit(default)
        le.setStyleSheet(
            'border:1px solid #B0D8EF;border-radius:4px;'
            'padding:4px 8px;font-size:12px;background:white;'
        )
        le.setFixedHeight(28)
        return le

    def _apply_styles(self):
        self.setStyleSheet(
            f'QMainWindow{{background:{_BG};}}'
            'QTabWidget::pane{border:none;background:#F0F9FF;}'
            'QTabBar::tab{background:#E3F2FD;border:1px solid #B0D8EF;'
            'border-bottom:none;padding:8px 16px;font-size:12px;'
            'font-weight:600;color:#1565C0;'
            'border-top-left-radius:6px;border-top-right-radius:6px;}'
            'QTabBar::tab:selected{background:#FAFEFF;color:#0D47A1;}'
            'QTabBar::tab:hover:!selected{background:#BBDEFB;}'
        )

    # ══════════════════════════════════════════════════════════════════════════
    # LOGGING
    # ══════════════════════════════════════════════════════════════════════════

    def log(self, msg, level='INFO'):
        t = datetime.now().strftime('%H:%M:%S')
        colours = {
            'INFO': '#1565C0', 'OK': '#2E7D32',
            'WARN': '#E65100', 'ERR': '#B71C1C',
        }
        c = colours.get(level, '#1565C0')
        self.status.append(
            f'<span style="color:#90A4AE;">[{t}]</span> '
            f'<span style="color:{c};">{msg}</span>'
        )

    def _rx_ecef(self):
        try:
            lat = float(self.inp_lat.text())
            lon = float(self.inp_lon.text())
            alt = float(self.inp_alt.text())
        except ValueError:
            lat, lon, alt = 20.0, 78.0, 0.0
        return lla_to_ecef(lat, lon, alt)

    # ══════════════════════════════════════════════════════════════════════════
    # BUTTON HANDLERS
    # ══════════════════════════════════════════════════════════════════════════

    def on_load_obs(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, 'Select RINEX OBS file', '',
            'RINEX files (*.rnx *.obs *.24o *.23o *.22o *.21o *.gz);;All files (*.*)'
        )
        if not fname:
            return
        try:
            df = read_rinex_obs(fname)
            if df.empty:
                raise ValueError('No observations found in file')
            self.data['obs_table'] = df
            self.processor.obs_df  = df
            n_ep   = df['UTC_Time'].nunique()
            n_sats = df['Sat'].nunique()
            self.lbl_obs_file.setText(os.path.basename(fname))
            self.st_obs.set_ok(f'{n_ep} epochs, {n_sats} satellites')
            self.card_epochs.update_value(n_ep)
            self.card_sats.update_value(n_sats)
            self.log(f'OBS loaded: {os.path.basename(fname)}  '
                     f'epochs={n_ep}  sats={n_sats}', 'OK')
        except Exception as e:
            self.st_obs.set_error(str(e))
            QMessageBox.critical(self, 'OBS Load Error', str(e))
            self.log(f'OBS load failed: {e}', 'ERR')

    def on_load_nav(self, system_char):
        files, _ = QFileDialog.getOpenFileNames(
            self, f'Select NAV files ({system_char})', '',
            'NAV files (*.rnx *.nav *.24n *.23n *.gz);;All files (*.*)'
        )
        if not files:
            return
        if system_char == 'G':
            self.data['nav_gps']    = files
            self.processor.nav_gps  = files
            self.lbl_nav_gps.setText(f'{len(files)} file(s) loaded')
        else:
            self.data['nav_irnss']   = files
            self.processor.nav_irnss = files
            self.lbl_nav_irnss.setText(f'{len(files)} file(s) loaded')
        self.st_nav.set_ok(f'{len(files)} file(s)')
        self.log(f'NAV ({system_char}) loaded: {len(files)} file(s)', 'OK')

    def on_extract_measurements(self):
        if self.data['obs_table'].empty:
            QMessageBox.warning(self, 'No OBS', 'Load a RINEX OBS file first.')
            return
        self.st_meas.set_running()
        try:
            self.processor.obs_df = self.data['obs_table']
            meas = self.processor.extract_measurements()
            report = signal_availability_report(meas)
            available = [k for k, v in report.items() if v > 0]
            n_ep   = meas['epoch'].nunique()
            n_sats = meas['sat'].nunique()
            self.st_meas.set_ok(f'{n_ep} epochs, {n_sats} sats — {", ".join(available)}')
            self.log(
                f'Measurements extracted — {n_ep} epochs, {n_sats} sats.  '
                f'Signals: {", ".join(available)}', 'OK'
            )
            for sig, n in report.items():
                if n > 0:
                    self.log(f'    {sig}: {n} observations', 'INFO')
        except Exception as e:
            self.st_meas.set_error(str(e))
            QMessageBox.critical(self, 'Extraction Error', str(e))
            self.log(f'Extraction failed: {e}', 'ERR')

    def on_compute_dop(self):
        if self.data['obs_table'].empty:
            QMessageBox.warning(self, 'No OBS', 'Load a RINEX OBS file first.')
            return
        self.st_satpos.set_running()
        self.st_dop.set_running()
        try:
            epoch  = self.data['obs_table']['UTC_Time'].iloc[0]
            sats   = sorted(self.data['obs_table']['Sat'].unique().tolist())
            satpos = compute_sat_positions_from_nav(
                self.data['nav_gps'], self.data['nav_irnss'], epoch, sats)
            self.data['sat_positions']   = satpos
            self.processor.sat_positions = satpos
            self.st_satpos.set_ok(f'{len(satpos)} satellites positioned')
            self.log(f'Satellite positions computed for {len(satpos)} satellites', 'OK')

            rx     = self._rx_ecef()
            dop    = compute_dop_table(satpos, rx)
            self.data['dop_table']   = dop
            self.processor.dop_table = dop
            self.populate_dop_table(dop)
            self.populate_selection_table(dop)

            pdop = dop['PDOP'].iloc[0]
            colour = _DOT_OK if pdop < 3 else (_DOT_WARN if pdop < 6 else _DOT_ERR)
            self.card_pdop.update_value(f'{pdop:.2f}', colour=colour)
            self.st_dop.set_ok(f'PDOP={pdop:.2f}  HDOP={dop["HDOP"].iloc[0]:.2f}')
            self.log(f'DOP computed — PDOP={pdop:.3f}  HDOP={dop["HDOP"].iloc[0]:.3f}  '
                     f'VDOP={dop["VDOP"].iloc[0]:.3f}', 'OK')
        except Exception as e:
            self.st_dop.set_error(str(e))
            QMessageBox.critical(self, 'DOP Error', str(e))
            self.log(f'DOP failed: {e}', 'ERR')

    def on_compute_errors(self):
        if self.data['dop_table'].empty:
            QMessageBox.warning(self, 'No DOP', 'Compute DOP first.')
            return
        self.st_errors.set_running()
        try:
            rx = self._rx_ecef()
            self.compute_and_populate_error_budget(self.data['dop_table'], rx)
            self.plot_sats_on_sky(self.data['sat_positions'], rx)
            rows  = self.processor.error_table or []
            if rows:
                avg = sum(r['total_m'] for r in rows) / len(rows)
                self.st_errors.set_ok(f'avg total = {avg:.2f} m')
                self.log(f'Error budget computed — avg total = {avg:.3f} m', 'OK')
            else:
                self.st_errors.set_ok()
                self.log('Error budget computed', 'OK')
        except Exception as e:
            self.st_errors.set_error(str(e))
            QMessageBox.critical(self, 'Error Budget Error', str(e))
            self.log(f'Error budget failed: {e}', 'ERR')

    def on_detect_slips(self):
        if self.processor.meas_df is None:
            QMessageBox.warning(self, 'No measurements',
                                'Run "Extract measurements" first.')
            return
        self.st_mw.set_running()
        self.st_slips.set_running()
        try:
            slip_df  = self.processor.compute_cycle_slips()
            self.data['slip_df'] = slip_df
            summary  = self.processor.slip_summary()
            confirmed = summary.get('CONFIRMED', 0)
            mw_only   = summary.get('MW_ONLY',   0)
            gf_only   = summary.get('GF_ONLY',   0)
            clean     = summary.get('CLEAN',     0)
            init      = summary.get('INIT',      0)
            total     = sum(summary.values())

            slip_colour = _DOT_OK if confirmed == 0 else (
                          _DOT_WARN if confirmed < 5 else _DOT_ERR)
            self.card_slips.update_value(confirmed, colour=slip_colour)
            self.st_mw.set_ok('computed for all pairs')
            self.st_slips.set_ok(
                f'confirmed={confirmed}  mw_only={mw_only}  '
                f'gf_only={gf_only}  clean={clean}'
            )
            self.log(
                f'Cycle slip detection complete — '
                f'total={total}  CONFIRMED={confirmed}  '
                f'MW_ONLY={mw_only}  GF_ONLY={gf_only}  '
                f'CLEAN={clean}  INIT={init}',
                'WARN' if confirmed > 0 else 'OK'
            )
            # Auto-refresh cycle slip plots
            self.refresh_slip_plots()
        except Exception as e:
            self.st_slips.set_error(str(e))
            QMessageBox.critical(self, 'Detection Error', str(e))
            self.log(f'Slip detection failed: {e}', 'ERR')

    def on_run_correction(self):
        if self.data['slip_df'].empty:
            QMessageBox.warning(self, 'No slips', 'Run MW detection first.')
            return
        self.st_correct.set_warning('pending — Week 2 task')
        self.log('Cycle slip correction — not yet implemented (Week 2)', 'WARN')

    def on_plot_selected(self):
        if self.data['sat_positions'].empty:
            QMessageBox.warning(self, 'No Data', 'Compute DOP first.')
            return
        selected = self._get_selected_sats()
        if not selected:
            QMessageBox.warning(self, 'No Selection',
                                'Check satellites in the selection table.')
            return
        rx     = self._rx_ecef()
        sel_df = self.data['sat_positions'][
            self.data['sat_positions']['Sat'].isin(selected)]
        if sel_df.empty:
            QMessageBox.warning(self, 'No Positions',
                                'Selected satellites have no computed positions.')
            return
        self.plot_sats_on_sky(sel_df, rx, highlight=True)
        self.log(f'Plotted {len(sel_df)} selected satellites', 'OK')

    def on_run_all(self):
        self.log('▶  Running full pipeline...', 'INFO')
        steps = [
            ('Extract measurements', self.on_extract_measurements),
            ('Compute DOP',          self.on_compute_dop),
            ('Compute errors',       self.on_compute_errors),
            ('Detect slips',         self.on_detect_slips),
        ]
        for name, fn in steps:
            try:
                fn()
            except Exception as e:
                self.log(f'Pipeline stopped at "{name}": {e}', 'ERR')
                return
        self.log('✓  Full pipeline complete', 'OK')

    # ══════════════════════════════════════════════════════════════════════════
    # TABLE POPULATION + SKYPLOT  (Tab 3)
    # ══════════════════════════════════════════════════════════════════════════

    def populate_dop_table(self, doptbl):
        self.tbl_dop.setRowCount(0)
        for i, row in doptbl.iterrows():
            r = self.tbl_dop.rowCount()
            self.tbl_dop.insertRow(r)
            vals = [
                str(r + 1),
                str(row['Sat']),
                str(row['Sys']),
                f"{row.get('El', float('nan')):.2f}",
                f"{row.get('Az', float('nan')):.1f}",
                f"{row['PDOP']:.3f}",
                f"{row['HDOP']:.3f}",
                f"{row['VDOP']:.3f}",
                f"{row.get('dPDOP_if_removed', float('nan')):.4f}",
            ]
            for col, v in enumerate(vals):
                self.tbl_dop.setItem(r, col, QTableWidgetItem(v))
        self.tbl_dop.resizeRowsToContents()

    def populate_selection_table(self, doptbl):
        self.tbl_sel.setRowCount(0)
        for i, row in doptbl.iterrows():
            r = self.tbl_sel.rowCount()
            self.tbl_sel.insertRow(r)
            chk  = QtWidgets.QCheckBox()
            cell = QWidget()
            lay  = QHBoxLayout(cell)
            lay.setAlignment(Qt.AlignCenter)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(chk)
            self.tbl_sel.setCellWidget(r, 0, cell)
            self.tbl_sel.setItem(r, 1, QTableWidgetItem(str(row['Sat'])))
            self.tbl_sel.setItem(r, 2, QTableWidgetItem(str(row['Sys'])))
        self.tbl_sel.resizeRowsToContents()
        self.log('Satellite selection table populated', 'INFO')

    def compute_and_populate_error_budget(self, doptbl, rx):
        self.tbl_err.setRowCount(0)
        rows = []
        for i, row in doptbl.iterrows():
            stats = compute_error_budget_for_sat(row, rx)
            r = self.tbl_err.rowCount()
            self.tbl_err.insertRow(r)
            vals = [
                str(r + 1),
                str(row['Sat']),
                f"{stats['el']:.2f}",
                f"{stats['iono_m']:.3f}",
                f"{stats['tropo_m']:.3f}",
                f"{stats['clock_m']:.3f}",
                f"{stats['multipath_m']:.3f}",
                f"{stats['total_m']:.3f}",
            ]
            for col, v in enumerate(vals):
                self.tbl_err.setItem(r, col, QTableWidgetItem(v))
            rows.append(stats)
        self.tbl_err.resizeRowsToContents()
        self.processor.error_table = rows
        self.log('Error budget table populated', 'INFO')

    def _get_selected_sats(self):
        selected = []
        for r in range(self.tbl_sel.rowCount()):
            cell = self.tbl_sel.cellWidget(r, 0)
            if isinstance(cell, QWidget):
                chk = cell.findChild(QtWidgets.QCheckBox)
                if chk and chk.isChecked():
                    selected.append(self.tbl_sel.item(r, 1).text())
        return selected

    def _configure_sky_axes(self):
        self.ax.clear()
        self.ax.set_theta_zero_location('N')
        self.ax.set_theta_direction(-1)
        self.ax.set_rlim(0, 90)
        rticks = np.arange(0, 91, 15)
        self.ax.set_yticks(rticks)
        self.ax.set_yticklabels([str(90 - r) for r in rticks])
        self.ax.grid(color='#9FD6EE', linestyle='--', linewidth=0.7)
        self.ax.set_facecolor('#E6F9FF')
        self.canvas.draw_idle()

    def plot_sats_on_sky(self, satpos_df, rx_ecef, highlight=False):
        self.ax.clear()
        self._configure_sky_axes()
        for _, row in satpos_df.iterrows():
            if any(np.isnan(v) for v in [row['X'], row['Y'], row['Z']]):
                continue
            az, el, _ = ecef2aer_deg([row['X'], row['Y'], row['Z']], rx_ecef)
            zen   = 90.0 - el
            theta = math.radians(az)
            if highlight:
                self.ax.plot(theta, zen, marker='o', markersize=10,
                             markerfacecolor='#FF6B6B', markeredgecolor='#C93F3F')
                self.ax.text(theta, zen, f" {row['Sat']}",
                             fontsize=10, fontweight='bold', color='#22577A')
            else:
                self.ax.plot(theta, zen, marker='o', markersize=7,
                             markerfacecolor='#1E90FF', markeredgecolor='#0F5BB5',
                             alpha=0.9)
                self.ax.text(theta, zen, f" {row['Sat']}",
                             fontsize=9, color='#0F5BB5')
        self.canvas.draw_idle()

    # ══════════════════════════════════════════════════════════════════════════
    # EXPORT
    # ══════════════════════════════════════════════════════════════════════════

    def on_export_csv(self):
        if self.data['dop_table'].empty:
            QMessageBox.warning(self, 'No Data', 'Compute DOP first.')
            return
        rx       = self._rx_ecef()
        rows_out = []
        for i, row in self.data['dop_table'].iterrows():
            stats = compute_error_budget_for_sat(row, rx)
            rows_out.append({
                'S.No': i + 1,
                'Sat': row['Sat'], 'Sys': row['Sys'],
                'El_deg': row.get('El', ''), 'Az_deg': row.get('Az', ''),
                'PDOP': row['PDOP'], 'HDOP': row['HDOP'],
                'VDOP': row['VDOP'], 'GDOP': row['GDOP'],
                'Iono_m':  stats['iono_m'],
                'Tropo_m': stats['tropo_m'],
                'Clock_m': stats['clock_m'],
                'MP_m':    stats['multipath_m'],
                'Total_m': stats['total_m'],
            })
        df_out = pd.DataFrame(rows_out)
        fname, _ = QFileDialog.getSaveFileName(
            self, 'Save CSV', 'satellite_results.csv',
            'CSV files (*.csv);;All files (*.*)'
        )
        if not fname:
            return
        try:
            df_out.to_csv(fname, index=False)
            self.log(f'Exported CSV: {fname}  ({len(df_out)} rows)', 'OK')
            QMessageBox.information(self, 'Exported', f'Saved to {fname}')
        except Exception as e:
            QMessageBox.critical(self, 'Export Error', str(e))
            self.log(f'Export failed: {e}', 'ERR')

    def export_state(self):
        if self.data['dop_table'].empty:
            raise RuntimeError('Run DOP computation first.')
        rx  = self._rx_ecef()
        els = []
        for _, row in self.data['sat_positions'].iterrows():
            az, el, _ = ecef2aer_deg([row['X'], row['Y'], row['Z']], rx)
            els.append(el)
        return {
            'epoch_time':         self.data['obs_table']['UTC_Time'].iloc[0],
            'sat_positions_ecef': self.data['sat_positions'][['X','Y','Z']].to_numpy(),
            'elevation_deg':      np.array(els),
            'sat_ids':            self.data['sat_positions']['Sat'].tolist(),
            'constellation':      self.data['sat_positions']['Sys'].tolist(),
        }

    def export_state_clicked(self):
        try:
            state = self.export_state()
            for k, v in state.items():
                print(k, v.shape if isinstance(v, np.ndarray) else type(v))
            self.log('Debug state exported to terminal', 'INFO')
        except Exception as e:
            QMessageBox.warning(self, 'Export failed', str(e))