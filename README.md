# SatNav
GPS + IRNSS Precise Point Positioning with GF Dynamic Test cycle slip detection and correction — Python/PyQt5
# GNSS PPP — GPS + IRNSS Cycle Slip Detection & Correction

A Python/PyQt5 post-processing tool for dual-constellation GNSS 
(GPS + IRNSS/NavIC) with Precise Point Positioning and cycle slip 
detection based on the GF Dynamic Test.

## Reference Algorithm

Wang, J. & Huang, D. (2023). *Dual-frequency GPS Cycle Slip Detection 
and Repair Based on Dynamic Test*. KSCE Journal of Civil Engineering, 
27(12), 5329–5337. https://doi.org/10.1007/s12205-023-0388-2

## Features

- RINEX 3 observation and navigation file parsing (GPS + IRNSS)
- Geometry-Free (GF) Dynamic Test cycle slip detection — all three 
  equations from the reference paper (Eq. 3, 6, 7, 8)
- Melbourne-Wübbena (MW) combination with L5-based wide-lane fallback
  when L1 pseudorange is absent
- Special slip type detection: (9,7) and (77,60) combinations
- Welford online running statistics for adaptive m1/m2 thresholds
- Sampling-rate-adaptive detection threshold (scaled for 30 s data)
- Slip correction via Q[a,b] directed rounding function
- Satellite geometry: DOP computation, skyplot, error budget
- PyQt5 interactive dashboard — 4 tabs, 7 visualisation panels
- Background-threaded detection (GUI stays responsive)
- Validated against gLab v6.0.0 on IISC station DOY 008, 2026

## Project Structure
```
gnss_project/
├── engine/
│   ├── io/               # RINEX 3 OBS + NAV parsers
│   ├── geometry/         # Orbit, DOP, coordinates
│   ├── models/           # Error budget (iono, tropo, clock, MP)
│   ├── measurements/     # Signal extraction (GPS + IRNSS)
│   ├── cycle_slip/       # GF+MW detection and correction
│   │   ├── mw_combination.py   # GF, MW, WL5 combinations
│   │   ├── mw_detector.py      # Combined GF+MW detection
│   │   ├── arc_manager.py      # Arc tracking, Welford stats
│   │   └── slip_corrector.py   # Integer slip correction
│   └── processor.py      # Pipeline orchestration
└── gui/
    └── main_window.py    # PyQt5 dashboard
```

## Test Data

IISC00IND — Indian Institute of Science, Bangalore (IGS station)
DOY 008, 2026 | 30-second interval | GPS + IRNSS

## Status

| Phase | Task | Status |
|-------|------|--------|
| 1 | GF cycle slip detection | ✅ Complete |
| 2 | GF + MW correction | ✅ Complete |
| 3 | Kalman filter PPP | 🔲 In progress |
| 4 | LAMBDA integer ambiguity resolution | 🔲 Planned |

## Requirements
```
Python >= 3.10
PyQt5
NumPy
Pandas
Matplotlib
```

## Reference

Wang, J., & Huang, D. (2023). Dual-frequency GPS Cycle Slip Detection 
and Repair Based on Dynamic Test. *KSCE Journal of Civil Engineering*, 
*27*(12), 5329–5337. https://doi.org/10.1007/s12205-023-0388-2
