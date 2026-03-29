import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PyQt5.QtWidgets import QApplication, QFileDialog
from engine.io.rinex_obs import read_rinex_obs


# -----------------------------
# USER SELECTS RINEX FILE
# -----------------------------

app = QApplication(sys.argv)

obs_file, _ = QFileDialog.getOpenFileName(
    None,
    "Select RINEX Observation File",
    "",
    "RINEX files (*.obs *.rnx *.24o *.gz)"
)

if not obs_file:
    print("No file selected")
    sys.exit()

print("\nSelected file:", obs_file)


# -----------------------------
# LOAD OBSERVATION FILE
# -----------------------------

obs_df = read_rinex_obs(obs_file)

print("\nRaw OBS rows:", len(obs_df))

# ------------------------------------------------
# FILTER GPS SATELLITES ONLY (recommended first)
# ------------------------------------------------
obs_df = obs_df[obs_df["Sat"].str.startswith("G")]
# remove rows without numeric values
obs_df = obs_df.dropna(subset=["Value"])
# remove placeholder values (0,1,2,3 etc)
obs_df = obs_df[obs_df["Value"] > 1000]

# -----------------------------
# PIVOT OBSERVATIONS
# -----------------------------

pivot = obs_df.pivot_table(
    index=["UTC_Time", "Sat"],
    columns="ObsType",
    values="Value",
    aggfunc="first"
).reset_index()
pivot.columns.name = None

print("\nAvailable observation columns:")
print(list(pivot.columns))


# ------------------------------------------------
# FIND BEST SIGNAL PAIR (MAX DATA)
# ------------------------------------------------

pairs = []

for col in pivot.columns:

    if col.startswith("C"):

        suffix = col[1:]            # example C1L -> 1L
        phase = "L" + suffix        # expected phase

        if phase in pivot.columns:
            pairs.append((col, phase))


if not pairs:
    raise RuntimeError("No valid pseudorange/phase pairs found")


best_pair = None
best_count = 0


for pr, cp in pairs:

    subset = pivot[[pr, cp]].dropna()
    

    count = len(subset)

    print(f"Pair {pr} / {cp} -> {count} measurements")

    if count > best_count:
        best_count = count
        best_pair = (pr, cp)
if best_pair is None:
    raise RuntimeError(
        "No valid pseudorange / carrier phase pair found in this RINEX file."
    )

pr_col, cp_col = best_pair


print("\nSelected signals:")
print("Pseudorange:", pr_col)
print("Carrier phase:", cp_col)
print("Total usable measurements:", best_count)

# -----------------------------
# BUILD MEASUREMENT TABLE
# -----------------------------

meas = pivot[["UTC_Time", "Sat", pr_col, cp_col]].copy()

meas = meas.rename(columns={
    "UTC_Time": "epoch",
    "Sat": "sat",
    pr_col: "P1",
    cp_col: "L1"
})

meas = meas.dropna()

meas = meas.drop_duplicates(subset=["epoch", "sat"])


# -----------------------------
# OUTPUT
# -----------------------------

print("\nMeasurement Table Preview\n")
print(meas.head())

print("\nTotal measurements:", len(meas))
print("Total epochs:", meas["epoch"].nunique())
print("Satellites:", meas["sat"].unique())