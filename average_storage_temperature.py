"""
Compute average_storage_temperature from tank depth columns and apply (x-32)*5/9*10.
Reads 1-second CSV with timestamps and tank{i}-depth{j} columns (i,j in [1,2,3]).
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Find CSV matching beech_{i}s_*.csv (e.g. beech_1s_, beech_2s_, ...)
script_dir = Path(__file__).resolve().parent
matches = sorted(script_dir.glob("beech_*s_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
if not matches:
    raise FileNotFoundError(f'No file matching "beech_*s_*.csv" in {script_dir}')
CSV_PATH = matches[0]
print(f"Using: {CSV_PATH.name}")

# First row is a filename, not the header
df = pd.read_csv(CSV_PATH, skiprows=[0])
df["timestamps"] = pd.to_datetime(df["timestamps"])

# Build list of tank{i}-depth{j} columns for i,j in [1,2,3]
tank_depth_cols = [
    f"tank{i}-depth{j}"
    for i in range(1, 4)
    for j in range(1, 4)
]

# Keep only columns that exist in the dataframe
available_cols = [c for c in tank_depth_cols if c in df.columns]
if not available_cols:
    raise ValueError(
        f"None of the tank depth columns {tank_depth_cols} found in CSV. "
        f"Available columns: {list(df.columns)}"
    )

# Average of all tank-depth columns
df["average_storage_temperature"] = df[available_cols].mean(axis=1)

# Apply (x-32)*5/9*10
df["average_storage_temperature"] = df["average_storage_temperature"]/ 100

# Optional: write result (uncomment to save)
# df.to_csv("beech_1s_with_avg_storage_temp.csv", index=False)

# Find relay3 and relay9 columns
def find_relay_col(needle: str) -> str:
    cols = [c for c in df.columns if needle in c.lower()]
    if not cols:
        raise ValueError(f'No column containing "{needle}" found. Columns: {list(df.columns)}')
    return cols[0]

relay3_col = find_relay_col("relay3")
relay9_col = find_relay_col("relay9")
relay3_pulled = (df[relay3_col].astype(int) == 1)
relay9_pulled = (df[relay9_col].astype(int) == 1)

# System state: StoreCharge (relay3), StoreDischarge (relay9), else Store Idle. If both pulled, show Charge.
df["store_charge"] = relay3_pulled.astype(int)
df["store_discharge"] = (relay9_pulled & ~relay3_pulled).astype(int)

print(f"Used columns: {available_cols}")
print(f"Relay3 column: {relay3_col}, Relay9 column: {relay9_col}")
print(df[["timestamps"] + available_cols + ["average_storage_temperature", "store_charge", "store_discharge"]].head(10))

# Two subplots: temperature and system state
fig, (ax_temp, ax_relay) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, height_ratios=[1.5, 0.5])

ax_temp.plot(df["timestamps"], df["average_storage_temperature"], color="steelblue")
ax_temp.set_ylabel("Average storage temperature")
ax_temp.set_title("Average storage temperature over time")
ax_temp.grid(True, alpha=0.3)

# Orange = StoreCharge, green = StoreDischarge, nothing = Store Idle
ax_relay.fill_between(df["timestamps"], 0, df["store_charge"], color="tab:orange", alpha=0.6, step="post")
ax_relay.fill_between(df["timestamps"], 0, df["store_discharge"], color="green", alpha=0.6, step="post")
ax_relay.set_ylabel("System state")
ax_relay.set_xlabel("Time")
ax_relay.set_ylim(-0.1, 1.1)
ax_relay.set_yticks([])
ax_relay.grid(True, alpha=0.3)

fig.autofmt_xdate()
plt.tight_layout()
plt.show()
