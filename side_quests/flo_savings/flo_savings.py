#!/usr/bin/env python3
"""Visualize when the FLO (Forward Looking Optimizer) was active over time."""

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HOUSE_ALIAS = "elm"
RESULTS_DIR = Path(__file__).parent / "results"

cop_params_per_house = {
    "beech": {
        "cop_intercept": 1.5,
        "cop_oat_coeff": 0.015,
        "cop_min": 1.5,
        "cop_min_oat_f": 0,
    },
    "oak": {
        "cop_intercept": 1.5,
        "cop_oat_coeff": 0.015,
        "cop_min": 1.5,
        "cop_min_oat_f": 0,
    },
    "fir": {
        "cop_intercept": 1.1,
        "cop_oat_coeff": 0.01,
        "cop_min": 1.1,
        "cop_min_oat_f": 0,
    },
    "maple": {
        "cop_intercept": 1.0,
        "cop_oat_coeff": 0.02,
        "cop_min": 1.0,
        "cop_min_oat_f": 0,
    },
    "elm": {
        "cop_intercept": 1.3,
        "cop_oat_coeff": 0.013,
        "cop_min": 1.3,
        "cop_min_oat_f": 0,
    },
}

WIND_OAT_REFERENCE_F = 65
COP_INTERCEPT = cop_params_per_house[HOUSE_ALIAS]["cop_intercept"]  
COP_OAT_COEFF = cop_params_per_house[HOUSE_ALIAS]["cop_oat_coeff"]
COP_MIN = cop_params_per_house[HOUSE_ALIAS]["cop_min"]
COP_MIN_OAT_F = cop_params_per_house[HOUSE_ALIAS]["cop_min_oat_f"]



def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "hour_start" not in df.columns or "flo" not in df.columns:
        raise ValueError("CSV must contain 'hour_start' and 'flo' columns")

    df["hour_start"] = pd.to_datetime(df["hour_start"])
    df = df.drop_duplicates(subset=["hour_start"], keep="last")
    df = df.sort_values("hour_start").reset_index(drop=True)
    df["flo"] = df["flo"].map({"True": True, "False": False, True: True, False: False})
    df["flo"] = df["flo"].fillna(False).astype(bool)
    df = add_load_predictions(df)
    return add_baseline_columns(df)


def fit_dist_kwh_model(df: pd.DataFrame) -> tuple[float, float, float]:
    """Fit dist_kwh ~ intercept + beta_0*oat + gamma*ws*(65-oat) using all valid rows."""
    reg_df = df.dropna(subset=["dist_kwh", "oat_f", "ws_mph"])
    if reg_df.empty:
        raise ValueError("No rows with dist_kwh, oat_f, and ws_mph available for load regression")

    oat = reg_df["oat_f"].to_numpy(dtype=float)
    ws = reg_df["ws_mph"].to_numpy(dtype=float)
    y = reg_df["dist_kwh"].to_numpy(dtype=float)
    wind_term = ws * (WIND_OAT_REFERENCE_F - oat)
    design_matrix = np.column_stack([np.ones(len(reg_df)), oat, wind_term])
    intercept, beta_0, gamma = np.linalg.lstsq(design_matrix, y, rcond=None)[0]
    return float(intercept), float(beta_0), float(gamma)


def add_load_predictions(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"dist_kwh", "oat_f", "ws_mph", "hp_kwh_th"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"CSV must contain columns: {', '.join(sorted(missing_columns))}")

    intercept, beta_0, gamma = fit_dist_kwh_model(df)
    print(
        "Load model: "
        f"dist_kwh = {intercept:.3f} + {beta_0:.3f}*oat + {gamma:.5f}*ws*({WIND_OAT_REFERENCE_F}-oat)"
    )

    df = df.copy()
    wind_term = df["ws_mph"] * (WIND_OAT_REFERENCE_F - df["oat_f"])
    df["pred_dist_kwh"] = intercept + beta_0 * df["oat_f"] + gamma * wind_term

    dist_sum = df["dist_kwh"].sum()
    if dist_sum == 0:
        raise ValueError("dist_kwh sum is zero, cannot scale load prediction")

    hp_th_sum = df["hp_kwh_th"].sum()
    hp_th_to_dist_ratio = hp_th_sum / dist_sum
    df["load_pred"] = df["pred_dist_kwh"] * hp_th_to_dist_ratio
    print(
        f"Load scaling ratio: {hp_th_to_dist_ratio:.3f} "
        f"(sum hp_kwh_th={hp_th_sum:,.1f} / sum dist_kwh={dist_sum:,.1f})"
    )

    return df


def compute_cop(oat_f: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(
            oat_f < COP_MIN_OAT_F,
            COP_MIN,
            COP_INTERCEPT + COP_OAT_COEFF * oat_f,
        ),
        index=oat_f.index,
    )


def add_baseline_columns(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"oat_f", "load_pred", "lmp_usd_per_mwh"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"CSV must contain columns: {', '.join(sorted(missing_columns))}")

    df = df.copy()
    df["cop"] = compute_cop(df["oat_f"])
    df["baseline_cost_usd"] = df["load_pred"] / df["cop"] * df["lmp_usd_per_mwh"] / 1000
    return df


def flo_periods(df: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return (start, end) intervals where FLO was running."""
    periods = []
    start = None
    last_end = None

    for _, row in df.iterrows():
        hour_start = row["hour_start"]
        hour_end = hour_start + pd.Timedelta(hours=1)
        if row["flo"]:
            if start is None:
                start = hour_start
            last_end = hour_end
        elif start is not None:
            periods.append((start, last_end))
            start = None
            last_end = None

    if start is not None:
        periods.append((start, last_end))

    return periods


def electricity_cost_when_flo(df: pd.DataFrame) -> float:
    if "hp_kwh_el" not in df.columns or "lmp_usd_per_mwh" not in df.columns:
        raise ValueError("CSV must contain 'hp_kwh_el' and 'lmp_usd_per_mwh' columns")

    flo_df = df[df["flo"]]
    hourly_cost_usd = flo_df["hp_kwh_el"] * flo_df["lmp_usd_per_mwh"] / 1000
    return float(hourly_cost_usd.sum())


def baseline_cost_when_flo(df: pd.DataFrame) -> float:
    flo_df = df[df["flo"]]
    return float(flo_df["baseline_cost_usd"].sum())


def plot_flo_timeline(
    df: pd.DataFrame,
    total_elec_cost_when_flo: float,
    total_baseline_cost_when_flo: float,
    output_path: Path | None = None,
) -> None:
    periods = flo_periods(df)
    data_start = df["hour_start"].iloc[0]
    data_end = df["hour_start"].iloc[-1] + pd.Timedelta(hours=1)

    fig, ax = plt.subplots(figsize=(12, 1.8))

    ax.axvspan(data_start, data_end, ymin=0.25, ymax=0.75, color="#e0e0e0", alpha=0.8)
    for start, end in periods:
        ax.axvspan(start, end, ymin=0.25, ymax=0.75, color="#2e7d32", alpha=0.85)

    ax.set_xlim(data_start, data_end)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title(
        f"$\\mathbf{{{HOUSE_ALIAS.capitalize()}}}$\n"
        f"Actual cost when running FLO: {total_elec_cost_when_flo:,.2f} USD\n"
        f"Estimated cost without storage: {total_baseline_cost_when_flo:,.2f} USD\n"
        f"Estimated savings: {100 * (total_baseline_cost_when_flo - total_elec_cost_when_flo) / total_baseline_cost_when_flo:.1f}%"
    )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")

    plt.tight_layout()

    if output_path is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = RESULTS_DIR / f"{HOUSE_ALIAS}_flo_timeline.png"

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close(fig)


def main() -> None:
    default_csv = Path(__file__).parent / f"data/{HOUSE_ALIAS}_electricity_use_2025-11-01-00-00-2026-06-05-07-46.csv"

    parser = argparse.ArgumentParser(description="Plot FLO active periods from hourly CSV data.")
    parser.add_argument("csv", nargs="?", type=Path, default=default_csv, help="Hourly CSV input file")
    parser.add_argument("-o", "--output", type=Path, help="Save plot to a custom file path")
    args = parser.parse_args()

    df = load_data(args.csv)
    flo_hours = int(df["flo"].sum())
    total_hours = len(df)
    print(f"Loaded {total_hours} hours ({df['hour_start'].iloc[0]} to {df['hour_start'].iloc[-1]})")
    print(f"FLO active: {flo_hours} hours ({100 * flo_hours / total_hours:.1f}%)")

    total_elec_cost_when_flo = electricity_cost_when_flo(df)
    total_baseline_cost_when_flo = baseline_cost_when_flo(df)
    print(f"Electricity bought when running FLO: ${total_elec_cost_when_flo:,.2f}")
    print(f"Baseline (no storage/smart control) when running FLO: ${total_baseline_cost_when_flo:,.2f}")

    plot_flo_timeline(df, total_elec_cost_when_flo, total_baseline_cost_when_flo, args.output)


if __name__ == "__main__":
    main()
