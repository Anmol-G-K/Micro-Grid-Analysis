
"""
Day 2 - Seasonal & Daily Patterns (Optimized for 70M+ rows)

Efficient temporal pattern extraction for very large datasets:
- Heatmaps: Day vs Hour for Load and PV
- Average daily   profiles (Load, PV, weekday vs weekend, summer vs winter)
- Distribution plots (Load, PV, Battery)

OPTIMIZATION STRATEGIES:
1. Polars lazy evaluation for initial processing
2. Downsampling to daily/hourly summaries before visualization
3. Efficient group_by_dynamic for large time series
4. Save results (plots + summary stats) to disk

Dependencies: polars, pandas, numpy, matplotlib, seaborn, plotly
"""

import polars as pl
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import json
import warnings
from datetime import datetime
# warnings.filterwarnings("ignore")

# =========================
# CONFIGURATION
# =========================
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "cleaned_power_dataset.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "plots" / "patterns"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = OUTPUT_DIR / "patterns_summary.json"

# Target columns
TARGET_COLS = [
    "battery_active_power",
    "pvpcs_active_power",
    "ge_body_active_power",
    "ge_active_power",
    "fc_active_power",
    "island_mode_mccb_active_power",
]

# =========================
# UTILITY FUNCTIONS
# =========================

def detect_timestamp_column(lf: pl.LazyFrame) -> str:
    """Detect timestamp column in dataset."""
    schema = lf.collect_schema()
    candidates = [c for c in schema.names() if "time" in c.lower() or "date" in c.lower()]
    if candidates:
        return candidates[0]
    for col, dtype in schema.items():
        if dtype == pl.Datetime:
            return col
    raise ValueError("No timestamp column found!")

def ensure_datetime_column(lf: pl.LazyFrame, ts_col: str) -> pl.LazyFrame:
    """Ensure timestamp column is proper datetime."""
    schema = lf.collect_schema()
    if schema.get(ts_col) == pl.Datetime:
        return lf
    if schema.get(ts_col) == pl.Utf8:
        return lf.with_columns(pl.col(ts_col).str.strptime(pl.Datetime, "%Y/%m/%d %H:%M:%S", strict=False))
    return lf

# =========================
# MAIN ANALYSIS FUNCTION
# =========================

def main():
    print("Starting Day 2 - Seasonal & Daily Patterns analysis...")
    print(f"Data path: {DATA_PATH}")

    # Step 1: Load dataset
    lf = pl.scan_parquet(DATA_PATH)
    ts_col = detect_timestamp_column(lf)
    lf = ensure_datetime_column(lf, ts_col)

    available_cols = [c for c in TARGET_COLS if c in lf.collect_schema().names()]
    if not available_cols:
        raise ValueError("No target columns found in dataset!")

    print(f"Available columns: {available_cols}")

    # Step 2: Add time features
    df = (
        lf.select([ts_col] + available_cols)
          .filter(pl.col(ts_col).is_not_null())
          .with_columns([
              pl.col(ts_col).dt.hour().alias("hour"),
              pl.col(ts_col).dt.day().alias("day"),
              pl.col(ts_col).dt.weekday().alias("weekday"),
              pl.col(ts_col).dt.month().alias("month"),
              pl.col(ts_col).dt.date().alias("date"),
          ])
          .collect()
    )

    df_pd = df.to_pandas()
    df_pd = df_pd.set_index(ts_col).sort_index()

    # =========================
    # HEATMAPS
    # =========================
    print("Generating heatmaps...")

    for col in ["ge_active_power", "pvpcs_active_power"]:
        if col not in df_pd.columns:
            continue
        pivot = df_pd.pivot_table(values=col, index="date", columns="hour", aggfunc="mean")
        plt.figure(figsize=(18, 8))   # make it wider
        sns.heatmap(pivot, cmap="viridis" if col == "ge_active_power" else "inferno")
        plt.xticks(rotation=45)   # tilt x labels
        plt.title(f"{col} Heatmap (Day vs Hour)")
        plt.xlabel("Hour of Day")
        plt.ylabel("Day")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"heatmap_{col}.png", dpi=150)
        plt.close()

    # =========================
    # AVERAGE DAILY PROFILES
    # =========================
    print("Generating daily average profiles...")

    avg_profile_paths = {}
    weekday_weekend_paths = {}
    seasonal_paths = {}

    for col in available_cols:
        label = col.replace("_", " ").title()

        # --- Daily average profile
        plt.figure(figsize=(10, 5))
        df_pd.groupby("hour")[col].mean().plot(label=label, linewidth=2)
        plt.title(f"Average Daily Profile – {label}")
        plt.xlabel("Hour of Day")
        plt.ylabel("Power")
        plt.grid(True)
        plt.tight_layout()
        filename = f"avg_daily_{col}.png"
        plt.savefig(OUTPUT_DIR / filename, dpi=150)
        plt.close()
        avg_profile_paths[col] = filename

        # --- Weekday vs Weekend
        plt.figure(figsize=(10, 5))
        df_pd[df_pd["weekday"] < 5].groupby("hour")[col].mean().plot(label="Weekday", linewidth=2)
        df_pd[df_pd["weekday"] >= 5].groupby("hour")[col].mean().plot(label="Weekend", linewidth=2)
        plt.title(f"Weekday vs Weekend – {label}")
        plt.xlabel("Hour of Day")
        plt.ylabel("Power")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        filename = f"weekday_weekend_{col}.png"
        plt.savefig(OUTPUT_DIR / filename, dpi=150)
        plt.close()
        weekday_weekend_paths[col] = filename

        # --- Summer vs Winter
        plt.figure(figsize=(10, 5))
        summer = df_pd[df_pd["month"].isin([6, 7, 8])]
        winter = df_pd[df_pd["month"].isin([12, 1, 2])]
        summer.groupby("hour")[col].mean().plot(label="Summer", linewidth=2)
        winter.groupby("hour")[col].mean().plot(label="Winter", linewidth=2)
        plt.title(f"Summer vs Winter – {label}")
        plt.xlabel("Hour of Day")
        plt.ylabel("Power")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        filename = f"summer_winter_{col}.png"
        plt.savefig(OUTPUT_DIR / filename, dpi=150)
        plt.close()
        seasonal_paths[col] = filename
    # =========================
    # DISTRIBUTION PLOTS
    # =========================
    print("Generating distributions...")

    for col in available_cols:
        plt.figure(figsize=(8, 5))
        sns.histplot(df_pd[col].dropna(), kde=True, bins=100)
        plt.title(f"Distribution of {col}")
        plt.xlabel("Power")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"dist_{col}.png", dpi=150)
        plt.close()

    # =========================
    # SUMMARY JSON
    # =========================
    dataset_size_mb = DATA_PATH.stat().st_size / (1024 ** 2)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_rows": len(df_pd),
        "n_cols": len(df_pd.columns),
        "columns": list(df_pd.columns),
        "time_range": [str(df_pd.index.min()), str(df_pd.index.max())],
        "dataset_size_mb": round(dataset_size_mb, 2),
        "missing_percentages": {
            col: round(df_pd[col].isna().mean() * 100, 2) for col in df_pd.columns
        },
        "column_statistics": {
            col: {
                "min": float(np.nanmin(df_pd[col])),
                "max": float(np.nanmax(df_pd[col])),
                "mean": float(np.nanmean(df_pd[col]))
            } for col in available_cols
        },
        "outputs": {
            "heatmaps": {
                col: f"heatmap_{col}.png" for col in ["ge_active_power", "pvpcs_active_power"] if col in available_cols
            },
            "average_profiles": avg_profile_paths,
            "weekday_vs_weekend": weekday_weekend_paths,
            "summer_vs_winter": seasonal_paths,
            "distributions": {
                col: f"dist_{col}.png" for col in available_cols
            }
        }
    }


    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Day 2 analysis complete. Results saved in {OUTPUT_DIR}")
    print(f"Summary file: {SUMMARY_FILE}")

if __name__ == "__main__":
    main()
