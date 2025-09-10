
# File: d5_forecasting_prep.py
# Description: Day 5 - Forecasting Preparation & Baseline Models (Polars Optimized)

import polars as pl
import numpy as np
import json
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error

# =====================
# CONFIG
# =====================
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "cleaned_power_dataset.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "plots" / "forecasting"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = OUTPUT_DIR / "forecast_ready.json"
RESOLUTION = "15m"  # polars format

# =====================
# HELPERS
# =====================
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / np.clip(denominator, 1e-6, None)
    return np.mean(diff) * 100


def add_calendar_features(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns([
            df["timestamp"].dt.hour().alias("hour_of_day"),
            df["timestamp"].dt.weekday().alias("day_of_week_id"),
            ((df["timestamp"].dt.month() % 12) // 3).alias("season_id")
        ])
        .with_columns([
            pl.when(pl.col("day_of_week_id") == 0).then(pl.lit("Mon"))
              .when(pl.col("day_of_week_id") == 1).then(pl.lit("Tue"))
              .when(pl.col("day_of_week_id") == 2).then(pl.lit("Wed"))
              .when(pl.col("day_of_week_id") == 3).then(pl.lit("Thu"))
              .when(pl.col("day_of_week_id") == 4).then(pl.lit("Fri"))
              .when(pl.col("day_of_week_id") == 5).then(pl.lit("Sat"))
              .when(pl.col("day_of_week_id") == 6).then(pl.lit("Sun"))
              .otherwise(pl.lit("Unknown"))
              .alias("day_of_week"),
            pl.when(pl.col("season_id") == 0).then(pl.lit("winter"))
              .when(pl.col("season_id") == 1).then(pl.lit("spring"))
              .when(pl.col("season_id") == 2).then(pl.lit("summer"))
              .when(pl.col("season_id") == 3).then(pl.lit("fall"))
              .otherwise(pl.lit("unknown"))
              .alias("season")
        ])
        .drop(["season_id", "day_of_week_id"])
    )



def add_lag_features(df: pl.DataFrame, cols, lags=[1]) -> pl.DataFrame:
    for col in cols:
        for lag in lags:
            df = df.with_columns(pl.col(col).shift(lag).alias(f"{col}_t-{lag}"))
    return df

# =====================
# MAIN PIPELINE
# =====================
def main():
    print("=== Day 5: Forecasting Preparation (Polars Optimized) ===")

    # Load dataset (lazy scan for speed)
    df = pl.scan_parquet(DATA_PATH).collect()

    if "timestamp" not in df.columns:
        raise ValueError("Dataset must contain a 'timestamp' column.")
    df = df.with_columns(pl.col("timestamp").cast(pl.Datetime))

    # Downsample to 15-min
    df_resampled = (
        df.sort("timestamp")
            .group_by_dynamic("timestamp", every=RESOLUTION)
            .agg([
                pl.mean("ge_active_power").alias("load_kw"),
                # pl.mean("island_mode_mccb_active_power").alias("load_kw"),
                pl.mean("pvpcs_active_power").alias("pv_kw"),
            ])
        .sort("timestamp")
    )

    # Add lag + calendar features
    df_resampled = add_lag_features(df_resampled, ["load_kw", "pv_kw"], lags=[1])

    df_resampled = add_calendar_features(df_resampled)

    # Drop NA (from lag)
    df_resampled = df_resampled.drop_nulls()

    load_stats = {
    "min": df_resampled["load_kw"].min(),
    "max": df_resampled["load_kw"].max(),
    "mean": df_resampled["load_kw"].mean(),
    "median": df_resampled["load_kw"].median(),
    "std_dev": df_resampled["load_kw"].std(),
    "percent_negative": ((df_resampled["load_kw"] < 0).sum() / df_resampled.height) * 100
    }

    pv_stats = {
        "min": df_resampled["pv_kw"].min(),
        "max": df_resampled["pv_kw"].max(),
        "mean": df_resampled["pv_kw"].mean(),
        "median": df_resampled["pv_kw"].median(),
        "std_dev": df_resampled["pv_kw"].std(),
        "percent_zero": ((df_resampled["pv_kw"] == 0).sum() / df_resampled.height) * 100
    }

    # Convert to pandas (only where statsmodels needs it)
    pdf = df_resampled.to_pandas()

    # Train/test split (last 20% test)
    split_idx = int(len(pdf) * 0.8)
    train, test = pdf.iloc[:split_idx], pdf.iloc[split_idx:]

    # =====================
    # Baseline Models
    # =====================
    metrics = {}

    # Persistence
    y_true = test["load_kw"].values
    y_pred_persist = test["load_kw_t-1"].values
    metrics["persistence"] = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred_persist))),
        "mae": float(mean_absolute_error(y_true, y_pred_persist)),
        "smape": float(smape(y_true, y_pred_persist)),
    }

    # Holt-Winters Exponential Smoothing
    model = ExponentialSmoothing(
        train["load_kw"],
        trend="add",
        seasonal="add",
        seasonal_periods=96,  # 15-min data → 96 intervals/day
    ).fit()
    y_pred_es = model.forecast(len(test))
    metrics["exponential_smoothing"] = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred_es))),
        "mae": float(mean_absolute_error(y_true, y_pred_es)),
        "smape": float(smape(y_true, y_pred_es)),
    }

    # =====================
    # Build JSON
    # =====================
    json_output = {
        "meta": {
            "dataset_id": "microgrid_day5_forecast_ready",
            "description": "15-min resolution dataset with lag and calendar features for forecasting tasks",
            "source_file": str(DATA_PATH.name),
            "resolution_minutes": 15,
            "time_range": {
                "start": str(pdf["timestamp"].min()),
                "end": str(pdf["timestamp"].max()),
            },
            "features_included": list(pdf.columns),
        },
        "time_index": [str(ts) for ts in pdf["timestamp"].head(3)],
        "data": {
            "load_kw": pdf["load_kw"].head(3).tolist(),
            "pv_kw": pdf["pv_kw"].head(3).tolist(),
            "lag_features": {
                "load_kw_t-1": pdf["load_kw_t-1"].head(3).tolist(),
                "pv_kw_t-1": pdf["pv_kw_t-1"].head(3).tolist(),
            },
            "calendar_features": {
                "hour_of_day": pdf["hour_of_day"].head(3).tolist(),
                "day_of_week": pdf["day_of_week"].head(3).tolist(),
                "season": pdf["season"].head(3).tolist(),
            },
        },
        "baseline_models": metrics,
        "statistics": {
            "load_kw": {k: float(v) for k, v in load_stats.items()},
            "pv_kw": {k: float(v) for k, v in pv_stats.items()},
        },

    }

    # Save JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(json_output, f, indent=2)

    print(f"✅ JSON saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
