# run_soc_sim.py
# Physics-calibrated + LightGBM residual hybrid SOC modelling with lag features
# Saves JSON summaries + plots per run (monthly or full)

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import itertools
import time

# ML libs
try:
    import lightgbm as lgb
    ML_BACKEND = "lightgbm"
except Exception:
    # fallback to sklearn hist gradient if LightGBM not available
    from sklearn.ensemble import HistGradientBoostingRegressor
    ML_BACKEND = "sklearn"

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Polars optional
use_polars = False
try:
    import polars as pl
    use_polars = True
except Exception:
    use_polars = False

# Paths
BASE = Path(__file__).resolve().parent.parent
DATA_PATH = BASE / "data" / "combined.parquet"
OUTPUT_BASE = BASE / "plots" / "soc_sim"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# Default simulation hyperparams (can be tuned by calibration)
ETA_CH_BASE = 0.95
ETA_DIS_BASE = 0.95
CAPACITY_KWH = 50.0
P_MAX = 50.0
INITIAL_SOC_PCT = 50.0
NOISE_THRESHOLD_KW = 1e-3

# Degradation (kept simple)
BETA_CYCLE = 0.0005
GAMMA_CAL = 1e-5

# GRID SEARCH RANGES for calibration
CAL_CAP_RANGE = [20.0, 25.0, 50.0, 75.0, 100.0]      # kWh
CAL_ETA_CH_RANGE = [0.9, 0.93, 0.95, 0.98]
CAL_ETA_DIS_RANGE = [0.9, 0.93, 0.95, 0.98]

# Lag feature configuration (in timesteps)
DEFAULT_LAGS = [1, 3, 6, 24]  # 1-step, 3-step, 6-step, 24-step lags

# Model params
LGB_PARAMS = {"n_estimators": 500, "learning_rate": 0.05, "verbosity": -1}
SKLEARN_PARAMS = {"max_iter": 200}

# MODE
MODE = "B"  # "A" full dataset, "B" monthly


def load_dataset(cols_needed=None):
    if cols_needed is None:
        cols_needed = [
            "timestamp",
            "battery_state_of_charge",
            "battery_charging_rate",
            "battery_discharging_rate",
            "solar_pv_output",
            "grid_load_demand",
            "temperature",
        ]
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    if use_polars:
        lf = pl.scan_parquet(DATA_PATH)
        # attempt to cast timestamp if string
        try:
            lf = lf.with_columns(pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias("timestamp"))
        except Exception:
            # ignore if already datetime or different format
            pass
        available = [c for c in lf.collect_schema().names() if c in cols_needed]
        pl_df = lf.select(available).collect()
        df = pl_df.to_pandas()
    else:
        df = pd.read_parquet(DATA_PATH, columns=cols_needed)
    # normalize index, ensure datetime index
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        # infer freq: if irregular, we will resample to the most common interval
        df = df.set_index("timestamp").sort_index()
    else:
        # if no timestamp column, assume index is datetime-like already
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.sort_index()

    # compute median timestep in seconds and set DELTA_T hours
    diffs = df.index.to_series().diff().dropna()
    if len(diffs) == 0:
        raise RuntimeError("Not enough timestamps to infer timestep")
    most_common = diffs.mode().iloc[0]
    delta_seconds = most_common.total_seconds()
    DELTA_T_HOURS = delta_seconds / 3600.0

    # Resample to the inferred freq (fill gaps by forward/backfill minimally)
    freq_str = pd.Timedelta(seconds=int(delta_seconds)).to_pytimedelta()
    # Use pandas resample with median interval string
    # Build resample rule
    if delta_seconds % 3600 == 0:
        rule = f"{int(delta_seconds // 3600)}H"
    elif delta_seconds % 60 == 0:
        rule = f"{int(delta_seconds // 60)}T"
    else:
        rule = f"{int(delta_seconds)}S"
    try:
        df = df.resample(rule).mean()
    except Exception:
        # if resample fails, keep as-is
        pass

    # ensure the battery columns exist; fill missing with 0 or NaN as appropriate
    for c in ["battery_charging_rate", "battery_discharging_rate", "solar_pv_output", "grid_load_demand", "temperature", "battery_state_of_charge"]:
        if c not in df.columns:
            df[c] = np.nan

    # compute net battery power (positive = discharge)
    df["battery_net_power"] = df["battery_discharging_rate"].fillna(0.0) - df["battery_charging_rate"].fillna(0.0)

    return df, DELTA_T_HOURS


def physics_soc(df, capacity_kwh, eta_ch, eta_dis, delta_t_hours, use_pct=True):
    """
    Run a physics SOC simulation.
    - df must contain battery_charging_rate and battery_discharging_rate (kW)
    - returns a pd.Series of SOC in percent aligned to df.index
    - use_pct: SOC stored in percent (True)
    """
    n = len(df)
    soc = np.zeros(n)
    if "battery_state_of_charge" in df.columns and not df["battery_state_of_charge"].isna().all():
        soc[0] = df["battery_state_of_charge"].iloc[0]
    else:
        soc[0] = INITIAL_SOC_PCT

    cap_now = capacity_kwh
    cycles = 0.0
    # treat small noise
    p_ch_series = df["battery_charging_rate"].fillna(0.0).to_numpy()
    p_dis_series = df["battery_discharging_rate"].fillna(0.0).to_numpy()

    for i in range(1, n):
        p_ch = max(0.0, p_ch_series[i])
        p_dis = max(0.0, p_dis_series[i])
        e_ch = eta_ch * p_ch * delta_t_hours
        e_dis = eta_dis * p_dis * delta_t_hours
        delta_E = e_ch - e_dis  # kWh
        soc[i] = soc[i-1] + (delta_E / cap_now) * 100.0
        soc[i] = min(100.0, max(0.0, soc[i]))
        # update cycles and capacity fade (simple)
        if (p_ch + p_dis) > 0:
            cycles += (p_ch + p_dis) * delta_t_hours / (2.0 * cap_now)
        cap_now = capacity_kwh * max(0.1, (1 - BETA_CYCLE * cycles) * (1 - GAMMA_CAL * i))  # never go below 10% capacity

    return pd.Series(soc, index=df.index)


def calibrate_physics(df, delta_t_hours, cap_grid=CAL_CAP_RANGE, eta_ch_grid=CAL_ETA_CH_RANGE, eta_dis_grid=CAL_ETA_DIS_RANGE):
    """
    Simple grid-search calibration of capacity and efficiencies.
    Minimizes RMSE between physics SOC and actual SOC on available valid points.
    """
    best = None
    actual = df["battery_state_of_charge"].to_numpy()
    valid_mask = ~np.isnan(actual)
    if valid_mask.sum() < 10:
        raise RuntimeError("Not enough actual SOC values to calibrate.")
    start = time.time()
    combos = list(itertools.product(cap_grid, eta_ch_grid, eta_dis_grid))
    for cap, ech, edi in combos:
        sim = physics_soc(df, cap, ech, edi, delta_t_hours)
        sim_vals = sim.to_numpy()[valid_mask]
        actual_vals = actual[valid_mask]
        rmse = np.sqrt(np.mean((sim_vals - actual_vals) ** 2))
        if best is None or rmse < best[0]:
            best = (rmse, cap, ech, edi)
    elapsed = time.time() - start
    print(f"Calibration done in {elapsed:.1f}s. Best RMSE={best[0]:.3f} @ cap={best[1]}, eta_ch={best[2]}, eta_dis={best[3]}")
    return {"rmse": float(best[0]), "capacity_kwh": float(best[1]), "eta_ch": float(best[2]), "eta_dis": float(best[3])}


def make_features(df, lags=DEFAULT_LAGS):
    """
    Generate lag & rolling features for ML:
    - keep original instantaneous features
    - lags of charging/discharging, net_power, pv, load, temperature
    - rolling means (3,6,24 windows)
    """
    fe = df.copy()
    # basic features
    fe["net_power"] = fe["battery_discharging_rate"].fillna(0.0) - fe["battery_charging_rate"].fillna(0.0)
    base_cols = ["battery_charging_rate", "battery_discharging_rate", "net_power", "solar_pv_output", "grid_load_demand", "temperature"]
    # create lags
    for lag in lags:
        for c in base_cols:
            fe[f"{c}_lag{lag}"] = fe[c].shift(lag)
    # rolling means
    for w in [3, 6, 24]:
        for c in ["net_power", "solar_pv_output", "grid_load_demand", "temperature"]:
            fe[f"{c}_roll_{w}"] = fe[c].rolling(window=w, min_periods=1).mean()
    # time features
    fe["hour"] = fe.index.hour
    fe["dayofweek"] = fe.index.dayofweek
    # drop rows with NaNs in base features only after fe creation
    return fe


def train_residual_model(df_with_feats, target_resid_col="residual", test_size=0.2, random_state=42):
    """
    Train LightGBM (or sklearn fallback) to predict residual (actual - physics).
    Returns trained model and validation metrics & indices.
    We'll use a time-respecting split (first 80% train, last 20% val).
    """
    # choose features: exclude actual/physics/residual and any index-like
    exclude = {"battery_state_of_charge", "soc_physics", "soc_hybrid", "residual"}
    feature_cols = [c for c in df_with_feats.columns if c not in exclude and df_with_feats[c].dtype in (np.float64, np.float32, np.int64, np.int32)]
    df_train = df_with_feats.dropna(subset=feature_cols + [target_resid_col]).copy()
    if len(df_train) < 50:
        raise RuntimeError("Not enough rows to train residual model.")
    # time split
    split_idx = int(0.8 * len(df_train))
    train = df_train.iloc[:split_idx]
    val = df_train.iloc[split_idx:]
    X_train, y_train = train[feature_cols], train[target_resid_col]
    X_val, y_val = val[feature_cols], val[target_resid_col]

    if ML_BACKEND == "lightgbm":
        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        preds = model.predict(X_val)
    else:
        model = HistGradientBoostingRegressor(**SKLEARN_PARAMS)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

    mae = float(mean_absolute_error(y_val, preds))
    rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
    return {"model": model, "feature_cols": feature_cols, "val_index": val.index, "preds_val": preds, "y_val": y_val, "mae": mae, "rmse": rmse}


def run_full_workflow(df, label):
    # infer timestep
    # NOTE: load_dataset already provided delta, but pass here if needed
    # compute delta_t in hours from index
    diffs = df.index.to_series().diff().dropna()
    delta_seconds = diffs.mode().iloc[0].total_seconds()
    delta_t_hours = delta_seconds / 3600.0

    # calibrate physics parameters
    calib = calibrate_physics(df, delta_t_hours)

    # run physics using calibrated params
    sim_physics = physics_soc(df, calib["capacity_kwh"], calib["eta_ch"], calib["eta_dis"], delta_t_hours)
    df = df.copy()
    df["soc_physics"] = sim_physics

    # compute residual (actual - physics)
    df["residual"] = df["battery_state_of_charge"] - df["soc_physics"]

    # feature engineering
    df_feats = make_features(df)

    # train residual ML
    ml_results = None
    try:
        ml_results = train_residual_model(df_feats, target_resid_col="residual")
        # apply ML correction on validation slice and fill sim_hybrid for all indices:
        df["soc_hybrid"] = df["soc_physics"].copy()
        val_idx = ml_results["val_index"]
        features = ml_results["feature_cols"]
        preds_val = ml_results["preds_val"]
        df.loc[val_idx, "soc_hybrid"] = df.loc[val_idx, "soc_physics"] + preds_val
    except Exception as e:
        print("⚠️ ML training failed or not enough data:", e)

    # Evaluate metrics
    metrics = {}
    mask = ~pd.isna(df["battery_state_of_charge"])
    if mask.sum() > 0:
        # physics metrics
        phys_rmse = float(np.sqrt(mean_squared_error(df.loc[mask, "battery_state_of_charge"], df.loc[mask, "soc_physics"])))
        phys_mae = float(mean_absolute_error(df.loc[mask, "battery_state_of_charge"], df.loc[mask, "soc_physics"]))
        metrics.update({"phys_rmse": phys_rmse, "phys_mae": phys_mae})
        # hybrid metrics
        if "soc_hybrid" in df:
            hyb_rmse = float(np.sqrt(mean_squared_error(df.loc[mask, "battery_state_of_charge"], df.loc[mask, "soc_hybrid"])))
            hyb_mae = float(mean_absolute_error(df.loc[mask, "battery_state_of_charge"], df.loc[mask, "soc_hybrid"]))
            metrics.update({"hyb_rmse": hyb_rmse, "hyb_mae": hyb_mae})
        if ml_results is not None:
            metrics.update({"ml_val_mae": ml_results["mae"], "ml_val_rmse": ml_results["rmse"]})

    # Save outputs
    out_dir = OUTPUT_BASE / label
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "label": label,
        "n_steps": int(len(df)),
        "delta_t_hours": float(delta_t_hours),
        "calibration": calib,
        "metrics": metrics,
        "ml_backend": ML_BACKEND,
    }
    (out_dir / "soc_summary.json").write_text(json.dumps(summary, indent=2))

    # Plots: timeseries comparison (actual / physics / hybrid)
    plt.figure(figsize=(14, 6))
    if "battery_state_of_charge" in df:
        plt.plot(df.index, df["battery_state_of_charge"], label="Actual SoC", alpha=0.8)
    plt.plot(df.index, df["soc_physics"], label="Physics SoC", alpha=0.8)
    if "soc_hybrid" in df:
        plt.plot(df.index, df["soc_hybrid"], label="Hybrid SoC", alpha=0.8)
    plt.legend()
    plt.title(f"SOC comparison - {label}")
    plt.ylabel("SOC (%)")
    plt.xlabel("time")
    plt.tight_layout()
    plt.savefig(out_dir / "soc_comparison.png")
    plt.close()

    # Residuals
    if "battery_state_of_charge" in df:
        plt.figure(figsize=(12, 5))
        plt.plot(df.index, df["soc_physics"] - df["battery_state_of_charge"], label="Physics residual", alpha=0.6)
        if "soc_hybrid" in df:
            plt.plot(df.index, df["soc_hybrid"] - df["battery_state_of_charge"], label="Hybrid residual", alpha=0.6)
        plt.axhline(0, color="k", linestyle="--")
        plt.legend()
        plt.title(f"SOC residuals - {label}")
        plt.ylabel("Residual %")
        plt.tight_layout()
        plt.savefig(out_dir / "soc_residuals.png")
        plt.close()

    # Save ML feature importance if available
    if ml_results is not None and ML_BACKEND == "lightgbm":
        model = ml_results["model"]
        try:
            importance = model.feature_importances_
            feat_imp = dict(zip(ml_results["feature_cols"], importance.tolist()))
            (out_dir / "ml_feature_importance.json").write_text(json.dumps(feat_imp, indent=2))
        except Exception:
            pass

    print(f"✔ Outputs saved to {out_dir}")
    return summary, df


if __name__ == "__main__":
    df, delta = load_dataset()
    if MODE == "A":
        run_full_workflow(df, "full_dataset")
    else:
        # monthly folders YYYY-MM
        for period, sub in df.groupby(df.index.to_period("M")):
            label = str(period)
            run_full_workflow(sub.copy(), label)
