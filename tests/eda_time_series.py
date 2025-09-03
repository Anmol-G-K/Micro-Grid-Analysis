"""
ts_forecast_metadata_polars.py

Produces JSON with time-series forecasting metadata:
- stationarity (ADF) and suggested differencing order d
- dominant seasonal periods (FFT + seasonal_decompose)
- ACF / PACF summaries
- cross-correlation (lagged) between specified variable pairs
- variance-stability heuristics (rolling variance, Box-Cox suggestion)
- forecast-horizon mapping based on detected frequency
- model recommendations (simple heuristic)

Optimised with Polars for large datasets (70M+ rows).
"""

import json
from pathlib import Path
import logging
import numpy as np
import polars as pl
import pandas as pd
from scipy import signal
from scipy.stats import boxcox_normmax
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings

# warnings.filterwarnings("ignore")

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger()

DATA_PATH = Path(__file__).resolve().parent.parent / "data/cleaned_dataset.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "plots" / "metadata"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "ts_forecast_metadata.json"

USER_NUMERIC_COLS = ["GE_Active_Power", "PVPCS_Active_Power", "Battery_Active_Power", "FC_Active_Power", "Inlet_Temperature_of_Chilled_Water", "Outlet_Temperature"]
USER_PAIRS = [("GE_Active_Power", "PVPCS_Active_Power"), ("GE_Active_Power", "Inlet_Temperature_of_Chilled_Water"), ("Battery_Active_Power", "PVPCS_Active_Power"), ("FC_Active_Power", "GE_Active_Power")]
COL_GROUPS = {
    "battery": ["Battery_Active_Power", "Battery_Active_Power_Set_Response"],
    "pv": ["PVPCS_Active_Power"],
    "fuel_cell": ["FC_Active_Power_FC_END_Set", "FC_Active_Power", "FC_Active_Power_FC_end_Set_Response"],
    "ge": ["GE_Body_Active_Power", "GE_Active_Power", "GE_Body_Active_Power_Set_Response"],
    "mccb": ["Island_mode_MCCB_Active_Power"],
    "voltages": ["MG-LV-MSB_AC_Voltage", "Receiving_Point_AC_Voltage", "Island_mode_MCCB_AC_Voltage"],
    "frequencies": ["Island_mode_MCCB_Frequency", "MG-LV-MSB_Frequency"],
    "temperatures": ["Inlet_Temperature_of_Chilled_Water", "Outlet_Temperature"]
}

def detect_ts_col(df: pl.DataFrame):
    log.info("Detecting timestamp column...")
    for c in df.columns:
        if "time" in c.lower() or "date" in c.lower() or "timestamp" in c.lower():
            log.info(f"Timestamp column detected by name: {c}")
            return c
    for c in df.columns:
        if pl.datatypes.is_datetime(df.schema[c]):
            log.info(f"Timestamp column detected by dtype: {c}")
            return c
    log.warning("No timestamp column detected.")
    return None

def infer_freq(df: pl.DataFrame, ts_col: str):
    log.info(f"Inferring frequency for column '{ts_col}'...")
    ts = pd.to_datetime(df[ts_col].to_numpy(), errors="coerce")
    diffs = pd.Series(ts).diff().dt.total_seconds().dropna()
    if len(diffs) == 0:
        log.warning("No time differences found; frequency inference failed.")
        return None, None
    seconds = int(round(diffs.median()))
    if seconds % 86400 == 0: freq_alias = f"{seconds // 86400}D"
    elif seconds % 3600 == 0: freq_alias = f"{seconds // 3600}H"
    elif seconds % 60 == 0: freq_alias = f"{seconds // 60}T"
    else: freq_alias = f"{seconds}S"
    log.info(f"Inferred frequency: {freq_alias} (approx. {seconds} seconds)")
    return seconds, freq_alias

def adf_diff(series: pd.Series, max_d=3):
    for d in range(max_d + 1):
        s = series if d == 0 else series.diff(d).dropna()
        if len(s) < 10:
            log.debug(f"ADF differencing order {d} skipped due to insufficient length")
            continue
        try:
            pval = adfuller(s)[1]
            log.debug(f"ADF test p-value at differencing {d}: {pval:.4f}")
            if pval < 0.05:
                log.info(f"ADF test suggests stationarity at differencing order {d}")
                return {"suggested_d": d}
        except Exception as e:
            log.debug(f"ADF test failed at differencing {d}: {e}")
            continue
    log.info("ADF test suggests non-stationarity up to max differencing order")
    return {"suggested_d": None}

def dominant_fft(series: pd.Series, step_sec: int, top_n=5):
    if len(series) < 10 or step_sec is None:
        log.info("Skipping FFT dominance due to insufficient data or unknown step size.")
        return {}
    y = signal.detrend(series.values.astype(float))
    freqs = np.fft.rfftfreq(len(y), d=step_sec)
    ps = np.abs(np.fft.rfft(y))**2
    mask = freqs > 0
    freqs, ps = freqs[mask], ps[mask]
    peaks, _ = signal.find_peaks(ps)
    idx = peaks[np.argsort(ps[peaks])[-top_n:]][::-1] if len(peaks) else np.argsort(ps)[-top_n:][::-1]
    periods = []
    for i in idx:
        if freqs[i] > 0:
            p = int(round(1 / freqs[i]))
            periods.append({"period_seconds": p, "period_hours": p/3600, "period_steps": int(round(p/step_sec))})
    log.info(f"Dominant FFT periods detected: {[p['period_seconds'] for p in periods]}")
    return {"dominant_periods": periods}

def seasonal_decomp(series: pd.Series, candidates):
    results = []
    for p in candidates:
        if p < 2 or len(series) < p * 2:
            log.debug(f"Skipping seasonal decomposition for period {p} due to insufficient length.")
            continue
        try:
            res = seasonal_decompose(series, period=p, model="additive")
            amp = res.seasonal.max() - res.seasonal.min()
            results.append({"period": p, "seasonal_amplitude": float(amp)})
        except Exception as e:
            log.debug(f"Seasonal decomposition failed for period {p}: {e}")
            continue
    log.info(f"Seasonal decomposition results: {results}")
    return {"decompose_success": results}

def acf_pacf(series: pd.Series, nlags=48):
    s = series.dropna()
    if len(s) < 10:
        log.info("Skipping ACF/PACF due to insufficient data.")
        return {}
    nlags = min(nlags, len(s)//2 - 1)
    acf_vals = acf(s, nlags=nlags)
    pacf_vals = pacf(s, nlags=nlags)
    conf = 2.0 / np.sqrt(len(s))
    acf_sig = [{"lag": i, "value": float(v)} for i, v in enumerate(acf_vals) if i and abs(v) > conf][:20]
    pacf_sig = [{"lag": i, "value": float(v)} for i, v in enumerate(pacf_vals) if i and abs(v) > conf][:20]
    log.info(f"ACF significant lags: {[d['lag'] for d in acf_sig]}")
    log.info(f"PACF significant lags: {[d['lag'] for d in pacf_sig]}")
    return {
        "acf_significant_lags": acf_sig,
        "pacf_significant_lags": pacf_sig,
        "significance_threshold": conf
    }

def boxcox_hint(series: pd.Series):
    s = series.dropna()
    if (s <= 0).any():
        log.info("Box-Cox suggestion: shift then boxcox or log (data contains non-positive values).")
        return {"suggestion": "shift_then_boxcox_or_log"}
    try:
        lam = float(boxcox_normmax(s, method="mle"))
        log.info(f"Box-Cox lambda estimated: {lam:.4f}")
        return {"suggestion": "boxcox", "lambda": lam}
    except Exception as e:
        log.warning(f"Box-Cox estimation failed: {e}")
        return {}

def variance_roll(series: pd.Series, window=24):
    var = series.rolling(window, min_periods=1).var()
    q25, q75 = var.quantile(0.25), var.quantile(0.75)
    ratio = float(q75/q25) if q25 > 0 else None
    log.info(f"Rolling variance median: {var.median():.4f}, Q75/Q25 ratio: {ratio}")
    return {"rolling_variance_median": var.median(), "variance_q75_q25_ratio": ratio}

def cross_corr(x: pd.Series, y: pd.Series, max_lags=48):
    a, b = x.align(y, join="inner")
    a, b = a.dropna(), b.dropna()
    if min(len(a), len(b)) < 10:
        log.info(f"Skipping cross-correlation: insufficient overlap for series of lengths {len(a)}, {len(b)}")
        return {}
    a, b = a - a.mean(), b - b.mean()
    corrs = np.correlate(a, b, mode="full") / (np.std(a) * np.std(b) * len(a))
    lags = np.arange(-max_lags, max_lags+1)
    mid = len(corrs)//2
    seg = corrs[mid - max_lags:mid + max_lags + 1]
    idx = np.argmax(np.abs(seg))
    best_lag = int(lags[idx])
    best_corr = float(seg[idx])
    log.info(f"Best cross-correlation lag: {best_lag} with correlation {best_corr:.4f}")
    return {"best_lag": best_lag, "best_corr": best_corr}

def recommend_models(stationary, has_seasonality):
    if not stationary:
        models = ["SARIMA", "Prophet"] if has_seasonality else ["ARIMA", "ETS"]
    else:
        models = ["SARIMA", "TBATS"] if has_seasonality else ["ARIMA", "SimpleExpSmoothing"]
    log.info(f"Recommended models: {models}")
    return models

def main():
    log.info(f"Loading dataset from {DATA_PATH}...")
    df = pl.read_parquet(DATA_PATH)
    ts_col = detect_ts_col(df)

    if ts_col and not df[ts_col].dtype == pl.Datetime:
        log.info(f"Parsing {ts_col} column to datetime...")
        df = df.with_columns(pl.col(ts_col).str.strptime(pl.Datetime, strict=False))

    seconds, freq = infer_freq(df, ts_col) if ts_col else (None, None)

    keep_cols = [ts_col] + USER_NUMERIC_COLS if ts_col else USER_NUMERIC_COLS
    log.info(f"Selecting columns: {keep_cols} and dropping nulls...")
    df = df.select(keep_cols).drop_nulls()

    df_p = df.to_pandas()

    if ts_col:
        df_p[ts_col] = pd.to_datetime(df_p[ts_col], errors="coerce")
        df_p = df_p.dropna(subset=[ts_col]).set_index(ts_col).sort_index()
        if freq:
            log.info(f"Downsampling data to frequency: {freq} ...")
            df_p = df_p.resample(freq).mean()
            log.info(f"Data shape after resampling: {df_p.shape}")

    metadata = {
        "dataset": {
            "shape": df_p.shape,
            "timestamp_column": ts_col,
            "numeric_columns": USER_NUMERIC_COLS,
            "approx_frequency_seconds": seconds,
            "approx_frequency_alias": freq
        },
        "groups": {},
        "cross_correlation": {}
    }

    overall_stationary = True
    seasonal_periods_all = []

    for col in USER_NUMERIC_COLS:
        if col not in df_p.columns:
            log.warning(f"Column {col} not found in data, skipping...")
            continue
        log.info(f"Processing column: {col}")
        series = df_p[col].dropna()
        col_meta = {}
        col_meta["adf"] = adf_diff(series)
        col_meta["fft"] = dominant_fft(series, seconds)
        periods = [p["period_steps"] for p in col_meta["fft"].get("dominant_periods", []) if p["period_steps"] < len(series)//2]
        seasonal_periods_all.extend(periods)
        col_meta["decompose"] = seasonal_decomp(series, periods)
        col_meta["acf_pacf"] = acf_pacf(series)
        col_meta["boxcox"] = boxcox_hint(series)
        col_meta["rolling_variance"] = variance_roll(series)
        stationary = col_meta["adf"].get("suggested_d", 0) == 0
        has_seasonality = len(col_meta["decompose"]["decompose_success"]) > 0
        col_meta["recommended_models"] = recommend_models(stationary, has_seasonality)
        overall_stationary &= stationary
        metadata["groups"][col] = col_meta

    for a, b in USER_PAIRS:
        if a in df_p.columns and b in df_p.columns:
            log.info(f"Computing cross-correlation between {a} and {b}...")
            metadata["cross_correlation"][f"{a}__vs__{b}"] = cross_corr(df_p[a], df_p[b])

    metadata["summary"] = {
        "overall_stationary": overall_stationary,
        "common_seasonal_periods": list(set(seasonal_periods_all)),
    }

    log.info(f"Writing metadata JSON to {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info("Done.")

if __name__ == "__main__":
    main()
