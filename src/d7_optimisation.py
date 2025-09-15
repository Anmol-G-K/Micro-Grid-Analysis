# robust_optimisation.py
"""
Multi-day microgrid optimisation (Polars + CVXPY) with battery-capacity
sweep recommendations and LLM-friendly JSON outputs.

Notes:
- Requires: polars, cvxpy, numpy, matplotlib, scikit-learn (optional for ML part).
- Tune CANDIDATE_ADDITIONAL_KWH and DAYS_TO_RUN for runtime vs accuracy tradeoff.
"""

import polars as pl
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
from pathlib import Path
import numpy as np
from datetime import datetime
import math

# Optional ML
try:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# -------------------------
# User / environment config
# -------------------------
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "combined.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "plots" / "optimisation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# If battery SOC in dataset looks like a fraction (0..1) or percent (0..100),
# set a default nominal capacity (kWh) to interpret SOC fraction -> kWh.
DEFAULT_NOMINAL_BATTERY_KWH = 100.0

# Candidate additional battery capacities to test (kWh). Adjust as needed.
CANDIDATE_ADDITIONAL_KWH = [0.0, 0.25, 0.5, 1.0, 2.0]  # multiples of base capacity (see below)
# To interpret above as multiples, below flag True -> candidate = multiple * base_C
CANDIDATES_AS_MULTIPLES_OF_BASE = True

# To limit runtime, you can run only first N days (None => all days)
DAYS_TO_RUN = 30  # e.g., 7 or None, None equals whole dataset

# Solver config
CVX_SOLVER = cp.SCS  # alternative: cp.ECOS if installed

# Optimization hyperparams (tweakable)
ETA_CH = 0.90
ETA_DIS = 0.93
P_MAX_INV = 50.0  # kW inverter limit per battery (single-bank limit)
SOC_MIN_FRAC = 0.10  # 10% default
SOC_MAX_FRAC = 0.90  # 90% default

# Objective weights
RENEWABLE_REWARD = 0.05   # higher -> stronger preference to use renewables
CHARGE_PENALTY = 0.01     # penalize charging (to discourage charging from grid)
# (main driver is grid_price_import, so these are secondary)

# Price parameters (can be changed to time-varying vector later)
GRID_PRICE_IMPORT = 0.12
GRID_PRICE_EXPORT = 0.03

# Grid limits
GRID_IMP_MAX_KW = 1e6
GRID_EXP_MAX_KW = 1e6

# -------------------------
# Utility helpers
# -------------------------
def infer_capacity_and_soc_kwh(soc_series, default_capacity=DEFAULT_NOMINAL_BATTERY_KWH):
    """
    Given a Polars Series (soc values), infer whether values are kWh or fraction/percent.
    Return (C_kwh, soc_kwh_initial) where:
      - C_kwh is chosen battery nominal capacity (kWh)
      - soc_kwh_initial is the initial SoC in kWh (first entry of series converted)
    Heuristic:
      - if max(soc) > 5.0 --> assume values are already in kWh and set capacity = max observed.
      - elif max(soc) > 1.0 and <= 100 --> probably percent; treat as percent of default_capacity.
      - else treat as fraction (0..1) and multiply by default_capacity.
    """
    arr = np.asarray(soc_series)
    if np.isnan(arr).all():
        # fallback
        return default_capacity, 0.0

    maxv = np.nanmax(arr)
    firstv = arr[0] if len(arr) > 0 else 0.0

    if maxv > 5.0:
        # likely kWh already
        C = float(maxv)
        soc0 = float(firstv)
        return C, soc0
    if 1.0 < maxv <= 100.0:
        # percent values (0-100)
        C = default_capacity
        soc0 = float(firstv) / 100.0 * C
        return C, soc0
    # assume fraction (0..1)
    C = default_capacity
    soc0 = float(firstv) * C
    return C, soc0

def make_time_axis(day_slice):
    # return array of python datetimes for plotting x-axis
    # day_slice["timestamp"] is Polars Series of datetimes
    try:
        timestamps = day_slice["timestamp"].to_numpy()
        # Convert numpy datetime64 -> python datetime
        times = [pd_ts.astype("datetime64[ms]").astype("O") if isinstance(pd_ts, np.datetime64) else pd_ts for pd_ts in timestamps]
        # The above conversion is somewhat defensive; simplest to use pandas
        import pandas as _pd
        times = _pd.to_datetime(timestamps).to_pydatetime()
    except Exception:
        # fallback to integer steps
        times = np.arange(len(day_slice))
    return times

def run_optim_day(load, solar, wind, soc_init_kwh, C_kwh,
                  P_max_inv=P_MAX_INV,
                  eta_ch=ETA_CH, eta_dis=ETA_DIS,
                  dt=0.25,  # hours (15-min => 0.25)
                  grid_price_import=GRID_PRICE_IMPORT,
                  grid_price_export=GRID_PRICE_EXPORT,
                  grid_imp_max_kw=GRID_IMP_MAX_KW,
                  grid_exp_max_kw=GRID_EXP_MAX_KW,
                  renewable_reward=RENEWABLE_REWARD,
                  charge_penalty=CHARGE_PENALTY,
                  solver=CVX_SOLVER):
    """
    Build and solve CVXPY optimisation for one day.
    Returns dictionary with:
        - success: bool
        - status, objective
        - arrays (P_solar_use, P_wind_use, P_batt_ch, P_batt_dis, P_grid_imp, P_grid_exp, SoC)
        - summary metrics (grid import/export kWh, renewable used, battery energy, SoC stats)
    """
    T = len(load)
    if T == 0:
        return None

    # Define variables
    P_batt_ch = cp.Variable(T, nonneg=True)
    P_batt_dis = cp.Variable(T, nonneg=True)
    P_solar_use = cp.Variable(T, nonneg=True)
    P_wind_use = cp.Variable(T, nonneg=True)
    P_grid_imp = cp.Variable(T, nonneg=True)
    P_grid_exp = cp.Variable(T, nonneg=True)
    SoC = cp.Variable(T+1)

    constraints = []
    constraints += [SoC[0] == soc_init_kwh]

    for t in range(T):
        # SoC dynamics (kWh)
        constraints += [
            SoC[t+1] == SoC[t] + eta_ch * P_batt_ch[t] * dt - (1.0/eta_dis) * P_batt_dis[t] * dt
        ]
        # Power balance: load must be met by renewable + battery discharge + grid import - grid export
        # Battery charging increases demand, but charge itself is a separate variable constrained by SoC dynamics.
        constraints += [
            load[t] == P_solar_use[t] + P_wind_use[t] + P_batt_dis[t] + P_grid_imp[t] - P_grid_exp[t]
        ]
        # Bounds
        constraints += [
            P_solar_use[t] <= solar[t],
            P_wind_use[t] <= wind[t],
            P_batt_ch[t] <= P_max_inv,
            P_batt_dis[t] <= P_max_inv,
            P_batt_ch[t] + P_batt_dis[t] <= P_max_inv,
            P_grid_imp[t] <= grid_imp_max_kw,
            P_grid_exp[t] <= grid_exp_max_kw
        ]

    # SoC bounds (kWh)
    SoC_min = SOC_MIN_FRAC * C_kwh
    SoC_max = SOC_MAX_FRAC * C_kwh
    constraints += [SoC >= SoC_min, SoC <= SoC_max]
    # End-of-horizon: keep battery at least at starting level to avoid stealing next day
    constraints += [SoC[T] >= soc_init_kwh]

    # Objective: minimise grid import cost minus export credit,
    # reward renewable usage and penalise unnecessary charging (secondary terms)
    obj = (grid_price_import * cp.sum(P_grid_imp) * dt
           - grid_price_export * cp.sum(P_grid_exp) * dt
           - renewable_reward * cp.sum(P_solar_use + P_wind_use) * dt
           + charge_penalty * cp.sum(P_batt_ch) * dt)

    problem = cp.Problem(cp.Minimize(obj), constraints)
    try:
        problem.solve(solver=solver, verbose=False)
    except Exception as e:
        # fallback try
        try:
            problem.solve()
        except Exception as e2:
            return {"success": False, "error": str(e2)}

    # collect results
    def extract(var):
        return np.array(var.value).flatten() if var.value is not None else None

    try:
        P_solar_use_v = extract(P_solar_use)
        P_wind_use_v = extract(P_wind_use)
        P_batt_ch_v = extract(P_batt_ch)
        P_batt_dis_v = extract(P_batt_dis)
        P_grid_imp_v = extract(P_grid_imp)
        P_grid_exp_v = extract(P_grid_exp)
        SoC_v = extract(SoC)

        # If any are None, abort
        if any(v is None for v in [P_solar_use_v, P_wind_use_v, P_batt_ch_v,
                                P_batt_dis_v, P_grid_imp_v, P_grid_exp_v, SoC_v]):
            return {"success": False, "error": "Some optimisation variables returned None."}
    except Exception as e:
        return {"success": False, "error": "variable extraction failed: " + str(e)}

    def energy_sum(arr):
        return float(np.nansum(arr) * dt)

    total_renewable_prod = float(np.nansum(solar) * dt + np.nansum(wind) * dt)
    renewable_used_kwh = energy_sum(P_solar_use_v) + energy_sum(P_wind_use_v)
    total_load_kwh = float(np.nansum(load) * dt)
    summary = {
        "success": True,
        "status": problem.status,
        "objective": float(problem.value) if problem.value is not None else None,

        "grid_import_kwh": energy_sum(P_grid_imp_v),
        "grid_export_kwh": energy_sum(P_grid_exp_v),
        "grid_net_kwh": energy_sum(P_grid_imp_v) - energy_sum(P_grid_exp_v),

        "solar_used_kwh": energy_sum(P_solar_use_v),
        "wind_used_kwh": energy_sum(P_wind_use_v),
        "total_renewable_prod_kwh": total_renewable_prod,
        "total_renewable_used_kwh": renewable_used_kwh,
        "renewable_fraction_of_load": renewable_used_kwh / total_load_kwh if total_load_kwh > 0 else None,

        "battery_charged_kwh": energy_sum(P_batt_ch_v),
        "battery_discharged_kwh": energy_sum(P_batt_dis_v),
        "battery_efficiency": (energy_sum(P_batt_dis_v) / energy_sum(P_batt_ch_v)) if energy_sum(P_batt_ch_v) > 0 else None,

        "SoC_min_kwh": float(np.nanmin(SoC_v)),
        "SoC_max_kwh": float(np.nanmax(SoC_v)),
        "SoC_avg_kwh": float(np.nanmean(SoC_v)),
        "total_load_kwh": total_load_kwh
    }

    # Also return time-series for plotting
    result = {
        "summary": summary,
        "P_solar_use_v": P_solar_use_v,
        "P_wind_use_v": P_wind_use_v,
        "P_batt_ch_v": P_batt_ch_v,
        "P_batt_dis_v": P_batt_dis_v,
        "P_grid_imp_v": P_grid_imp_v,
        "P_grid_exp_v": P_grid_exp_v,
        "SoC_v": SoC_v
    }

    return result

def simulate_capacity_candidates(day_slice, base_C, soc0_kwh,
                                 candidates=CANDIDATE_ADDITIONAL_KWH,
                                 multiples=CANDIDATES_AS_MULTIPLES_OF_BASE,
                                 max_candidates=5):
    """
    Run optimisation for multiple candidate additional battery capacities and
    return dict of results keyed by candidate_kwh.
    """
    load = day_slice["load_kw"].to_numpy()
    solar = day_slice["solar_power"].to_numpy()
    wind = day_slice["wind_power"].to_numpy()
    dt = 15/60.0

    results = {}
    for c in candidates[:max_candidates]:
        if multiples:
            add_kwh = c * base_C
        else:
            add_kwh = float(c)
        C_new = base_C + add_kwh
        # keep same SoC fraction
        soc_frac = soc0_kwh / base_C if base_C > 0 else 0.0
        soc0_new = soc_frac * C_new

        res = run_optim_day(load, solar, wind, soc0_new, C_new,
                            P_max_inv=P_MAX_INV,
                            eta_ch=ETA_CH, eta_dis=ETA_DIS,
                            dt=dt,
                            grid_price_import=GRID_PRICE_IMPORT,
                            grid_price_export=GRID_PRICE_EXPORT)

        if res is None or not res.get("summary", None):
            results[add_kwh] = {"success": False, "error": "optim failed or returned None"}
        else:
            results[add_kwh] = {
                "success": True,
                "summary": res["summary"],
                "P_grid_imp_v": res["P_grid_imp_v"].tolist(),  # store full time series
                "SoC_v": res["SoC_v"].tolist()
            }

    return results

# -------------------------
# Step 1: Load & preprocess (Polars)
# -------------------------
# Required target columns (expanded to include useful weather/metadata)
TARGET_COLS = [
    "timestamp",
    "solar_pv_output",
    "wind_power_output",
    "total_renewable_energy",
    "battery_state_of_charge",
    "battery_charging_rate",
    "battery_discharging_rate",
    "grid_load_demand",
    "solar_irradiance",
    "wind_speed",
    "temperature",
    "humidity",
    "atmospheric_pressure",
    "power_exchange",
    "voltage",
    "frequency"
]

# Read parquet
df = pl.read_parquet(DATA_PATH)

# Ensure timestamp parsed
df = df.with_columns(pl.col("timestamp").str.strptime(pl.Datetime, strict=False))

# Check required cols exist, otherwise fail early
miss = [c for c in TARGET_COLS if c not in df.columns]
if miss:
    raise KeyError(f"Missing required columns in dataset: {miss}")

# Keep required columns only
df = df.select(TARGET_COLS)

# Downsample to 15-min and aggregate useful columns (means)
df_15min = (
    df.group_by_dynamic(index_column="timestamp", every="15m", closed="left")
      .agg([
          pl.mean("grid_load_demand").alias("load_kw"),
          pl.mean("total_renewable_energy").alias("pv_kw"),
          pl.mean("battery_state_of_charge").alias("soc_raw"),
          pl.mean("solar_pv_output").alias("solar_power"),
          pl.mean("wind_power_output").alias("wind_power"),
          pl.mean("solar_irradiance").alias("solar_irradiance"),
          pl.mean("wind_speed").alias("wind_speed"),
          pl.mean("temperature").alias("temperature")
      ])
)

# Extract unique dates
dates_df = df_15min.select(pl.col("timestamp").dt.date().alias("date")).unique().sort("date")
unique_dates = [row["date"] for row in dates_df.to_dicts()]

# Optionally slice days to run
if DAYS_TO_RUN is not None:
    unique_dates = unique_dates[:DAYS_TO_RUN]

# -------------------------
# Main loop: run per-day optimization + candidate sweep
# -------------------------
all_days_summary = {}

for DAY in unique_dates:
    # filter day
    day_slice = df_15min.filter(pl.col("timestamp").dt.date() == DAY).sort("timestamp")
    if len(day_slice) == 0:
        continue

    # arrays
    load = day_slice["load_kw"].to_numpy()
    solar = day_slice["solar_power"].to_numpy()
    wind = day_slice["wind_power"].to_numpy()

    # infer capacity and initial soc in kWh
    C_base, soc0_kwh = infer_capacity_and_soc_kwh(day_slice["soc_raw"].to_numpy(),
                                                  default_capacity=DEFAULT_NOMINAL_BATTERY_KWH)

    # run baseline optimisation at base capacity
    baseline_res = run_optim_day(load, solar, wind, soc0_kwh, C_base)

    if (baseline_res is None) or (not baseline_res.get("summary")):
        all_days_summary[str(DAY)] = {"error": "baseline optimisation failed"}
        continue

    baseline_summary = baseline_res["summary"]

    # run candidate capacity sweep (multiples of base_C or absolute kWh)
    sweep_results = simulate_capacity_candidates(day_slice, C_base, soc0_kwh,
                                                 candidates=CANDIDATE_ADDITIONAL_KWH,
                                                 multiples=CANDIDATES_AS_MULTIPLES_OF_BASE,
                                                 max_candidates=len(CANDIDATE_ADDITIONAL_KWH))

    # choose recommended candidate: pick the candidate with minimal grid_net_kwh (prefers less import)
    best = None
    for add_kwh, r in sweep_results.items():
        if not r.get("success", False):
            continue
        s = r["summary"]
        gnet = s.get("grid_net_kwh", float("inf"))
        if best is None or gnet < best["grid_net_kwh"]:
            best = {"add_kwh": add_kwh, "grid_net_kwh": gnet, "summary": s}

    # prepare JSON friendly summary for this day (LLM-friendly)
    summary = {
        "date": str(DAY),
        "baseline": baseline_summary,
        "candidates": {},
        "recommended_additional_battery_kwh": None,
        "recommended_expected_grid_import_kwh": None,
        "recommended_expected_grid_export_kwh": None,
        "notes": ""
    }

    # add detailed candidate summaries
    for add_kwh, r in sweep_results.items():
        entry = {"success": r.get("success", False)}
        if r.get("success", False):
            entry["summary"] = r["summary"]
        else:
            entry["error"] = r.get("error", "unknown")
        summary["candidates"][str(add_kwh)] = entry

    if best is not None:
        summary["recommended_additional_battery_kwh"] = float(best["add_kwh"])
        summary["recommended_expected_grid_import_kwh"] = float(best["summary"]["grid_import_kwh"])
        summary["recommended_expected_grid_export_kwh"] = float(best["summary"]["grid_export_kwh"])
        reduction_kwh = baseline_summary["grid_import_kwh"] - best["summary"]["grid_import_kwh"]
        reduction_pct = (reduction_kwh / baseline_summary["grid_import_kwh"]) if baseline_summary["grid_import_kwh"] > 0 else None
        summary["notes"] = (f"Best candidate adds {best['add_kwh']:.2f} kWh to base {C_base:.1f} kWh; "
                            f"this reduces grid import by {reduction_kwh:.2f} kWh ({reduction_pct:.2%}).")
    else:
        summary["notes"] = "No successful candidate optimisation run."

    # add peaks and timing info to baseline summary to help LLM reasoning
    # Peak load time
    try:
        ts = day_slice["timestamp"].to_numpy()
        import pandas as _pd
        ts_pd = _pd.to_datetime(ts)
        peak_load_idx = int(np.nanargmax(np.nan_to_num(load)))
        peak_pv_idx = int(np.nanargmax(np.nan_to_num(solar + wind)))
        # times as ISO strings
        summary["baseline"]["peak_load_time"] = str(ts_pd.iloc[peak_load_idx])
        summary["baseline"]["peak_load_kw"] = float(load[peak_load_idx])
        summary["baseline"]["peak_renewable_time"] = str(ts_pd.iloc[peak_pv_idx])
        summary["baseline"]["peak_renewable_kw"] = float((solar + wind)[peak_pv_idx])
    except Exception:
        pass

    # store
    all_days_summary[str(DAY)] = summary

    # -------------------------
    # Plot: combined dispatch + SoC (one file per day)
    # -------------------------
    # we use the baseline_res time-series for plotting
    res = baseline_res
    P_solar_use_v = res["P_solar_use_v"]
    P_wind_use_v = res["P_wind_use_v"]
    P_batt_dis_v = res["P_batt_dis_v"]
    P_batt_ch_v = res["P_batt_ch_v"]
    P_grid_imp_v = res["P_grid_imp_v"]
    P_grid_exp_v = res["P_grid_exp_v"]
    SoC_v = res["SoC_v"]

    # x-axis times
    try:
        import pandas as _pd
        times = _pd.to_datetime(day_slice["timestamp"].to_numpy()).to_pydatetime()
        x = times
        xticks = None
    except Exception:
        x = np.arange(len(load))
        xticks = None

    if best is not None and "P_grid_imp_v" in sweep_results[best["add_kwh"]]:
        # Plot both baseline and best candidate
        fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # Grid import comparison
        ax[0].plot(x, P_grid_imp_v, label="Baseline Grid Import", color="tab:blue", linewidth=2)
        ax[0].plot(x, sweep_results[best["add_kwh"]]["P_grid_imp_v"], label=f"Optimized (+{best['add_kwh']} kWh)", color="tab:green", linewidth=2)
        ax[0].set_ylabel("Grid Import (kW)")
        ax[0].set_title(f"Grid Import Comparison — {DAY}")
        ax[0].legend()
        ax[0].grid(alpha=0.3)

        # Optional: SoC comparison
        ax[1].plot(x, 100.0 * SoC_v[:-1]/C_base, label="Baseline SoC (%)", color="tab:blue")
        best_soc = np.array(sweep_results[best["add_kwh"]]["SoC_v"])
        C_best = C_base + best["add_kwh"]
        ax[1].plot(x, 100.0 * best_soc[:-1]/C_best, label="Optimized SoC (%)", color="tab:green")
        ax[1].set_ylabel("State of Charge (%)")
        ax[1].set_xlabel("Time")
        ax[1].legend()
        ax[1].grid(alpha=0.3)

        plt.tight_layout()
        fig_path = OUTPUT_DIR / f"optim_grid_import_comparison_{DAY}.png"
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)


# -------------------------
# Save combined JSON for LLM ingestion
# -------------------------
out_path = OUTPUT_DIR / "optim_summary_all_days.json"
with open(out_path, "w") as fh:
    json.dump(all_days_summary, fh, indent=2, default=str)

# -------------------------
# Optional lightweight ML: predict grid_import_kwh from features (Ridge)
# -------------------------
if SKLEARN_AVAILABLE:
    try:
        import pandas as _pd
        rows = []
        for d, v in all_days_summary.items():
            if isinstance(v, dict) and "baseline" in v and v["baseline"].get("total_load_kwh") is not None:
                bs = v["baseline"]
                rows.append({
                    "date": d,
                    "total_load_kwh": bs["total_load_kwh"],
                    "total_renewable_prod_kwh": bs.get("total_renewable_prod_kwh", 0.0),
                    "battery_capacity_kwh": (float(v.get("baseline", {}).get("SoC_max_kwh", DEFAULT_NOMINAL_BATTERY_KWH)) if v.get("baseline") else DEFAULT_NOMINAL_BATTERY_KWH),
                    "grid_import_kwh": bs["grid_import_kwh"]
                })
        if len(rows) >= 5:
            df_ml = _pd.DataFrame(rows)
            X = df_ml[["total_load_kwh", "total_renewable_prod_kwh", "battery_capacity_kwh"]].values
            y = df_ml["grid_import_kwh"].values
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            model = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=3).fit(Xs, y)
            # attach simple model summary
            ml_info = {
                "sklearn": True,
                "coef": model.coef_.tolist(),
                "intercept": float(model.intercept_),
                "score_r2": float(model.score(Xs, y))
            }
            # write ML metadata into the combined JSON
            all_days_summary["_ml_model_summary"] = ml_info
            with open(out_path, "w") as fh:
                json.dump(all_days_summary, fh, indent=2, default=str)
    except Exception:
        # If ML fails, we continue silently
        pass

print("Done — combined JSON and per-day plots saved to:", OUTPUT_DIR)
