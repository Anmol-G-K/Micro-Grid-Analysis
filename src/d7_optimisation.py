# robust_optimisation.py
import polars as pl
import pandas as pd
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
from pathlib import Path
import numpy as np
from datetime import datetime

# --- Paths ---
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "combined.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "plots" / "optimisation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Step 1: Load data
# -------------------------
# --- Load with Polars ---
df = pl.read_parquet(DATA_PATH)

# Convert timestamp to Datetime
df = df.with_columns(
    pl.col("timestamp").str.strptime(pl.Datetime, strict=False)
)

# Keep required columns
required_cols = ['timestamp', 'grid_load_demand', 'total_renewable_energy', 'battery_state_of_charge', 'solar_pv_output','wind_power_output']  
df = df.select(required_cols)

# Downsample to 15-min using the correct method
df_15min = (
    df.group_by_dynamic(index_column="timestamp", every="15m", closed="left")
      .agg([
          pl.mean("grid_load_demand").alias("load_kw"),
          pl.mean("total_renewable_energy").alias("pv_kw"),
          pl.mean("battery_state_of_charge").alias("soc_kwh"),
          pl.mean("solar_pv_output").alias("solar_power"),
          pl.mean("wind_power_output").alias("wind_power")
      ])
)

# Filter for the desired day
# Get unique dates from timestamp column
unique_dates = df_15min.select(
    pl.col("timestamp").dt.date().alias("date")
).unique().to_series().to_list()


# Convert to numpy arrays for CVXPY
all_summaries = []
all_days_summary = {}

for DAY in unique_dates:
    day_slice = df_15min.filter(pl.col("timestamp").dt.date() == DAY).sort("timestamp")

    # Convert to numpy arrays
    load = day_slice["load_kw"].to_numpy()
    pv = day_slice["pv_kw"].to_numpy()
    solar = day_slice["solar_power"].to_numpy()
    wind = day_slice["wind_power"].to_numpy()
    soc_init = day_slice["soc_kwh"].to_numpy()[0]
    T = len(load)
    dt = 15 / 60.0  # 15 minutes in hours


    # -------------------------
    # Step 2: Device & battery params
    # -------------------------
    # C = df_15min['soc_kwh'].max()  # battery capacity from data
    C = day_slice['soc_kwh'].max()
    eta_ch = 0.9
    eta_dis = 0.93
    P_max = 50.0  # battery inverter kW
    SoC_min = 0.1 * C
    SoC_max = 0.9 * C

    # External devices
    grid_imp_max_kw = 1e6
    grid_exp_max_kw = 1e6
    grid_price_import = 0.12
    grid_price_export = 0.03

    # -------------------------
    # Step 3: Decision variables
    # -------------------------
    P_batt_ch = cp.Variable(T, nonneg=True)
    P_solar_use = cp.Variable(T, nonneg=True)
    P_wind_use = cp.Variable(T, nonneg=True)
    P_batt_dis = cp.Variable(T, nonneg=True)
    P_pv_use = cp.Variable(T, nonneg=True)
    P_grid_imp = cp.Variable(T, nonneg=True)
    P_grid_exp = cp.Variable(T, nonneg=True)
    SoC = cp.Variable(T+1)

    # -------------------------
    # Step 4: Constraints
    # -------------------------
    constraints = [SoC[0] == soc_init]

    for t in range(T):
        # SoC dynamics
        constraints += [
            SoC[t+1] == SoC[t] + eta_ch * P_batt_ch[t] * dt - (1/eta_dis) * P_batt_dis[t] * dt
        ]
        # Power balance: Load must be met by renewable + battery + grid
        constraints += [
            load[t] == P_solar_use[t] + P_wind_use[t] + P_batt_dis[t] + P_grid_imp[t] - P_grid_exp[t]
        ]
        # Device bounds
        constraints += [
            0 <= P_solar_use[t],
            P_solar_use[t] <= solar[t],
            0 <= P_wind_use[t],
            P_wind_use[t] <= wind[t],
            0 <= P_batt_ch[t],
            P_batt_ch[t] <= P_max,
            0 <= P_batt_dis[t],
            P_batt_dis[t] <= P_max,
            P_batt_ch[t] + P_batt_dis[t] <= P_max,  # prevents simultaneous max charge/discharge
            0 <= P_grid_imp[t],
            P_grid_imp[t] <= grid_imp_max_kw,
            0 <= P_grid_exp[t],
            P_grid_exp[t] <= grid_exp_max_kw
        ]


    # SoC bounds for all timesteps
    constraints += [SoC >= SoC_min, SoC <= SoC_max]
    # End-of-horizon SoC: battery should end above initial level
    constraints += [SoC[T] >= soc_init]

    # -------------------------
    # Step 5: Objective (Minimize grid import, encourage renewable usage)
    # -------------------------
    objective = cp.Minimize(
        cp.sum(
            grid_price_import * P_grid_imp * dt
            - grid_price_export * P_grid_exp * dt
            - 0.05 * (P_solar_use + P_wind_use) * dt   # reward using renewable energy
            + 0.01 * P_batt_ch * dt                     # penalize charging from grid unnecessarily
        )
    )


    # -------------------------
    # Step 6: Solve
    # -------------------------
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)

    print("Solver status:", prob.status)
    print("Objective cost:", prob.value)

    # -------------------------
    # Step 7: Post-processing
    # -------------------------
    P_pv_use_v = P_pv_use.value
    P_batt_ch_v = P_batt_ch.value
    P_batt_dis_v = P_batt_dis.value
    P_grid_imp_v = P_grid_imp.value
    P_grid_exp_v = P_grid_exp.value
    SoC_v = SoC.value

    # Energy summary
    def energy_sum(arr):
        return float(np.nansum(arr) * dt)
    summary = {
        # Grid
        "grid_import_kwh": energy_sum(P_grid_imp_v),
        "grid_export_kwh": energy_sum(P_grid_exp_v),
        "grid_net_kwh": energy_sum(P_grid_imp_v) - energy_sum(P_grid_exp_v),
        
        # Renewable usage
        "solar_used_kwh": energy_sum(P_solar_use.value),
        "wind_used_kwh": energy_sum(P_wind_use.value),
        "total_renewable_used_kwh": energy_sum(P_solar_use.value) + energy_sum(P_wind_use.value),
        "renewable_fraction_of_load": (energy_sum(P_solar_use.value) + energy_sum(P_wind_use.value)) / np.nansum(load) if np.nansum(load) > 0 else 0.0,
        
        # Battery
        "battery_charged_kwh": energy_sum(P_batt_ch_v),
        "battery_discharged_kwh": energy_sum(P_batt_dis_v),
        "battery_efficiency": energy_sum(P_batt_dis_v) / energy_sum(P_batt_ch_v) if energy_sum(P_batt_ch_v) > 0 else None,
        
        # SoC
        "SoC_min_kwh": float(np.nanmin(SoC_v)),
        "SoC_max_kwh": float(np.nanmax(SoC_v)),
        "SoC_avg_kwh": float(np.nanmean(SoC_v)),
        
        # Load
        "total_load_kwh": float(np.nansum(load) * dt)
    }

    # Collect summary per day
    daily_summary = summary.copy()
    daily_summary["date"] = str(DAY)
    
    # Store in dictionary keyed by date
    all_days_summary[str(DAY)] = daily_summary


    # Step 8: Optimized Plots
    # -------------------------
    x = np.arange(T)
    fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # --- Top: Power dispatch stackplot ---
    ax[0].stackplot(
        x,
        np.maximum(P_solar_use.value, 0.0),
        np.maximum(P_wind_use.value, 0.0),
        np.maximum(P_batt_dis_v, 0.0),
        np.maximum(P_grid_imp_v, 0.0),
        labels=["Solar_used", "Wind_used", "Battery_dis", "Grid_imp"],
        alpha=0.8
    )
    ax[0].plot(x, -np.maximum(P_batt_ch_v, 0.0), label="Battery_charge (neg)", color="tab:orange")
    ax[0].plot(x, -np.maximum(P_grid_exp_v, 0.0), label="Grid_export (neg)", color="tab:red")
    ax[0].plot(x, load, "k--", linewidth=1.2, label="Load")
    ax[0].legend(loc="upper right")
    ax[0].set_ylabel("Power (kW)")
    ax[0].set_title(f"Power Dispatch Breakdown {DAY}")
    ax[0].grid(alpha=0.3)

    # --- Bottom: SoC ---
    ax[1].plot(x, 100.0 * SoC_v[:-1]/C, label="SoC (%)", linewidth=1.5, color="tab:blue")
    ax[1].axhline(100*SoC_min/C, color="grey", linestyle="--", linewidth=1.0, label="SoC min")
    ax[1].axhline(100*SoC_max/C, color="grey", linestyle="-.", linewidth=1.0, label="SoC max")
    ax[1].set_xlabel("Timestep (15-min intervals)")
    ax[1].set_ylabel("SoC (%)")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"optim_combined_{DAY}.png", dpi=150)
    plt.close()



with open(OUTPUT_DIR / "optim_summary_all_days.json", "w") as f:
    json.dump(all_days_summary, f, indent=2)
    print("Plots and summary saved to:", OUTPUT_DIR)
