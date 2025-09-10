# final_task_with_imports.py
import polars as pl
import pandas as pd
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import importlib.util
import json
from pathlib import Path
import numpy as np

# --- Paths ---
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "cleaned_power_dataset.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "plots" / "optimisation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Dynamic module loader (no sys.path changes) ---
def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

d5 = load_module("d5", Path("d5_forecasting_preparation.py"))
d6 = load_module("d6", Path("d6_battery_SoC_simulation.py"))

# -------------------------
# Step 1: Prepare a 1-day forecast slice
# -------------------------
# Use the preloaded data_df from d6 (1-min); downsample to 15-min
df = d6.data_df.copy()
df_15min = (
    df.resample("15min")
      .mean()[["ge_active_power", "pvpcs_active_power"]]
      .rename(columns={"ge_active_power": "load_kw",
                       "pvpcs_active_power": "pv_kw"})
)

# simple (pandas) lag/calendar features (not required for optimisation)
for col in ["load_kw", "pv_kw"]:
    df_15min[f"{col}_t-1"] = df_15min[col].shift(1)
df_15min["hour"] = df_15min.index.hour
df_15min["dayofweek"] = df_15min.index.dayofweek
df_15min["month"] = df_15min.index.month

# Choose day (configurable)
DAY = "2022-06-15"
day_slice = df_15min.loc[DAY]

load = day_slice["load_kw"].values.astype(float)
pv = day_slice["pv_kw"].values.astype(float)
T = len(load)
dt = 15 / 60.0  # 15 minutes in hours

# -------------------------
# Step 2: Battery & device params (from d6)
# -------------------------
C = d6.CAPACITY_KWH          # kWh
eta_ch = d6.ETA_CH          # charging eff
eta_dis = d6.ETA_DIS        # discharging eff
P_max = d6.P_MAX            # kW battery inverter/power limit
SoC0 = C * d6.INITIAL_SOC_PCT / 100.0
SoC_min = 0.10 * C          # usable min (10%)
SoC_max = 0.90 * C          # usable max (90%)

# External device params (tweak these)
fc_max_kw = 50.0            # fuel cell max power (kW)
grid_imp_max_kw = 1e6      # large if unlimited
grid_exp_max_kw = 1e6      # large if allowed
# Cost parameters (example values — tune for your region)
grid_price_import = 0.12    # $/kWh
grid_price_export = 0.03    # $/kWh (credit for export)
fc_cost = 0.15              # $/kWh for running fuel cell

# -------------------------
# Step 3: Decision variables
# -------------------------
P_batt_ch = cp.Variable(T, nonneg=True)     # kW
P_batt_dis = cp.Variable(T, nonneg=True)    # kW
P_fc = cp.Variable(T, nonneg=True)          # kW
P_pv_use = cp.Variable(T, nonneg=True)      # kW (PV used locally)
P_grid_imp = cp.Variable(T, nonneg=True)    # kW
P_grid_exp = cp.Variable(T, nonneg=True)    # kW (export)

SoC = cp.Variable(T+1)                       # kWh

# -------------------------
# Step 4: Constraints
# -------------------------
constraints = []
# initial SoC
constraints += [SoC[0] == SoC0]

for t in range(T):
    # SoC dynamics (kWh)
    constraints += [
        SoC[t+1] == SoC[t] + eta_ch * P_batt_ch[t] * dt - (1.0/eta_dis) * P_batt_dis[t] * dt
    ]

    # power balance: Load = PV_used + Batt_dis - Batt_ch + FC + Grid_import - Grid_export
    constraints += [
        load[t] == P_pv_use[t] + P_batt_dis[t] - P_batt_ch[t] + P_fc[t] + P_grid_imp[t] - P_grid_exp[t]
    ]

    # device bounds
    constraints += [
        0 <= P_pv_use[t],
        P_pv_use[t] <= pv[t],            # cannot use more PV than available
        P_batt_ch[t] <= P_max,
        P_batt_dis[t] <= P_max,
        P_batt_ch[t] + P_batt_dis[t] <= P_max,  # prevents simultaneous full charge & discharge
        P_fc[t] <= fc_max_kw,
        P_grid_imp[t] <= grid_imp_max_kw,
        P_grid_exp[t] <= grid_exp_max_kw
    ]

# SoC bounds for all timesteps
constraints += [SoC >= SoC_min, SoC <= SoC_max]

# End-of-horizon SoC constraint (keep battery level at or above starting level)
constraints += [SoC[T] >= SoC0]

# -------------------------
# Step 5: Objective (cost-based)
# -------------------------
# Minimise grid import cost - export credit + FC running cost
obj_terms = []
obj_terms.append(grid_price_import * cp.sum(P_grid_imp) * dt)
obj_terms.append(-grid_price_export * cp.sum(P_grid_exp) * dt)
obj_terms.append(fc_cost * cp.sum(P_fc) * dt)

objective = cp.Minimize(cp.sum(obj_terms))

# -------------------------
# Step 6: Solve
# -------------------------
solver_results = {}
for solver in [cp.SCS]:
    prob = cp.Problem(objective, constraints)
    try:
        result = prob.solve(solver=solver, verbose=False)
        solver_results[solver] = {
            "objective": prob.value,
            "P_grid_imp": P_grid_imp.value.flatten().tolist(),
            "P_grid_exp": P_grid_exp.value.flatten().tolist(),
            "P_batt_dis": P_batt_dis.value.flatten().tolist(),
            "P_batt_ch": P_batt_ch.value.flatten().tolist(),
            "SoC": SoC.value.flatten().tolist(),
        }
        print(f"✔ Solver {solver}: objective = {prob.value:.4f}")
    except Exception as e:
        print(f"✖ Solver {solver} failed: {e}")

print("Solver status:", prob.status)
print("Objective value (cost):", result)

# -------------------------
# Step 7: Post-processing & sanity checks
# -------------------------
# flatten CVXPY outputs
def flat(x):
    if x is None:
        return None
    a = np.array(x)
    return a.flatten()

P_pv_use_v = flat(P_pv_use.value)
P_batt_ch_v = flat(P_batt_ch.value)
P_batt_dis_v = flat(P_batt_dis.value)
P_fc_v = flat(P_fc.value)
P_grid_imp_v = flat(P_grid_imp.value)
P_grid_exp_v = flat(P_grid_exp.value)
SoC_v = flat(SoC.value)

# sanity stats (kWh)
grid_imp_kwh = float(np.nansum(P_grid_imp_v) * dt)
grid_exp_kwh = float(np.nansum(P_grid_exp_v) * dt)
pv_used_kwh = float(np.nansum(P_pv_use_v) * dt)
batt_ch_kwh = float(np.nansum(P_batt_ch_v) * dt)
batt_dis_kwh = float(np.nansum(P_batt_dis_v) * dt)
fc_kwh = float(np.nansum(P_fc_v) * dt)

print("Grid import kWh:", grid_imp_kwh)
print("Grid export kWh:", grid_exp_kwh)
print("PV used kWh:", pv_used_kwh)
print("Battery charged kWh:", batt_ch_kwh)
print("Battery discharged kWh:", batt_dis_kwh)
print("FC energy kWh:", fc_kwh)
print("SoC min/max (kWh):", float(np.nanmin(SoC_v)), float(np.nanmax(SoC_v)))

# Save summary JSON
summary = {
    "day": DAY,
    "dt_hours": dt,
    "objective_cost": result,
    "solver_status": prob.status,
    "energy_kwh": {
        "grid_import_kwh": grid_imp_kwh,
        "grid_export_kwh": grid_exp_kwh,
        "pv_used_kwh": pv_used_kwh,
        "battery_charged_kwh": batt_ch_kwh,
        "battery_discharged_kwh": batt_dis_kwh,
        "fc_kwh": fc_kwh
    },
    "soc": {
        "initial_soc_kwh": float(SoC0),
        "min_soc_kwh": float(np.nanmin(SoC_v)),
        "max_soc_kwh": float(np.nanmax(SoC_v))
    }
}
with open(OUTPUT_DIR / f"optim_summary_{DAY}.json", "w") as f:
    json.dump(summary, f, indent=2)

with open(OUTPUT_DIR / f"solver_comparison_{DAY}.json", "w") as f:
    json.dump(solver_results, f, indent=2)


# -------------------------
# Step 8: Plots (use flattened arrays)
# -------------------------
x = np.arange(T)

# Stacked area: PV used, battery discharge, FC, grid import
plt.figure(figsize=(14,5))
plt.stackplot(
    x,
    np.maximum(P_pv_use_v, 0.0),
    np.maximum(P_batt_dis_v, 0.0),
    np.maximum(P_fc_v, 0.0),
    np.maximum(P_grid_imp_v, 0.0),
    labels=["PV_used", "Battery_dis", "FuelCell", "Grid_imp"]
)
# plot battery charge (as negative bars/line)
plt.plot(x, -np.maximum(P_batt_ch_v, 0.0), label="Battery_charge (neg)")
# plot grid export as negative (if any)
plt.plot(x, -np.maximum(P_grid_exp_v, 0.0), label="Grid_export (neg)")
plt.plot(x, load, "k--", label="Load")
plt.legend(loc="upper right")
plt.xlabel("Timestep")
plt.ylabel("Power (kW)")
plt.title(f"Dispatch {DAY}")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"dispatch_{DAY}.png", dpi=150)
plt.close()

# SoC plot (in %)
plt.figure(figsize=(14,3))
plt.plot(x, 100.0 * SoC_v[:-1] / C, label="SoC (%)")
plt.axhline(100*SoC_min/C, color="grey", linestyle="--", linewidth=0.8, label="SoC min")
plt.axhline(100*SoC_max/C, color="grey", linestyle="-.", linewidth=0.8, label="SoC max")
plt.legend()
plt.xlabel("Timestep")
plt.ylabel("SoC (%)")
plt.title(f"SoC {DAY}")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"soc_{DAY}.png", dpi=150)
plt.close()

print("Plots and summary saved to:", OUTPUT_DIR)
