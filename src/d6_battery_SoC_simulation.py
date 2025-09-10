# run_soc_sim.py
# Battery SoC simulation using Polars if available, fallback to pandas.
#   BASE / "plots" / "soc_sim"

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Try to import polars but gracefully fallback
use_polars = False
try:
    import polars as pl
    use_polars = True
except Exception:
    use_polars = False
    print("polars not available; falling back to pandas for parquet reading.")

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "cleaned_power_dataset.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "plots" / "soc_sim"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----- Simulation parameters (baseline) -----
DELTA_T = 1.0 / 60.0  # hours (1 minute)
ETA_CH = 0.95
ETA_DIS = 0.95
CAPACITY_KWH = 200.0  # baseline battery capacity determined from cleaned dataset max
P_MAX = 200.0         # kW, clip battery power to ±P_MAX
INITIAL_SOC_PCT = 50.0 # Initial Threshold to 50%
NOISE_THRESHOLD_KW = 0.05

MODE = "A"  # change to "B" if you want monthly outputs


# ----- Load or synthesize -----
data_df = None
if DATA_PATH.exists():
    try:
        print("Loading dataset from:", DATA_PATH)
        cols_needed = ["timestamp", "battery_active_power", "pvpcs_active_power", "ge_active_power", "island_mode_mccb_active_power"]

        if use_polars:
            # LazyFrame for scalability
            lf = pl.scan_parquet(str(DATA_PATH))

            # Safety check: make sure required cols are present
            schema_cols = lf.collect_schema().names()
            missing = [c for c in cols_needed if c not in schema_cols]
            if missing:
                raise ValueError(f"Missing columns in parquet: {missing}")

            # Resample with Polars
            pl_df = (
                lf
                .select(cols_needed)
                .sort("timestamp")
                .group_by_dynamic("timestamp", every="1m")
                .agg([pl.col(c).mean().alias(c) for c in cols_needed if c != "timestamp"])
                .collect()
            )

            pdf = pl_df.to_pandas()
            pdf["timestamp"] = pd.to_datetime(pdf["timestamp"], errors="coerce")
            pdf = pdf.set_index("timestamp").sort_index()

        else:
            # Pandas fallback
            pdf = pd.read_parquet(str(DATA_PATH), columns=cols_needed)
            pdf["timestamp"] = pd.to_datetime(pdf["timestamp"], errors="coerce")
            pdf = pdf.set_index("timestamp").sort_index()
            pdf = pdf.resample("1min").mean()

        # 🔍 Debug check
        print("\n🔍 Null check after resampling:")
        print(pdf[["battery_active_power", "pvpcs_active_power", "ge_active_power"]].isna().sum())

        data_df = pdf
        print("Loaded dataset successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset due to error: {e}")
else:
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Aborting.")


# ----- Main Function -----

def run_soc_simulation(sim_df: pd.DataFrame, label: str):
    # ----- Preprocess battery power -----
    batt = sim_df["battery_active_power"].copy()
    batt[np.abs(batt) < NOISE_THRESHOLD_KW] = 0.0
    batt = np.clip(batt, -P_MAX, P_MAX)
    sim_df["battery_active_power_clipped"] = batt

    # ----- SoC simulation -----
    n = len(sim_df)
    soc = np.zeros(n)
    soc[0] = INITIAL_SOC_PCT
    energy_charged_kwh = 0.0
    energy_discharged_kwh = 0.0

    for i in range(1, n):
        p_b = sim_df["battery_active_power_clipped"].iat[i]
        p_ch = max(0.0, -p_b)
        p_dis = max(0.0, p_b)
        e_ch = ETA_CH * p_ch * DELTA_T
        e_dis = (1.0 / ETA_DIS) * p_dis * DELTA_T
        energy_charged_kwh += e_ch
        energy_discharged_kwh += e_dis
        delta_E = e_ch - e_dis
        soc[i] = soc[i-1] + (delta_E / CAPACITY_KWH) * 100.0
        soc[i] = min(100.0, max(0.0, soc[i]))

    sim_df["soc_pct"] = soc

    # ----- Metrics and save outputs -----
    total_energy_charged = energy_charged_kwh
    total_energy_discharged = energy_discharged_kwh
    percent_time_at_0 = 100.0 * (sim_df["soc_pct"] == 0.0).sum() / n
    percent_time_at_100 = 100.0 * (sim_df["soc_pct"] == 100.0).sum() / n
    equivalent_full_cycles = (total_energy_charged + total_energy_discharged) / (2.0 * CAPACITY_KWH)
    round_trip_eff = (total_energy_discharged / total_energy_charged) if total_energy_charged > 0 else None

    # Save plots
    soc_plot_path = OUTPUT_DIR / f"soc_timeseries_{label}.png"
    plt.figure(figsize=(12,4))
    plt.plot(sim_df.index, sim_df["soc_pct"])
    plt.title(f"Battery State of Charge (%) - {label}")
    plt.xlabel("Time")
    plt.ylabel("SoC (%)")
    plt.tight_layout()
    plt.savefig(str(soc_plot_path), dpi=150)
    plt.close()

    energy_plot_path = OUTPUT_DIR / f"energy_timeseries_{label}.png"
    plt.figure(figsize=(12,4))
    plt.plot(sim_df.index, sim_df["ge_active_power"], label="Load (ge_active_power)")
    plt.plot(sim_df.index, sim_df["pvpcs_active_power"], label="PV (pvpcs_active_power)")
    plt.plot(sim_df.index, sim_df["battery_active_power_clipped"], label="Battery Active Power (kW)")
    plt.title(f"Load, PV, and Battery Power - {label}")
    plt.xlabel("Time")
    plt.ylabel("Power (kW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(energy_plot_path), dpi=150)
    plt.close()

    summary = {
        "simulation": {
            "start": str(sim_df.index[0]) if n > 0 else None,
            "end": str(sim_df.index[-1]) if n > 0 else None,
            "n_steps": int(n),
            "time_step_hours": DELTA_T,
            "capacity_kwh": CAPACITY_KWH,
            "p_max_kw": P_MAX,
            "eta_ch": ETA_CH,
            "eta_dis": ETA_DIS,
            "initial_soc_pct": INITIAL_SOC_PCT,
        },
        "metrics": {
            "total_energy_charged_kwh": total_energy_charged,
            "total_energy_discharged_kwh": total_energy_discharged,
            "percent_time_at_soc_0": percent_time_at_0,
            "percent_time_at_soc_100": percent_time_at_100,
            "equivalent_full_cycles": equivalent_full_cycles,
            "round_trip_efficiency_estimate": round_trip_eff,
        },
    }

    summary_path = OUTPUT_DIR / f"soc_summary_{label}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # # Save small previews
    # sim_df[["battery_active_power_clipped", "pvpcs_active_power", "ge_active_power", "soc_pct"]].iloc[:10].to_csv(
    #     OUTPUT_DIR / f"soc_preview_{label}_first10.csv"
    # )
    # sim_df[["battery_active_power_clipped", "pvpcs_active_power", "ge_active_power", "soc_pct"]].iloc[-10:].to_csv(
    #     OUTPUT_DIR / f"soc_preview_{label}_last10.csv"
    # )

    print(f"✔ {label} results saved.")



if __name__ == "__main__":
    # ----- Choose 1-month slice -----
    if MODE == "A":
        # ---- Option A: use entire dataset ----
        sim_df = data_df.copy()
        label = "full_dataset"
        print("\n▶ Running SoC simulation on entire dataset...")

        OUTPUT_DIR = Path(__file__).resolve().parent.parent / "plots" / "soc_sim" / "entire_dataset"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        run_soc_simulation(sim_df, label)

    elif MODE == "B":
        # ---- Option B: run month by month ----
        months = [
            "Apr_2023", "Aug_2022", "Dec_2022", "Feb_2023", "Jan_2023", 
            "Jul_2022", "Jul_2023", "Jun_2022", "Jun_2023", "Mar_2023", 
            "May_2022", "May_2023", "Nov_2022", "Oct_2022", "Sep_2022"
        ]

        for month in months:
            # Parse month into datetime
            dt = pd.to_datetime(month, format="%b_%Y")
            slice_start = dt.replace(day=1)
            slice_end = (slice_start + pd.offsets.MonthEnd(1))

            sim_df = data_df.loc[slice_start:slice_end].copy()
            label = month

            print(f"\n▶ Running SoC simulation for {month} ({slice_start.date()} → {slice_end.date()})")
            OUTPUT_DIR = Path(__file__).resolve().parent.parent / "plots" / "soc_sim" / "monthly"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            # --- call a helper function (see Step 3) to run simulation + save results ---
            run_soc_simulation(sim_df, label)

        print("\n✅ Monthly simulations completed. Files saved in:", OUTPUT_DIR)
        exit(0)  # stop here, we don’t run the rest again

    else:
        raise ValueError("Invalid MODE. Use 'A' or 'B'.")

