from pathlib import Path
import polars as pl
import pandas as pd
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "cleaned_power_dataset.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "plots" / "scatter_plots_cleaned"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Seaborn style
sns.set_theme(style="whitegrid", context="talk")

# Timestamp format and sampling
DATETIME_FORMAT = "%Y/%m/%d %H:%M:%S"
# Clean first, then (optionally) resample in Pandas for plotting
RESAMPLE_EVERY = "5m"  # Accepts like "1m", "5m", "15m"; set to None to disable

TARGET_COLUMNS = [
    "battery_active_power",
    "battery_active_power_set_response",
    "pvpcs_active_power",
    "ge_body_active_power",
    "ge_active_power",
    "ge_body_active_power_set_response",
    "fc_active_power_fc_end_set",
    "fc_active_power",
    "fc_active_power_fc_end_set_response",
    "island_mode_mccb_active_power",
    "mg-lv-msb_ac_voltage",
    "receiving_point_ac_voltage",
    "island_mode_mccb_ac_voltage",
    "island_mode_mccb_frequency",
    "mg-lv-msb_frequency",
    "inlet_temperature_of_chilled_water",
    "outlet_temperature",
]

def load_lazy() -> pl.LazyFrame:
    lf = pl.scan_parquet(DATA_PATH)
    lf = lf.rename({col: col.lower() for col in lf.columns})
    # NO timestamp parsing here!
    return lf



def collect_sorted(lf: pl.LazyFrame, columns: list[str]) -> pl.DataFrame:
    available_cols = [c for c in columns if c in lf.columns]
    if "timestamp" not in lf.columns:
        raise ValueError("'timestamp' column missing after parsing")
    select_cols = ["timestamp", *available_cols]
    df = (
        lf.select(select_cols)
        .sort("timestamp")
        .collect()
    )
    return df



def clean_columns_pandas(pdf: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
	# Assumes pdf has a DatetimeIndex of timestamp
	for col in columns:
		if col not in pdf.columns:
			print(f"Skipping {col}: not found in dataset")
			continue
		anomaly_info = improved_anomaly_detection(pdf[col])
		strategy = CLEANING_STRATEGIES.get(col, "interpolate")
		pdf[col] = apply_cleaning_strategy(pdf[col], anomaly_info["mask"], strategy)
	return pdf


def plot_columns(pdf: pd.DataFrame, columns: list[str]) -> None:
	# pdf is indexed by timestamp
	for col in columns:
		if col not in pdf.columns:
			continue
		series = pdf[col].dropna()
		if series.empty:
			print(f"Skipping {col}: no non-null values to plot")
			continue
		plt.figure(figsize=(14, 6))
		plt.plot(series.index, series.values, lw=0.8, alpha=0.9)
		plt.title(f"Timestamp vs {col} (Cleaned)")
		plt.xlabel("Timestamp")
		plt.ylabel(col)
		plt.xticks(rotation=45)
		plt.grid(alpha=0.25)
		sns.despine()
		outfile = OUTPUT_DIR / f"scatter_Timestamp_vs_{col.replace(' ', '_').replace('/', '-')}.png"
		plt.savefig(outfile, dpi=300, bbox_inches="tight")
		plt.close()
		print(f"Saved: {outfile}")


def normalize_offset(offset: str | None) -> str | None:
	if offset is None:
		return None
	o = offset.strip().lower()
	if o in {"none", "", "0", "off"}:
		return None
	# Map common minute shorthand (m) to pandas 'T'
	if o.endswith("m"):
		num = o[:-1]
		return f"{num}T"
	# seconds (s) and hours (h) pass through as-is if valid like '30s'/'1h'
	return o


def parse_args():
	parser = argparse.ArgumentParser(description="Clean then plot time-series columns vs timestamp.")
	parser.add_argument("--resample", type=str, default=RESAMPLE_EVERY,
		help="Resample interval (e.g., 1m, 5m, 15m). Use 'none' to disable.")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	interval = normalize_offset(args.resample)

	print("Loading dataset (lazy) and sorting by timestamp...")
	lf = load_lazy()
	df_pl = collect_sorted(lf, TARGET_COLUMNS)
	print(f"Polars collected shape: {df_pl.shape}")

	# Convert to pandas and set timestamp as index
	pdf = df_pl.to_pandas()
	pdf["timestamp"] = pd.to_datetime(pdf["timestamp"], format=DATETIME_FORMAT, errors="coerce")
	pdf = pdf.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")

	print(f"Dataset Name: {DATA_PATH.name}")

	# Optional resampling AFTER cleaning
	if interval:
		print(f"Resampling cleaned data by: {interval}")
		pdf = pdf.resample(interval).mean()
	else:
		print("Resampling disabled; plotting full-resolution cleaned data.")

	print("Generating plots...")
	plot_columns(pdf, TARGET_COLUMNS)
	print(f"Plots saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
	main()
