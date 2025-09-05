from pathlib import Path
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "cleaned_dataset.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "plots" / "scatter_plots_cleaned"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Seaborn style
sns.set_theme(style="whitegrid", context="talk")

# Timestamp format and sampling
DATETIME_FORMAT = "%Y/%m/%d %H:%M:%S"
# Clean first, then (optionally) resample in Pandas for plotting
RESAMPLE_EVERY = "5m"  # Accepts like "1m", "5m", "15m"; set to None to disable

# Cleaning strategies per column (mirrors d4_anomaly_cleaning.py)
CLEANING_STRATEGIES = {
	"battery_active_power": "remove_extremes_then_interpolate",
	"battery_active_power_set_response": "remove_extremes_then_interpolate",
	"pvpcs_active_power": "remove_extremes_then_interpolate",
	"ge_body_active_power": "remove_extremes_then_interpolate",
	"ge_active_power": "remove_extremes_then_interpolate",
	"ge_body_active_power_set_response": "remove_extremes_then_interpolate",
	"fc_active_power_fc_end_set": "remove_extremes_then_interpolate",
	"fc_active_power": "remove_extremes_then_interpolate",
	"fc_active_power_fc_end_set_response": "remove_extremes_then_interpolate",
	"island_mode_mccb_active_power": "remove_extremes_then_interpolate",
	"mg-lv-msb_ac_voltage": "remove_extremes_then_interpolate",
	"receiving_point_ac_voltage": "remove_extremes_then_interpolate",
	"island_mode_mccb_ac_voltage": "remove_extremes_then_interpolate",
	"island_mode_mccb_frequency": "remove_extremes_then_interpolate",
	"mg-lv-msb_frequency": "remove_extremes_then_interpolate",
	"inlet_temperature_of_chilled_water": "remove_extremes_then_interpolate",
	"outlet_temperature": "remove_extremes_then_interpolate",
}

TARGET_COLUMNS = list(CLEANING_STRATEGIES.keys())

# Anomaly detection params (mirrors d4)
Z_SCORE_THRESHOLD = 4.0
IQR_FACTOR = 3.0
MAD_THRESHOLD = 3.5

# ---- Cleaning helpers (lifted from d4_anomaly_cleaning.py, adapted minimal deps) ----
import pandas as pd
import numpy as np
from scipy import stats

# CLI
import argparse


def detect_sentinel_values(series, sentinel_values=[-999999.0]):
	if isinstance(series, pd.Series):
		mask = series.isin(sentinel_values)
	else:
		mask = pd.Series(False, index=series.index)
	return {
		"mask": mask,
		"count": int(mask.sum()),
		"values": series[mask].tolist() if isinstance(series, pd.Series) else [],
	}


def improved_anomaly_detection(series, method="combined"):
	sentinel_info = detect_sentinel_values(series)
	sentinel_mask = sentinel_info["mask"]

	clean_series = series[~sentinel_mask]

	if method == "zscore":
		z_scores = np.abs(stats.zscore(clean_series))
		anomaly_mask_clean = z_scores > Z_SCORE_THRESHOLD
	elif method == "iqr":
		Q1 = clean_series.quantile(0.25)
		Q3 = clean_series.quantile(0.75)
		IQR = Q3 - Q1
		lower_bound = Q1 - IQR_FACTOR * IQR
		upper_bound = Q3 + IQR_FACTOR * IQR
		anomaly_mask_clean = (clean_series < lower_bound) | (clean_series > upper_bound)
	elif method == "combined":
		z_mask = improved_anomaly_detection(clean_series, "zscore")["mask"]
		iqr_mask = improved_anomaly_detection(clean_series, "iqr")["mask"]
		anomaly_mask_clean = z_mask & iqr_mask

	full_mask = pd.Series(False, index=series.index)
	full_mask[sentinel_mask] = True
	full_mask.loc[clean_series.index] = anomaly_mask_clean

	return {
		"mask": full_mask,
		"count": int(full_mask.sum()),
		"sentinel_info": sentinel_info,
		"statistical_anomalies": int(getattr(anomaly_mask_clean, "sum", lambda: 0)()),
	}


def apply_cleaning_strategy(series, anomaly_mask, strategy):
	if strategy == "keep":
		return series.copy()
	elif strategy == "remove":
		cleaned = series.copy()
		cleaned[anomaly_mask] = np.nan
		return cleaned
	elif strategy == "remove_extremes_then_interpolate":
		cleaned = series.copy()
		cleaned[anomaly_mask] = np.nan
		# If index is datetime and monotonic, interpolate in time
		if not isinstance(cleaned.index, pd.DatetimeIndex):
			cleaned.index = pd.to_datetime(cleaned.index, errors='coerce')
		if cleaned.index.is_monotonic_increasing:
			cleaned = cleaned.interpolate(method='time')
		else:
			cleaned = cleaned.interpolate(method='linear')
		return cleaned
	elif strategy == "interpolate":
		cleaned = series.copy()
		cleaned[anomaly_mask] = np.nan
		return cleaned.interpolate(method='linear')
	else:
		return series.copy()

# ---- End cleaning helpers ----


def load_lazy() -> pl.LazyFrame:
	lf = pl.scan_parquet(DATA_PATH)
	lf = lf.rename({col: col.lower() for col in lf.columns})
	lf = lf.with_columns(
		pl.col("timestamp").str.strptime(pl.Datetime, DATETIME_FORMAT, strict=False)
	)
	return lf


def collect_sorted(lf: pl.LazyFrame, columns: list[str]) -> pl.DataFrame:
	available_cols = [c for c in columns if c in lf.columns]
	if "timestamp" not in lf.columns:
		raise ValueError("'timestamp' column missing after parsing")
	select_cols = ["timestamp", *available_cols]
	df = (
		lf.select(select_cols)
		.sort("timestamp")
		.collect(streaming=True)
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

	print("Applying anomaly cleaning (no files will be saved)...")
	pdf = clean_columns_pandas(pdf, TARGET_COLUMNS)

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
