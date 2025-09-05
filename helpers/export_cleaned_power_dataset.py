"""
Export Cleaned Power Dataset Using Polars

This script loads raw power data, applies anomaly detection and cleaning
strategies from the imported module, and exports the fully cleaned
dataset in both Parquet and Feather formats.

Optimized for large datasets using Polars (~70M rows).

Author: Your Name
"""

import sys
from pathlib import Path
import polars as pl

# Directories
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "data"
SRC_DIR = ROOT_DIR/ "src"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR)) # dont change this 

# --- Step 3: Import your module from src
from d4_anomaly_cleaning import (
    load_and_preprocess_data,
    improved_anomaly_detection,
    apply_cleaning_strategy,
    CLEANING_STRATEGIES,
)


# --- Step 1: Load data
print("🔄 Loading and preprocessing raw dataset using Polars...")
df = load_and_preprocess_data()
print(f"📊 Dataset shape before cleaning: {df.shape}")

# --- Step 2: Clean target columns
TARGET_COLUMNS = list(CLEANING_STRATEGIES.keys())
cleaned_columns = []

for column in TARGET_COLUMNS:
    if column not in df.columns:
        print(f"⏭️  Skipping '{column}' (not found)")
        continue

    print(f"🧼 Cleaning column: {column}")
    strategy = CLEANING_STRATEGIES.get(column, "interpolate")

    series_pd = df[column].to_pandas()
    anomaly_info = improved_anomaly_detection(series_pd)
    anomaly_mask = anomaly_info["mask"]

    cleaned_pd = apply_cleaning_strategy(series_pd, anomaly_mask, strategy)

    cleaned_pl = pl.Series(name=column, values=cleaned_pd.tolist())
    df = df.with_columns(cleaned_pl)
    cleaned_columns.append(column)

print(f"📊 Dataset shape after cleaning:  {df.shape}")

# --- Step 3: Save to disk
parquet_path = OUTPUT_DIR / "cleaned_power_dataset.parquet"
feather_path = OUTPUT_DIR / "cleaned_power_dataset.feather"

print("\n💾 Saving cleaned dataset...")
df.write_parquet(parquet_path)
df.write_ipc(feather_path)  # Feather format (Apache Arrow IPC)

print(f"✅ Cleaned dataset saved:\n  • {parquet_path}\n  • {feather_path}")