# issues infered from tests/data_report.py mainly date time parsing

import polars as pl
import pandas as pd
from pathlib import Path

# ------------------- Config -------------------
DATA_PATH = Path(__file__).resolve().parent.parent / "data/combined.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATHER_FILE = OUTPUT_DIR / "cleaned_dataset.feather"
PARQUET_FILE = OUTPUT_DIR / "cleaned_dataset.parquet"

DATETIME_FORMAT = "%Y/%m/%d %H:%M:%S"

# ------------------- Functions -------------------

def detect_timestamp_col(df: pl.DataFrame) -> str | None:
    """Detect timestamp column by name or Polars datetime type."""
    candidates = [c for c in df.columns if "time" in c.lower() or "date" in c.lower() or "timestamp" in c.lower()]
    if candidates:
        return candidates[0]
    for col, dtype in df.schema.items():
        if isinstance(dtype, pl.datatypes.Datetime):
            return col
    return None

def filter_valid_timestamps(df: pl.DataFrame, ts_col: str, fmt: str) -> pl.DataFrame:
    """Remove rows with invalid timestamps based on strict datetime parsing."""
    # Use pandas just to validate timestamp strings
    ts_series = df[ts_col].to_pandas()
    parsed = pd.to_datetime(ts_series, format=fmt, errors="coerce")
    valid_mask = ~parsed.isna()

    # Use Polars to filter based on mask
    return df.filter(pl.Series(valid_mask.values))

def save_dataset(df: pl.DataFrame, feather_path: Path, parquet_path: Path):
    """Save DataFrame to both Feather and Parquet."""
    df.write_ipc(feather_path)
    df.write_parquet(parquet_path)
    print(f"Saved Feather: {feather_path}")
    print(f"Saved Parquet: {parquet_path}")

# ------------------- Main -------------------

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    df = pl.read_parquet(DATA_PATH)
    ts_col = detect_timestamp_col(df)

    if ts_col is None:
        print("No timestamp column detected. Skipping timestamp validation.")
        cleaned_df = df
    else:
        print(f"Timestamp column detected: {ts_col}")
        cleaned_df = filter_valid_timestamps(df, ts_col, DATETIME_FORMAT)
        removed = df.height - cleaned_df.height
        print(f"Removed {removed} rows with invalid timestamps.")

    save_dataset(cleaned_df, FEATHER_FILE, PARQUET_FILE)

if __name__ == "__main__":
    main()
