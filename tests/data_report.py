import polars as pl
import pandas as pd
import json
from pathlib import Path

# ------------------- Config -------------------
DATA_PATH = Path(__file__).resolve().parent.parent / "data/combined.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "plots" / "metadata"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ISSUES_FILE = OUTPUT_DIR / "column_data_issues.json"
INVALID_TS_FILE = OUTPUT_DIR / "invalid_timestamps.json"

# ------------------- Functions -------------------

def detect_timestamp_col(df: pl.DataFrame) -> str | None:
    """Try to detect a timestamp column by name or datatype."""
    candidates = [c for c in df.columns if "time" in c.lower() or "date" in c.lower() or "timestamp" in c.lower()]
    if candidates:
        return candidates[0]
    for c in df.columns:
        if isinstance(df.schema[c], pl.datatypes.Datetime):
            return c
    return None

def count_and_extract_invalid_timestamps(series: pd.Series, fmt: str) -> tuple[int, list[str]]:
    """Count how many entries fail to parse datetime and return those invalid entries."""
    parsed = pd.to_datetime(series, format=fmt, errors='coerce')
    invalid_mask = parsed.isna()
    invalid_values = series[invalid_mask].tolist()
    return int(invalid_mask.sum()), invalid_values

def detect_data_issues(df: pl.DataFrame, ts_col: str = None) -> tuple[dict, list[str]]:
    """Analyze column-wise issues: nulls, infs, datetime parse errors. Return issues and invalid timestamps."""
    issue_report = {}
    invalid_timestamps = []
    for col in df.columns:
        col_data = df[col]
        dtype = df.schema[col]

        nan_count = col_data.null_count()

        is_float = dtype in (pl.Float32, pl.Float64)

        inf_count = (
            ((col_data == float("inf")) | (col_data == float("-inf"))).sum()
            if is_float else 0
        )

        datetime_parse_errors = 0
        if ts_col and col == ts_col:
            col_series = df[col].to_pandas()
            datetime_parse_errors, invalid_timestamps = count_and_extract_invalid_timestamps(col_series, "%Y/%m/%d %H:%M:%S")

        issue_report[col] = {
            "dtype": str(dtype),
            "null_count": int(nan_count),
            "inf_count": int(inf_count),
            "datetime_parse_errors": int(datetime_parse_errors)
        }

    return issue_report, invalid_timestamps

def get_dataset_summary(df: pl.DataFrame, data_path: Path) -> dict:
    """Returns high-level dataset stats."""
    n_rows, n_cols = df.shape
    file_size_mb = data_path.stat().st_size / (1024 * 1024)

    return {
        "dataset_shape": {"rows": n_rows, "columns": n_cols},
        "total_cells": n_rows * n_cols,
        "file_size_mb": round(file_size_mb, 2)
    }

# ------------------- Main -------------------

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

    df = pl.read_parquet(DATA_PATH)
    ts_col = detect_timestamp_col(df)

    summary = get_dataset_summary(df, DATA_PATH)
    issues, invalid_timestamps = detect_data_issues(df, ts_col)

    report = {
        "summary": summary,
        "column_issues": issues
    }

    with open(ISSUES_FILE, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Column-wise data issue report saved to: {ISSUES_FILE}")

    if invalid_timestamps:
        with open(INVALID_TS_FILE, "w") as f:
            json.dump({"invalid_timestamps": invalid_timestamps}, f, indent=2)
        print(f"Invalid timestamps saved to: {INVALID_TS_FILE}")
    else:
        print("No invalid timestamps found.")

if __name__ == "__main__":
    main()
