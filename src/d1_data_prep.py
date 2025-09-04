
from pathlib import Path
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

# =========================
# Paths
# =========================
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "cleaned_dataset.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "plots" / "data_prep"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ISSUES_FILE = OUTPUT_DIR / "column_null_counts.json"
INVALID_TS_FILE = OUTPUT_DIR / "missing_time_intervals.json"

# =========================
# 1. Load dataset
# =========================
df = pl.read_parquet(DATA_PATH)
print("Dataset loaded:", df.shape)

# =========================
# 2. Parse Timestamp & sort
# =========================
df = df.with_columns(
    pl.col("Timestamp").str.strptime(pl.Datetime, "%Y/%m/%d %H:%M:%S", strict=False).alias("Timestamp")
).sort("Timestamp")

# Filter out rows with null timestamps
df_filtered = df.filter(pl.col("Timestamp").is_not_null())

# =========================
# 3. Check timestamp continuity (gaps > 15 minutes)
# =========================
df_filtered = df_filtered.with_columns(
    (pl.col("Timestamp").diff().dt.total_seconds() / 60).alias("gap_minutes")
)

gaps = df_filtered.filter(pl.col("gap_minutes") > 15)
print(f"\n⏳ Found {gaps.shape[0]} gaps > 15 minutes")
if gaps.shape[0] > 0:
    print(gaps.select(["Timestamp", "gap_minutes"]).head(10))

# Save invalid timestamps if any
if gaps.shape[0] > 0:
    gaps.select(["Timestamp", "gap_minutes"]).write_json(INVALID_TS_FILE)
    print(f"🚨 Invalid timestamps saved to: {INVALID_TS_FILE}")

# =========================
# 4. Check duplicate timestamps
# =========================
dupes = (
    df_filtered.group_by("Timestamp")
      .count()
      .filter(pl.col("count") > 1)
)
print(f"\n🔁 Found {dupes.shape[0]} duplicate timestamps")

# =========================
# 5. Check missing values
# =========================
nulls = df_filtered.null_count()
print("\n❓ Null value counts per column:\n", nulls)

# Save column data issues
with open(ISSUES_FILE, "w") as f:
    json.dump(nulls.to_dict(as_series=False), f, indent=4)
print(f"Column data issues saved to: {ISSUES_FILE}")

# =========================
# 6. Missing fraction per day & per week
# =========================
def missing_fraction(df: pl.DataFrame, every: str = "1d") -> pl.DataFrame:
    return (
        df.group_by_dynamic("Timestamp", every=every)
          .agg([
              (pl.any_horizontal(pl.all().is_null()).mean().alias("missing_fraction"))
          ])
    )

daily_missing = missing_fraction(df_filtered, every="1d")
weekly_missing = missing_fraction(df_filtered, every="1w")

# =========================
# 7. Plot & Save
# =========================
plt.figure(figsize=(12, 4))
plt.plot(daily_missing["Timestamp"], daily_missing["missing_fraction"], marker="o")
plt.title("Daily Missing Fraction")
plt.ylabel("Fraction Missing")
plt.xlabel("Date")
plt.grid(True)
plt.tight_layout()
daily_plot_file = OUTPUT_DIR / "daily_missing_fraction.png"
plt.savefig(daily_plot_file)
plt.close()
print(f"Daily missing fraction plot saved to: {daily_plot_file}")

plt.figure(figsize=(12, 4))
plt.plot(weekly_missing["Timestamp"], weekly_missing["missing_fraction"], marker="o", color="orange")
plt.title("Weekly Missing Fraction")
plt.ylabel("Fraction Missing")
plt.xlabel("Week")
plt.grid(True)
plt.tight_layout()
weekly_plot_file = OUTPUT_DIR / "weekly_missing_fraction.png"
plt.savefig(weekly_plot_file)
plt.close()
print(f"Weekly missing fraction plot saved to: {weekly_plot_file}")

