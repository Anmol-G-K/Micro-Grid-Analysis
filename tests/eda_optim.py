# tests/file.py — LLM-friendly, optimized text-based EDA with time-series info

from pathlib import Path
import polars as pl
import numpy as np
import pandas as pd
import json
import pyarrow.feather as feather
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from scipy.stats import zscore

# ---------- Paths ----------
data_path = Path(__file__).resolve().parent.parent / "data/cleaned_dataset.parquet"  # or .feather
# data_path = Path(__file__).resolve().parent.parent / "data/cleaned_dataset.feather"  
output_dir = Path(__file__).resolve().parent.parent / "plots" / "metadata"
output_dir.mkdir(parents=True, exist_ok=True)

# ---------- Load data ----------
if not data_path.exists():
    raise FileNotFoundError(f"Data file not found at {data_path}")

if data_path.suffix == ".parquet":
    df = pl.read_parquet(data_path)
elif data_path.suffix == ".feather":
    # Read with PyArrow and convert to Polars
    table = feather.read_table(data_path)
    df = pl.from_arrow(table)
else:
    raise ValueError(f"Unsupported file format: {data_path.suffix}")

print(f"Data loaded: {df.shape}")

# ---------- Preprocessing ----------
ts_col = "Timestamp" if "Timestamp" in df.columns else None
num_cols = [c for c in df.columns if c != ts_col]
cat_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt in [pl.Categorical, pl.Utf8] and c != ts_col]

# Convert non-timestamp to float and handle sentinel values
for c in num_cols:
    df = df.with_columns([
        pl.when(pl.col(c) <= -999999)
          .then(None)
          .otherwise(pl.col(c).cast(pl.Float64))
          .alias(c)
    ])


# ---------- 0) Time-series frequency ----------
ts_freq = None
time_stats = {}
if ts_col:
    df_ts = df.select([ts_col]).to_pandas().sort_values(ts_col)
    df_ts[ts_col] = pd.to_datetime(df_ts[ts_col], format="%Y/%m/%d %H:%M:%S")
    df_ts = df_ts.dropna(subset=[ts_col])
    if len(df_ts) > 1:
        df_ts["diff"] = df_ts[ts_col].diff().dt.total_seconds() / 3600.0  # hours
        ts_freq = f"approx. every {df_ts['diff'].median():.2f} hours"

# ---------- 1) Summary Statistics ----------
summary_stats = df.select(num_cols).describe().to_dict(as_series=False)

# ---------- 2) Missing Values ----------
missing_pct = {c: float(df.select(pl.col(c).null_count())[0,0]) / df.height * 100 for c in num_cols}

# ---------- 3) Correlations ----------
corr_matrix = df.select(num_cols).to_pandas().corr().round(2).to_dict()
top_corrs = {}
for c in num_cols:
    top_corrs[c] = sorted(
        ((k, v) for k, v in corr_matrix[c].items() if k != c),
        key=lambda x: abs(x[1]), reverse=True
    )[:3]

# ---------- 4) Outlier Detection (Z-score) ----------
sample_n = min(50000, df.height)
df_sample = df.sample(n=sample_n, with_replacement=False)
z_arr = np.abs(zscore(df_sample.select(num_cols).to_numpy(), nan_policy="omit"))
outlier_count = int((z_arr > 3).any(axis=1).sum())
outlier_summary = {c: int((z_arr[:, i] > 3).sum()) for i, c in enumerate(num_cols)}

# ---------- 5) PCA ----------
X = df_sample.select(num_cols).to_numpy()
imputer = SimpleImputer(strategy="mean")  # or "median"
X_imputed = imputer.fit_transform(X)

n_comp = min(5, len(num_cols))
pca = PCA(n_components=n_comp, random_state=42)
pca.fit(X_imputed)
pca_var = pca.explained_variance_ratio_.round(4).tolist()

# ---------- 6) Discriminant Analysis ----------
lda_summary = None
if cat_cols:
    target = cat_cols[0]
    X_lda = df.select(num_cols).to_numpy()
    y_lda = df.select(target).to_numpy().ravel()
    try:
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_lda, y_lda)
        lda_summary = {
            "target_column": target,
            "classes": lda.classes_.tolist(),
            "explained_variance_ratio": lda.explained_variance_ratio_.round(4).tolist()
            if hasattr(lda, "explained_variance_ratio_") else None
        }
    except Exception as e:
        lda_summary = {"target_column": target, "error": str(e)}

# ---------- 7) Categorical Summaries ----------
cat_summary = {}
for c in cat_cols:
    counts = df.select(c).value_counts().to_dict()
    cat_summary[c] = counts

# ---------- 8) Time-series summaries per numeric column ----------
if ts_col:
    df_pandas = df.to_pandas()
    df_pandas[ts_col] = pd.to_datetime(df_pandas[ts_col], format="%Y/%m/%d %H:%M:%S")
    df_pandas = df_pandas.set_index(ts_col)
    for col in num_cols:
        s = df_pandas[col].dropna()
        if len(s) > 0:
            time_stats[col] = {
                "total": float(s.sum()),
                "mean_hourly": float(s.resample("h").mean().mean()),
                "max_value": float(s.max()),
                "time_of_max": str(s.idxmax()),
                "min_value": float(s.min()),
                "time_of_min": str(s.idxmin())
            }

# ---------- 9) Save LLM-friendly JSON ----------
eda_results = {
    "dataset_info": {
        "shape": df.shape,
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols
    },
    "summary_statistics": summary_stats,
    "missing_percent": missing_pct,
    "correlations": corr_matrix,
    "top_correlations": top_corrs,
    "outliers": {
        "zscore_threshold": 3,
        "rows_with_outliers": outlier_count,
        "per_column": outlier_summary
    },
    "pca": {
        "explained_variance_ratio": pca_var
    },
    "discriminant_analysis": lda_summary,
    "categorical_summary": cat_summary,
    "time_series_summary": {
        "timestamp_column": ts_col,
        "approx_frequency_hours": ts_freq,
        "per_numeric_column": time_stats
    }
}

output_file = output_dir / "eda_results.json"
with open(output_file, "w") as f:
    json.dump(eda_results, f, indent=4, sort_keys=True)
print(f"EDA complete. LLM-friendly results saved to: {output_file}")

# ---------- 10) Optional: Text summary ----------
txt_file = output_dir / "eda_summary.txt"
with open(txt_file, "w") as f:
    f.write(f"Dataset shape: {df.shape}\n")
    f.write(f"Numeric columns ({len(num_cols)}): {num_cols}\n")
    f.write(f"Categorical columns ({len(cat_cols)}): {cat_cols}\n")
    if ts_col:
        f.write(f"Timestamp column: {ts_col}, approx frequency: {ts_freq}\n")
    f.write("\nTop 5 numeric summary stats per column:\n")
    for c in num_cols[:5]:
        stats = summary_stats[c]
        f.write(f"{c}: {stats}\n")
    f.write(f"\nTotal rows with outliers (Z>|3|): {outlier_count}\n")
    f.write(f"PCA variance ratio (first {n_comp} components): {pca_var}\n")
    if lda_summary:
        f.write(f"Discriminant Analysis target: {lda_summary.get('target_column')}, classes: {lda_summary.get('classes')}\n")
    if ts_col:
        f.write("\nSample time-series summaries per numeric column:\n")
        for col, stats in list(time_stats.items())[:5]:
            f.write(f"{col}: {stats}\n")
print(f"Text summary saved to: {txt_file}")
