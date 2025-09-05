# eda.py — Exploratory Data Analysis for Mesa del Sol microgrid

from pathlib import Path
import pandas as pd
import numpy as np
import os

# ---------- Paths ----------
csv_path = "data/combined.csv"
plots_dir = Path("plots")
plots_dir.mkdir(parents=True, exist_ok=True)

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found at {csv_path}")

# ---------- Load & basic cleaning ----------
# Parse timestamp; everything else as object -> we'll coerce to numeric
df = pd.read_csv(
    csv_path,
    parse_dates=["Timestamp"],
    infer_datetime_format=True,
    dayfirst=False
)

print(f"Dataset loaded with shape: {df.shape}")

# Keep Timestamp separate, coerce other columns to numeric
non_time_cols = [c for c in df.columns if c != "Timestamp"]
for c in non_time_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# The source uses -999999 as a sentinel for missing; turn it into NaN
for c in non_time_cols:
    if (df[c].min(skipna=True) is not None) and (df[c].min(skipna=True) <= -999999):
        df[c] = df[c].replace(-999999, np.nan)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# ---------- Matplotlib setup (non-interactive/headless safe) ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context="notebook", style="whitegrid")

# ---------- 1) Summary stats ----------
print("\nSummary Statistics:")
print(df[numeric_cols].describe())

# ---------- 2) Correlation heatmap ----------
plt.figure(figsize=(24, 18))   # Bigger figure so labels are readable
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
plt.title("Correlation Heatmap", fontsize=18)
plt.savefig(plots_dir / "correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()


# ---------- 3) Pairwise scatter (for ALL numeric vars, sampled) ----------
# Full pairplot on 4M rows would be unreadable/slow. Sample for clarity.
pp_sample_n = 2000
pp = df[numeric_cols].dropna().sample(
    n=min(pp_sample_n, len(df)), random_state=42
)
# Limit to at most 10 variables to keep the grid readable; pick by variance
vars_for_scatter = (
    pp[numeric_cols].var().sort_values(ascending=False).head(10).index.tolist()
    if len(numeric_cols) > 10 else numeric_cols
)

sns.pairplot(
    pp[vars_for_scatter],
    plot_kws={"s": 8, "alpha": 0.3, "edgecolor": "none"},
    diag_kind="hist"
)
plt.suptitle("Pairwise Scatter (sampled)", y=1.02)
plt.savefig(plots_dir / "pairwise_scatter.png", dpi=300, bbox_inches="tight")
plt.close()
# ---------- 3b) Individual Scatter Plots (all numeric pairs) ----------
scatter_folder = plots_dir / "scatter_plots"
scatter_folder.mkdir(parents=True, exist_ok=True)
top_n = 10
selected_cols = df[numeric_cols].var().sort_values(ascending=False).head(top_n).index.tolist()

for i in range(len(selected_cols)):
    for j in range(i + 1, len(selected_cols)):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x=df[selected_cols[i]],
            y=df[selected_cols[j]],
            alpha=0.6,
            edgecolor=None
        )
        plt.title(f"Scatter: {selected_cols[i]} vs {selected_cols[j]}", fontsize=14)
        plt.xlabel(selected_cols[i])
        plt.ylabel(selected_cols[j])
        plt.tight_layout()
        plt.savefig(
            scatter_folder / f"{selected_cols[i]}_vs_{selected_cols[j]}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()


# # ---------- 4) Histograms (clipped tails to show shape) ----------
# Extreme outliers (e.g., sentinels, spikes) can flatten histograms.
# Clip each column to 1st–99th percentile for visualization only.
clipped = df[numeric_cols].copy()
q1 = clipped.quantile(0.01)
q99 = clipped.quantile(0.99)
clipped = clipped.clip(lower=q1, upper=q99, axis=1)

n = len(clipped.columns)
ncols = 4
nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows))
axes = np.ravel(axes) if n > 1 else [axes]
for i, col in enumerate(clipped.columns):
    ax = axes[i]
    ax.hist(clipped[col].dropna(), bins=40)
    ax.set_title(col)
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
for j in range(i + 1, len(axes)):
    axes[j].axis("off")
fig.suptitle("Histograms (1–99% clipped for readability)", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(plots_dir / "histograms.png", dpi=300)
plt.close(fig)
# ---------- 4b) Individual Histograms (seaborn, styled, per feature) ----------
hist_folder = plots_dir / "histograms"
hist_folder.mkdir(parents=True, exist_ok=True)
top_n = 10
selected_cols = df[numeric_cols].var().sort_values(ascending=False).head(top_n).index.tolist()

for col in selected_cols:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col].dropna(), kde=True, bins=30, color="skyblue")
    plt.title(f"Histogram of {col}", fontsize=14)
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(hist_folder / f"{col}_histogram.png", dpi=300, bbox_inches="tight")
    plt.close()


# ---------- 5) Box plots (stacked, readable) ----------
plt.figure(figsize=(24, 14))
# Use clipped data again so whiskers are interpretable on one image
sns.boxplot(data=clipped, orient="h", fliersize=1, palette="husl")
plt.title("Box Plots (clipped 1-99% for display)")
plt.tight_layout()
plt.savefig(plots_dir / "box_plots.png", dpi=300)
plt.close()

# ---------- 6) Missingness (bar chart of % missing) ----------
missing_pct = df[numeric_cols].isna().mean().sort_values(ascending=True) * 100
plt.figure(figsize=(10, max(6, 0.35 * len(missing_pct))))
missing_pct.plot(kind="barh")
plt.xlabel("Percent Missing")
plt.title("Missing Values by Column")
plt.tight_layout()
plt.savefig(plots_dir / "missing_values_bar.png", dpi=300)
plt.close()

# ---------- 7) Time-series decomposition (robust) ----------
# Use a representative power signal if present; else the first numeric col.
target_col = None
preferred = [
    "GE_Active_Power",
    "Battery_Active_Power",
    "PVPCS_Active_Power",
]
for cand in preferred:
    if cand in df.columns:
        target_col = cand
        break
if target_col is None and numeric_cols:
    target_col = numeric_cols[0]

if target_col is not None:
    from statsmodels.tsa.seasonal import seasonal_decompose

    ts = df[["Timestamp", target_col]].dropna()
    # Ensure datetime and sort
    ts["Timestamp"] = pd.to_datetime(ts["Timestamp"], errors="coerce")
    ts = ts.dropna(subset=["Timestamp"]).sort_values("Timestamp")
    ts = ts.set_index("Timestamp")[target_col]

    # Keep only finite values
    ts = ts[np.isfinite(ts)]

    # Resample to hourly mean for a clean regular frequency
    ts_hourly = ts.resample("1H").mean().dropna()

    # Need at least 2 periods; use daily seasonality (24 hours) if long enough
    if len(ts_hourly) >= 48:
        result = seasonal_decompose(ts_hourly, model="additive", period=24)
        fig = result.plot()
        fig.set_size_inches(24, 14)
        fig.suptitle(f"Hourly Seasonal Decomposition: {target_col}", y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(plots_dir / f"time_series_decomposition_{target_col}.png", dpi=300)
        plt.close(fig)
    else:
        print("Skipping decomposition: insufficient hourly data (need >= 48 points)")

# ---------- 8) Multicollinearity & PCA (sampled to keep it fast) ----------
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA

# Sample rows for numeric analyses to control runtime/memory
anal_sample_n = 50000
X_full = df[numeric_cols].dropna()
X = X_full.sample(n=min(anal_sample_n, len(X_full)), random_state=42) if len(X_full) else X_full

if X.shape[1] >= 2 and len(X) >= 10:
    # VIF
    vif = pd.DataFrame({
        "Feature": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    }).sort_values("VIF", ascending=False)
    print("\nVariance Inflation Factor (VIF) on sampled data:")
    print(vif)

    # PCA (keep up to 5 components or <= n_features)
    n_comp = min(5, X.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(X)
    print("\nPCA Explained Variance Ratio:", pca.explained_variance_ratio_)

# ---------- 9) Outliers via Z-score (on sampled data) ----------
from scipy.stats import zscore
if X.shape[1] >= 1 and len(X) >= 10:
    z = np.abs(zscore(X, nan_policy="omit"))
    outlier_rows = (z > 3).any(axis=1)
    print(f"\nOutlier rows (|Z| > 3) in sampled data: {int(outlier_rows.sum())} of {len(X)}")

print(f"\nEDA complete. Plots saved to: {plots_dir.resolve().name}/")
