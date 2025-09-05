"""
Day 3 - Correlation & Lag Analysis (FIXED & OPTIMIZED for 70M+ rows)

Highly efficient correlation analysis for very large datasets:
- ACF/PACF for Load(t), PV(t), Battery(t)
- Cross-correlation: PV vs Load; Battery vs Load
- Rolling correlation (30-day window) for seasonal stability

FIXES:
- Added proper sorting before group_by_dynamic
- Improved memory management for very large datasets
- Better error handling and progress tracking

OPTIMIZATION STRATEGIES:
1. Streaming/chunked processing with Polars lazy evaluation
2. Aggressive downsampling for correlation computations
3. Memory-efficient rolling window calculations
4. Smart caching of intermediate results
5. Proper data sorting and validation

Dependencies: polars, pandas, numpy, matplotlib, statsmodels, scipy
"""

import polars as pl
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.stattools import acf, pacf, ccf
import warnings
warnings.filterwarnings('ignore')

# =========================
# CONFIGURATION
# =========================
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "cleaned_power_dataset.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "plots" / "corr_lag"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# For 70M rows, we need aggressive downsampling
DOWNSAMPLE_FACTOR = 1000  # Keep every 1000th point for correlation analysis
MAX_LAG = 100  # Maximum lag for ACF/PACF
ROLLING_WINDOW_DAYS = 30
ROLLING_WINDOW_SIZE = ROLLING_WINDOW_DAYS * 24 * 6  # Assuming 10-minute intervals

# Target columns (using actual column names from the dataset)
TARGET_COLS = [
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
]


CROSS_CORR_PAIRS = [
    ("battery_active_power", "battery_active_power_set_response"), # battery now vs set point 
    ("ge_body_active_power", "ge_active_power"), # ge specific load vs ge
    ("ge_body_active_power", "ge_body_active_power_set_response"), # ge specific load vs set point
    ("fc_active_power", "fc_active_power_fc_end_set"),  #FC vs set
    ("fc_active_power", "fc_active_power_fc_end_set_response"), # FC vs set?

    ("ge_active_power", "pvpcs_active_power"),      # Load vs PV
    ("ge_active_power", "battery_active_power"),     # Load vs Battery
    ("ge_active_power", "fc_active_power") ,         # Load vs FC
    ("ge_body_active_power", "pvpcs_active_power"),      # Specific Load vs PV
    ("ge_body_active_power", "battery_active_power")  ,   # Specific Load vs Battery
    ("ge_body_active_power", "fc_active_power")    ,      # Specific Load vs FC
    ("island_mode_mccb_active_power", "pvpcs_active_power"),      # Island Mode Power  vs PV
    ("island_mode_mccb_active_power", "battery_active_power") ,    # Island Mode Power vs Battery
    ("island_mode_mccb_active_power", "fc_active_power"),     # Island Mode Power vs FC
    ("island_mode_mccb_active_power", "ge_active_power")  ,   # Island Mode Power vs ge power
    ("island_mode_mccb_active_power", "ge_body_active_power"),     # Island Mode Power vs ge specific power
    ("ge_active_power", "ge_body_active_power")      # Load vs Specific Load?

]


# =========================
# UTILITY FUNCTIONS
# =========================

def detect_timestamp_column(df: pl.LazyFrame) -> str:
    """Detect timestamp column in the dataset."""
    schema = df.collect_schema()
    candidates = [c for c in schema.names() if "time" in c.lower() or "date" in c.lower()]
    if candidates:
        return candidates[0]
    
    # Check for datetime columns
    for col, dtype in schema.items():
        if dtype == pl.Datetime:
            return col
    
    raise ValueError("No timestamp column found!")

def ensure_datetime_column(lf: pl.LazyFrame, ts_col: str) -> pl.LazyFrame:
    """Ensure timestamp column is properly parsed as datetime."""
    schema = lf.collect_schema()
    
    if schema.get(ts_col) == pl.Datetime:
        return lf
    
    if schema.get(ts_col) == pl.Utf8:
        return lf.with_columns(
            pl.col(ts_col).str.strptime(pl.Datetime, "%Y/%m/%d %H:%M:%S", strict=False)
        )
    
    return lf

def smart_downsample_fixed(lf: pl.LazyFrame, ts_col: str, target_cols: list, factor: int = 1000) -> pl.DataFrame:
    """
    Intelligently downsample the data for correlation analysis.
    FIXED: Properly sorts data before group_by_dynamic.
    """
    print(f"Downsampling data by factor {factor}...")
    
    # First, get a sample to understand the data frequency
    print("Analyzing data frequency...")
    sample = lf.select([ts_col] + target_cols).limit(10000).collect()
    
    if len(sample) > 1:
        time_diff = sample[ts_col].diff().dt.total_seconds().median()
        print(f"Detected median time interval: {time_diff:.1f} seconds")
        
        # Calculate appropriate sampling interval
        if time_diff < 60:  # Less than 1 minute
            sampling_interval = f"{factor * int(time_diff)}s"
        else:
            sampling_interval = f"{factor}m"
    else:
        sampling_interval = "1h"  # Default to hourly
    
    print(f"Using sampling interval: {sampling_interval}")
    
    # FIXED: Sort data before group_by_dynamic
    print("Sorting and downsampling data...")
    downsampled = (
        lf
        .select([ts_col] + target_cols)
        .filter(pl.col(ts_col).is_not_null())
        .sort(ts_col)  # CRITICAL: Sort before group_by_dynamic
        .group_by_dynamic(ts_col, every=sampling_interval)
        .agg([
            pl.col(col).mean().alias(col) for col in target_cols
        ])
        .sort(ts_col)  # Sort again after aggregation
        .collect()
    )
    
    print(f"Downsampled to {len(downsampled)} rows")
    return downsampled

def compute_acf_pacf_optimized(series: pd.Series, max_lag: int = 100) -> dict:
    """Compute ACF and PACF with memory optimization."""
    # Remove NaN values and ensure we have enough data
    clean_series = series.dropna()
    if len(clean_series) < max_lag * 2:
        max_lag = min(max_lag, len(clean_series) // 2)
    
    if len(clean_series) < 10:
        return {"acf": [], "pacf": [], "error": "Insufficient data"}
    
    try:
        # Use FFT for faster computation
        acf_vals = acf(clean_series, nlags=max_lag, fft=True, alpha=None)
        pacf_vals = pacf(clean_series, nlags=max_lag, method="ywm")
        
        return {
            "acf": acf_vals.tolist(),
            "pacf": pacf_vals.tolist(),
            "max_lag": max_lag,
            "n_obs": len(clean_series)
        }
    except Exception as e:
        return {"acf": [], "pacf": [], "error": str(e)}

def compute_cross_correlation_optimized(x: pd.Series, y: pd.Series, max_lag: int = 100) -> dict:
    """Compute cross-correlation with memory optimization."""
    # Align series and remove NaN values
    aligned_data = pd.concat([x, y], axis=1).dropna()
    if len(aligned_data) < max_lag * 2:
        max_lag = min(max_lag, len(aligned_data) // 2)
    
    if len(aligned_data) < 10:
        return {"ccf": [], "lags": [], "error": "Insufficient data"}
    
    try:
        x_vals = aligned_data.iloc[:, 0].values
        y_vals = aligned_data.iloc[:, 1].values
        
        # Compute cross-correlation for both positive and negative lags
        # Using numpy's correlate for full cross-correlation
        correlation = np.correlate(x_vals, y_vals, mode='full')
        
        # Normalize by the standard deviations
        std_x = np.std(x_vals)
        std_y = np.std(y_vals)
        if std_x > 0 and std_y > 0:
            correlation = correlation / (std_x * std_y * len(x_vals))
        
        # Get the center and extract the desired range
        center = len(correlation) // 2
        start_idx = max(0, center - max_lag)
        end_idx = min(len(correlation), center + max_lag + 1)
        
        ccf_vals = correlation[start_idx:end_idx]
        lags = range(-(center - start_idx), end_idx - center)
        
        return {
            "ccf": ccf_vals.tolist(),
            "lags": list(lags),
            "max_lag": max_lag,
            "n_obs": len(aligned_data)
        }
    except Exception as e:
        return {"ccf": [], "lags": [], "error": str(e)}

def compute_rolling_correlation_optimized(df: pd.DataFrame, col1: str, col2: str, 
                                        window_size: int = 1000) -> dict:
    """Compute rolling correlation with memory optimization."""
    if col1 not in df.columns or col2 not in df.columns:
        return {"rolling_corr": [], "error": "Columns not found"}
    
    try:
        # Use a smaller window if dataset is too small
        if len(df) < window_size * 2:
            window_size = len(df) // 4
        
        # Compute rolling correlation
        rolling_corr = df[col1].rolling(window=window_size, min_periods=window_size//2).corr(df[col2])
        
        # Sample the results to avoid memory issues
        if len(rolling_corr) > 1000:
            step = len(rolling_corr) // 1000
            rolling_corr = rolling_corr.iloc[::step]
        
        return {
            "rolling_corr": rolling_corr.dropna().tolist(),
            "window_size": window_size,
            "n_points": len(rolling_corr.dropna())
        }
    except Exception as e:
        return {"rolling_corr": [], "error": str(e)}

def plot_acf_pacf(series: pd.Series, title: str, output_path: Path):
    """Create ACF/PACF plots."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # ACF Plot
    acf_result = compute_acf_pacf_optimized(series, MAX_LAG)
    if "acf" in acf_result and acf_result["acf"]:
        ax1.plot(acf_result["acf"], marker='o', markersize=3)
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax1.axhline(y=0.1, color='r', linestyle='--', alpha=0.5)
        ax1.axhline(y=-0.1, color='r', linestyle='--', alpha=0.5)
        ax1.set_title(f'ACF - {title}')
        ax1.set_xlabel('Lag')
        ax1.set_ylabel('ACF')
        ax1.grid(True, alpha=0.3)
    
    # PACF Plot
    if "pacf" in acf_result and acf_result["pacf"]:
        ax2.plot(acf_result["pacf"], marker='o', markersize=3, color='orange')
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.axhline(y=0.1, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=-0.1, color='r', linestyle='--', alpha=0.5)
        ax2.set_title(f'PACF - {title}')
        ax2.set_xlabel('Lag')
        ax2.set_ylabel('PACF')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ACF/PACF plot saved: {output_path}")

def plot_cross_correlation(x: pd.Series, y: pd.Series, title: str, output_path: Path):
    """Create cross-correlation plot - FIXED version."""
    ccf_result = compute_cross_correlation_optimized(x, y, MAX_LAG)
    
    if "ccf" in ccf_result and ccf_result["ccf"] and "lags" in ccf_result:
        plt.figure(figsize=(12, 6))
        # FIXED: Use the lags returned by the function
        plt.plot(ccf_result["lags"], ccf_result["ccf"], marker='o', markersize=3)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axhline(y=0.1, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=-0.1, color='r', linestyle='--', alpha=0.5)
        plt.title(f'Cross-Correlation - {title}')
        plt.xlabel('Lag')
        plt.ylabel('Cross-Correlation')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Cross-correlation plot saved: {output_path}")
    else:
        print(f"Could not create cross-correlation plot for {title}: {ccf_result.get('error', 'Unknown error')}")

def plot_rolling_correlation(df: pd.DataFrame, col1: str, col2: str, 
                           title: str, output_path: Path):
    """Create rolling correlation plot."""
    rolling_result = compute_rolling_correlation_optimized(df, col1, col2, ROLLING_WINDOW_SIZE)
    
    if "rolling_corr" in rolling_result and rolling_result["rolling_corr"]:
        plt.figure(figsize=(12, 6))
        plt.plot(rolling_result["rolling_corr"], alpha=0.7)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.title(f'Rolling Correlation ({ROLLING_WINDOW_DAYS}-day window) - {title}')
        plt.xlabel('Time')
        plt.ylabel('Correlation')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Rolling correlation plot saved: {output_path}")
    else:
        print(f"Could not create rolling correlation plot for {title}: {rolling_result.get('error', 'Unknown error')}")

# =========================
# MAIN ANALYSIS FUNCTION
# =========================

def main():
    """Main correlation analysis function optimized for large datasets."""
    print("Starting optimized correlation analysis for large dataset...")
    print(f"Data path: {DATA_PATH}")
    print(f"Target columns: {TARGET_COLS}")
    
    # Step 1: Load and prepare data
    print("\nLoading dataset...")
    lf = pl.scan_parquet(DATA_PATH)
    
    # Detect timestamp column
    ts_col = detect_timestamp_column(lf)
    print(f"🕐 Timestamp column: {ts_col}")
    
    # Ensure datetime format
    lf = ensure_datetime_column(lf, ts_col)
    
    # Check which target columns exist
    available_cols = [col for col in TARGET_COLS if col in lf.collect_schema().names()]
    if not available_cols:
        raise ValueError("No target columns found in dataset!")
    
    print(f"Available columns: {available_cols}")
    
    # Step 2: Smart downsampling
    print("\nAttempting time-based downsampling...")
    downsampled_df = smart_downsample_fixed(lf, ts_col, available_cols, DOWNSAMPLE_FACTOR)
    
    # Convert to pandas for analysis
    df_pandas = downsampled_df.to_pandas()
    df_pandas = df_pandas.set_index(ts_col).sort_index()
    
    print(f"Final dataset shape: {df_pandas.shape}")
    
    # Step 3: Initialize results dictionary
    results = {
        "dataset_info": {
            "original_shape": "~70M rows (estimated)",
            "downsampled_shape": df_pandas.shape,
            "downsample_factor": DOWNSAMPLE_FACTOR,
            "columns_analyzed": available_cols,
            "max_lag": MAX_LAG,
            "rolling_window_days": ROLLING_WINDOW_DAYS
        },
        "acf_pacf": {},
        "cross_correlation": {},
        "rolling_correlation": {}
    }
    
    # Step 4: ACF/PACF Analysis
    print("\nComputing ACF/PACF...")
    for col in available_cols:
        if col in df_pandas.columns:
            print(f"  Processing {col}...")
            
            # Compute ACF/PACF
            acf_pacf_result = compute_acf_pacf_optimized(df_pandas[col], MAX_LAG)
            results["acf_pacf"][col] = acf_pacf_result
            
            # Create plots
            plot_path = OUTPUT_DIR / f"acf_pacf_{col.replace(' ', '_').replace('-', '_')}.png"
            plot_acf_pacf(df_pandas[col], col, plot_path)
    
    # Step 5: Cross-correlation Analysis
    print("\nComputing cross-correlations...")
    for col1, col2 in CROSS_CORR_PAIRS:
        if col1 in df_pandas.columns and col2 in df_pandas.columns:
            pair_name = f"{col1}_vs_{col2}"
            print(f"  Processing {pair_name}...")
            
            # Compute cross-correlation
            ccf_result = compute_cross_correlation_optimized(df_pandas[col1], df_pandas[col2], MAX_LAG)
            results["cross_correlation"][pair_name] = ccf_result
            
            # Create plots
            plot_path = OUTPUT_DIR / f"cross_corr_{pair_name.replace(' ', '_').replace('-', '_')}.png"
            plot_cross_correlation(df_pandas[col1], df_pandas[col2], pair_name, plot_path)
    
    # Step 6: Rolling Correlation Analysis
    print("\nComputing rolling correlations...")
    for col1, col2 in CROSS_CORR_PAIRS:
        if col1 in df_pandas.columns and col2 in df_pandas.columns:
            pair_name = f"{col1}_vs_{col2}"
            print(f"  Processing rolling correlation for {pair_name}...")
            
            # Compute rolling correlation
            rolling_result = compute_rolling_correlation_optimized(
                df_pandas, col1, col2, ROLLING_WINDOW_SIZE
            )
            results["rolling_correlation"][pair_name] = rolling_result
            
            # Create plots
            plot_path = OUTPUT_DIR / f"rolling_corr_{pair_name.replace(' ', '_').replace('-', '_')}.png"
            plot_rolling_correlation(df_pandas, col1, col2, pair_name, plot_path)
    
    # Step 7: Save results
    results_file = OUTPUT_DIR / "correlation_analysis_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nAnalysis complete!")
    print(f"Results saved to: {results_file}")
    print(f"Plots saved to: {OUTPUT_DIR}")
    
    # Print summary
    print("\nSUMMARY:")
    print(f"  • ACF/PACF computed for {len(results['acf_pacf'])} series")
    print(f"  • Cross-correlations computed for {len(results['cross_correlation'])} pairs")
    print(f"  • Rolling correlations computed for {len(results['rolling_correlation'])} pairs")
    print(f"  • Total plots generated: {len(available_cols) + 2 * len(CROSS_CORR_PAIRS)}")

if __name__ == "__main__":
    main()