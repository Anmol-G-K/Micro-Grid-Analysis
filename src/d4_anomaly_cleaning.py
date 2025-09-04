"""
Day 4 - Anomaly Detection & Data Cleaning (IMPROVED VERSION)

Goal: Ensure clean input for forecasting/optimization by identifying and handling anomalies.

IMPROVEMENTS:
- Fixed plotting issues (blank plots)
- More detailed JSON output with per-method statistics
- Better anomaly detection thresholds
- Enhanced visualization with proper data handling
- Detailed outlier information per method

Features:
- Multiple anomaly detection methods with detailed results
- Robust visualization that handles edge cases
- Comprehensive logging with per-method statistics
- Memory-efficient processing for large datasets
- Statistical validation of cleaning decisions

Dependencies: polars, pandas, numpy, matplotlib, seaborn, scipy
"""

import polars as pl
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =========================
# CONFIGURATION
# =========================
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "cleaned_dataset.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "plots" / "anomaly_cleaning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Anomaly detection configuration (IMPROVED THRESHOLDS)
Z_SCORE_THRESHOLD = 3.0
IQR_FACTOR = 1.5
MAD_THRESHOLD = 2.5  # Reduced from 3.0 for better sensitivity
SAMPLE_SIZE = 100000  # For visualization (memory efficiency)
CLEANING_LOG_FILE = OUTPUT_DIR / "data_cleaning_log_detailed.json"

# Target columns for analysis
TARGET_COLS = [
    "GE_Active_Power",        # Load
    "PVPCS_Active_Power",     # PV
    "Battery_Active_Power"    # Battery
]

# Cleaning strategies per column (can be modified based on domain knowledge)
CLEANING_STRATEGIES = {
    "GE_Active_Power": "interpolate",      # Load - likely genuine peaks, interpolate
    "PVPCS_Active_Power": "keep",          # PV - solar peaks are real
    "Battery_Active_Power": "remove"       # Battery - sensor errors more likely
}

# =========================
# UTILITY FUNCTIONS
# =========================

def detect_timestamp_column(df: pl.LazyFrame) -> str:
    """Detect timestamp column in the dataset."""
    schema = df.collect_schema()
    candidates = [c for c in schema.names() if "time" in c.lower() or "date" in c.lower()]
    if candidates:
        return candidates[0]
    
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

def sample_data_for_analysis(lf: pl.LazyFrame, ts_col: str, target_cols: list, 
                           sample_size: int = 100000) -> pl.DataFrame:
    """Sample data for anomaly analysis while maintaining temporal structure."""
    print(f"🔄 Sampling {sample_size:,} rows for analysis...")
    
    # Get total count
    total_rows = lf.select(pl.count()).collect().item()
    
    if total_rows <= sample_size:
        # Use all data if smaller than sample size
        sample_df = lf.select([ts_col] + target_cols).collect()
    else:
        # Stratified sampling to maintain temporal distribution
        step = total_rows // sample_size
        sample_df = (
            lf
            .select([ts_col] + target_cols)
            .with_row_index("row_idx")
            .filter(pl.col("row_idx") % step == 0)
            .drop("row_idx")
            .collect()
        )
    
    print(f"✅ Sampled {len(sample_df):,} rows from {total_rows:,} total rows")
    return sample_df

# =========================
# ANOMALY DETECTION METHODS (IMPROVED)
# =========================

def detect_anomalies_zscore(series: pd.Series, threshold: float = 3.0) -> dict:
    """Detect anomalies using Z-score method with detailed results."""
    # Remove NaN values for calculation
    clean_series = series.dropna()
    if len(clean_series) < 10:
        return {
            "anomaly_mask": pd.Series(False, index=series.index),
            "anomaly_count": 0,
            "anomaly_indices": [],
            "anomaly_values": [],
            "method_stats": {"mean": 0, "std": 0, "threshold": threshold}
        }
    
    # Calculate Z-scores
    z_scores = np.abs(stats.zscore(clean_series))
    
    # Create boolean mask for anomalies
    anomaly_mask = pd.Series(False, index=series.index)
    anomaly_mask[clean_series.index] = z_scores > threshold
    
    # Get anomaly details
    anomaly_indices = clean_series.index[z_scores > threshold].tolist()
    anomaly_values = clean_series[z_scores > threshold].tolist()
    
    return {
        "anomaly_mask": anomaly_mask,
        "anomaly_count": int(anomaly_mask.sum()),
        "anomaly_indices": anomaly_indices,
        "anomaly_values": anomaly_values,
        "method_stats": {
            "mean": float(clean_series.mean()),
            "std": float(clean_series.std()),
            "threshold": threshold,
            "z_scores": z_scores[z_scores > threshold].tolist()
        }
    }

def detect_anomalies_iqr(series: pd.Series, factor: float = 1.5) -> dict:
    """Detect anomalies using IQR method with detailed results."""
    clean_series = series.dropna()
    if len(clean_series) < 10:
        return {
            "anomaly_mask": pd.Series(False, index=series.index),
            "anomaly_count": 0,
            "anomaly_indices": [],
            "anomaly_values": [],
            "method_stats": {"q1": 0, "q3": 0, "iqr": 0, "factor": factor}
        }
    
    Q1 = clean_series.quantile(0.25)
    Q3 = clean_series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    # Create boolean mask for anomalies
    anomaly_mask = pd.Series(False, index=series.index)
    anomaly_condition = (clean_series < lower_bound) | (clean_series > upper_bound)
    anomaly_mask[clean_series.index] = anomaly_condition
    
    # Get anomaly details
    anomaly_indices = clean_series.index[anomaly_condition].tolist()
    anomaly_values = clean_series[anomaly_condition].tolist()
    
    return {
        "anomaly_mask": anomaly_mask,
        "anomaly_count": int(anomaly_mask.sum()),
        "anomaly_indices": anomaly_indices,
        "anomaly_values": anomaly_values,
        "method_stats": {
            "q1": float(Q1),
            "q3": float(Q3),
            "iqr": float(IQR),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "factor": factor
        }
    }

def detect_anomalies_mad(series: pd.Series, threshold: float = 2.5) -> dict:
    """Detect anomalies using Median Absolute Deviation (MAD) method with detailed results."""
    clean_series = series.dropna()
    if len(clean_series) < 10:
        return {
            "anomaly_mask": pd.Series(False, index=series.index),
            "anomaly_count": 0,
            "anomaly_indices": [],
            "anomaly_values": [],
            "method_stats": {"median": 0, "mad": 0, "threshold": threshold}
        }
    
    # Calculate MAD
    median = clean_series.median()
    mad = np.median(np.abs(clean_series - median))
    
    # Modified Z-score using MAD
    modified_z_scores = 0.6745 * (clean_series - median) / mad
    
    # Create boolean mask for anomalies
    anomaly_mask = pd.Series(False, index=series.index)
    anomaly_condition = np.abs(modified_z_scores) > threshold
    anomaly_mask[clean_series.index] = anomaly_condition
    
    # Get anomaly details
    anomaly_indices = clean_series.index[anomaly_condition].tolist()
    anomaly_values = clean_series[anomaly_condition].tolist()
    
    return {
        "anomaly_mask": anomaly_mask,
        "anomaly_count": int(anomaly_mask.sum()),
        "anomaly_indices": anomaly_indices,
        "anomaly_values": anomaly_values,
        "method_stats": {
            "median": float(median),
            "mad": float(mad),
            "threshold": threshold,
            "modified_z_scores": modified_z_scores[anomaly_condition].tolist()
        }
    }

def detect_anomalies_combined(series: pd.Series, methods: list = None) -> dict:
    """Detect anomalies using multiple methods and combine results with detailed information."""
    if methods is None:
        methods = ['zscore', 'iqr', 'mad']
    
    results = {}
    
    for method in methods:
        try:
            if method == 'zscore':
                results[method] = detect_anomalies_zscore(series, Z_SCORE_THRESHOLD)
            elif method == 'iqr':
                results[method] = detect_anomalies_iqr(series, IQR_FACTOR)
            elif method == 'mad':
                results[method] = detect_anomalies_mad(series, MAD_THRESHOLD)
            else:
                print(f"⚠️ Unknown method: {method}")
                continue
                
            print(f"    {method}: {results[method]['anomaly_count']} anomalies detected")
            
        except Exception as e:
            print(f"    ⚠️ Error with {method}: {e}")
            results[method] = {
                "anomaly_mask": pd.Series(False, index=series.index),
                "anomaly_count": 0,
                "anomaly_indices": [],
                "anomaly_values": [],
                "method_stats": {}
            }
    
    # Combine results (union of all methods)
    if results:
        combined_mask = pd.Series(False, index=series.index)
        combined_indices = set()
        combined_values = []
        
        for method, result in results.items():
            combined_mask = combined_mask | result["anomaly_mask"]
            combined_indices.update(result["anomaly_indices"])
            combined_values.extend(result["anomaly_values"])
        
        results['combined'] = {
            "anomaly_mask": combined_mask,
            "anomaly_count": int(combined_mask.sum()),
            "anomaly_indices": list(combined_indices),
            "anomaly_values": combined_values,
            "method_stats": {
                "methods_used": list(results.keys()),
                "total_unique_anomalies": len(combined_indices)
            }
        }
        print(f"    �� Combined: {results['combined']['anomaly_count']} anomalies detected")
    
    return results

# =========================
# DATA CLEANING METHODS
# =========================

def interpolate_anomalies(series: pd.Series, anomaly_mask: pd.Series, 
                         method: str = "linear") -> pd.Series:
    """Interpolate anomalous values using specified method."""
    cleaned_series = series.copy()
    
    if anomaly_mask.sum() == 0:
        return cleaned_series
    
    # Get valid (non-anomalous) data points
    valid_mask = ~anomaly_mask
    valid_indices = series.index[valid_mask]
    valid_values = series[valid_mask]
    
    if len(valid_values) < 2:
        print("⚠️ Not enough valid data points for interpolation")
        return cleaned_series
    
    # FIXED: Convert datetime indices to numeric for interpolation
    if hasattr(valid_indices, 'to_pydatetime'):
        # Convert datetime index to numeric (timestamp)
        valid_indices_numeric = pd.to_numeric(valid_indices)
        anomalous_indices = series.index[anomaly_mask]
        anomalous_indices_numeric = pd.to_numeric(anomalous_indices)
    else:
        # Already numeric indices
        valid_indices_numeric = valid_indices
        anomalous_indices = series.index[anomaly_mask]
        anomalous_indices_numeric = anomalous_indices
    
    # Create interpolation function
    if method == "linear":
        try:
            interp_func = interp1d(valid_indices_numeric, valid_values, 
                                  kind='linear', bounds_error=False, 
                                  fill_value='extrapolate')
            # Interpolate anomalous values
            interpolated_values = interp_func(anomalous_indices_numeric)
            cleaned_series.loc[anomalous_indices] = interpolated_values
        except Exception as e:
            print(f"⚠️ Linear interpolation failed: {e}, using forward fill")
            cleaned_series = series.fillna(method='ffill')
    elif method == "seasonal":
        # Simple seasonal interpolation (hourly pattern)
        cleaned_series = interpolate_seasonal(series, anomaly_mask)
    else:
        # Forward fill
        cleaned_series = series.fillna(method='ffill')
    
    return cleaned_series

def interpolate_seasonal(series: pd.Series, anomaly_mask: pd.Series) -> pd.Series:
    """Interpolate using seasonal patterns (hourly averages)."""
    cleaned_series = series.copy()
    
    if anomaly_mask.sum() == 0:
        return cleaned_series
    
    # Get valid data
    valid_mask = ~anomaly_mask
    valid_data = series[valid_mask]
    
    if len(valid_data) < 24:  # Need at least 24 hours of data
        return series.fillna(method='ffill')
    
    # Calculate hourly averages from valid data
    if hasattr(series.index, 'hour'):
        hourly_avg = valid_data.groupby(valid_data.index.hour).mean()
    else:
        # If no datetime index, use simple forward fill
        return series.fillna(method='ffill')
    
    # Replace anomalous values with seasonal averages
    anomalous_indices = series.index[anomaly_mask]
    for idx in anomalous_indices:
        if hasattr(idx, 'hour'):
            cleaned_series.loc[idx] = hourly_avg.get(idx.hour, valid_data.mean())
        else:
            cleaned_series.loc[idx] = valid_data.mean()
    
    return cleaned_series

# =========================
# VISUALIZATION FUNCTIONS (IMPROVED)
# =========================

def plot_anomalies_timeseries(df: pd.DataFrame, ts_col: str, col: str, 
                            anomaly_results: dict, output_path: Path):
    """Plot time series with anomalies highlighted using multiple methods."""
    # Filter out methods with no anomalies for cleaner plots
    active_methods = {k: v for k, v in anomaly_results.items() 
                     if v['anomaly_count'] > 0 and k != 'combined'}
    
    if not active_methods:
        # Create a simple plot showing no anomalies
        fig, ax = plt.subplots(1, 1, figsize=(15, 6))
        ax.plot(df[ts_col], df[col], alpha=0.7, linewidth=0.8, label='Data', color='blue')
        ax.set_title(f'Time Series - {col} (No Anomalies Detected)')
        ax.set_xlabel('Time')
        ax.set_ylabel(col)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"�� Anomaly plot saved: {output_path}")
        return
    
    n_methods = len(active_methods)
    fig, axes = plt.subplots(n_methods + 1, 1, figsize=(15, 4 * (n_methods + 1)))
    
    if n_methods == 0:
        axes = [axes]
    
    # Main time series plot
    axes[0].plot(df[ts_col], df[col], alpha=0.7, linewidth=0.8, label='Normal', color='blue')
    axes[0].set_title(f'Time Series - {col}')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel(col)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot anomalies for each method
    colors = ['red', 'orange', 'green', 'purple', 'brown']
    for i, (method, result) in enumerate(active_methods.items()):
        ax = axes[i + 1] if i + 1 < len(axes) else axes[-1]
        ax.plot(df[ts_col], df[col], alpha=0.7, linewidth=0.8, label='Normal', color='blue')
        
        # Highlight anomalies
        if result['anomaly_count'] > 0:
            # Get anomaly data using the mask
            anomaly_mask = result['anomaly_mask']
            anomaly_data = df[anomaly_mask.values]
            
            if len(anomaly_data) > 0:
                ax.scatter(anomaly_data[ts_col], anomaly_data[col], 
                          color=colors[i % len(colors)], s=20, alpha=0.8, 
                          label=f'{method.title()} Anomalies ({result["anomaly_count"]})')
        
        ax.set_title(f'Anomalies - {method.title()} Method ({result["anomaly_count"]} anomalies)')
        ax.set_xlabel('Time')
        ax.set_ylabel(col)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"�� Anomaly plot saved: {output_path}")

def plot_anomaly_distribution(df: pd.DataFrame, col: str, anomaly_results: dict, 
                            output_path: Path):
    """Plot distribution of anomalies for each method."""
    # Filter out methods with no anomalies
    active_methods = {k: v for k, v in anomaly_results.items() 
                     if v['anomaly_count'] > 0 and k != 'combined'}
    
    if not active_methods:
        # Create a simple distribution plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.hist(df[col].dropna(), bins=50, alpha=0.7, density=True, 
                label='All Data', color='blue')
        ax.set_title(f'Distribution - {col} (No Anomalies Detected)')
        ax.set_xlabel(col)
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Distribution plot saved: {output_path}")
        return
    
    n_methods = len(active_methods)
    fig, axes = plt.subplots(2, (n_methods + 1) // 2, figsize=(15, 8))
    axes = axes.flatten() if n_methods > 1 else [axes]
    
    # Overall distribution
    axes[0].hist(df[col].dropna(), bins=50, alpha=0.7, density=True, 
                label='All Data', color='blue')
    axes[0].set_title(f'Overall Distribution - {col}')
    axes[0].set_xlabel(col)
    axes[0].set_ylabel('Density')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Distribution for each method
    colors = ['red', 'orange', 'green', 'purple', 'brown']
    for i, (method, result) in enumerate(active_methods.items()):
        ax = axes[i + 1] if i + 1 < len(axes) else axes[-1]
        
        # Normal data
        normal_mask = ~result['anomaly_mask'].values
        normal_data = df[col][normal_mask].dropna()
        ax.hist(normal_data, bins=30, alpha=0.7, density=True, 
               label='Normal', color='blue')
        
        # Anomalous data
        if result['anomaly_count'] > 0:
            anomaly_data = df[col][result['anomaly_mask'].values].dropna()
            ax.hist(anomaly_data, bins=20, alpha=0.7, density=True, 
                   color=colors[i % len(colors)], label=f'Anomalies ({result["anomaly_count"]})')
        
        ax.set_title(f'{method.title()} Method ({result["anomaly_count"]} anomalies)')
        ax.set_xlabel(col)
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Distribution plot saved: {output_path}")

def plot_cleaning_comparison(df: pd.DataFrame, ts_col: str, col: str, 
                           original: pd.Series, cleaned: pd.Series, 
                           anomaly_mask: pd.Series, output_path: Path):
    """Plot comparison of original vs cleaned data."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Time series comparison
    ax1.plot(df[ts_col], original, alpha=0.7, linewidth=0.8, label='Original', color='blue')
    ax1.plot(df[ts_col], cleaned, alpha=0.7, linewidth=0.8, label='Cleaned', color='green')
    
    # Highlight cleaned points
    if anomaly_mask.sum() > 0:
        # Get the timestamps where anomalies occur
        anomaly_timestamps = df[ts_col][anomaly_mask.values]
        cleaned_values = cleaned[anomaly_mask.values]
        
        ax1.scatter(anomaly_timestamps, cleaned_values, 
                   color='red', s=20, alpha=0.8, label=f'Cleaned Points ({anomaly_mask.sum()})')
    
    ax1.set_title(f'Data Cleaning Comparison - {col}')
    ax1.set_xlabel('Time')
    ax1.set_ylabel(col)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Difference plot
    difference = cleaned - original
    ax2.plot(df[ts_col], difference, alpha=0.7, linewidth=0.8, color='orange')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title(f'Cleaning Differences - {col}')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Difference (Cleaned - Original)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Cleaning comparison plot saved: {output_path}")

# =========================
# REPORTING FUNCTIONS (ENHANCED)
# =========================

def generate_detailed_cleaning_report(cleaning_results: dict, dataset_info: dict) -> dict:
    """Generate comprehensive cleaning report with detailed per-method statistics."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "dataset_info": dataset_info,
        "anomaly_detection_config": {
            "z_score_threshold": Z_SCORE_THRESHOLD,
            "iqr_factor": IQR_FACTOR,
            "mad_threshold": MAD_THRESHOLD,
            "sample_size": SAMPLE_SIZE
        },
        "cleaning_results": {},
        "summary": {
            "total_anomalies_detected": 0,
            "total_anomalies_cleaned": 0,
            "cleaning_strategies_used": {},
            "methods_used": [],
            "per_method_summary": {}
        }
    }
    
    # Initialize per-method summary
    for method in ['zscore', 'iqr', 'mad']:
        report["summary"]["per_method_summary"][method] = {
            "total_anomalies": 0,
            "columns_affected": [],
            "avg_anomalies_per_column": 0
        }
    
    for col, results in cleaning_results.items():
        # Get the primary anomaly mask (combined or first available)
        primary_result = results.get('combined', 
                                   next(iter(results['anomaly_results'].values())) 
                                   if results['anomaly_results'] else None)
        
        if primary_result is None:
            continue
        
        # Detailed per-method results
        method_details = {}
        for method, result in results['anomaly_results'].items():
            if method == 'combined':
                continue
                
            method_details[method] = {
                "anomaly_count": result['anomaly_count'],
                "anomaly_percentage": (result['anomaly_count'] / len(results['original'])) * 100,
                "anomaly_indices": result['anomaly_indices'][:10],  # First 10 for brevity
                "anomaly_values": result['anomaly_values'][:10],    # First 10 for brevity
                "method_statistics": result['method_stats']
            }
            
            # Update per-method summary
            if method in report["summary"]["per_method_summary"]:
                report["summary"]["per_method_summary"][method]["total_anomalies"] += result['anomaly_count']
                if result['anomaly_count'] > 0:
                    report["summary"]["per_method_summary"][method]["columns_affected"].append(col)
        
        report["cleaning_results"][col] = {
            "strategy": results["strategy"],
            "anomaly_detection_methods": list(results['anomaly_results'].keys()),
            "method_details": method_details,
            "combined_results": {
                "anomalies_detected": primary_result['anomaly_count'],
                "anomalies_cleaned": primary_result['anomaly_count'] if results["strategy"] != "keep" else 0,
                "anomaly_percentage": (primary_result['anomaly_count'] / len(results['original'])) * 100,
                "unique_anomaly_indices": len(primary_result['anomaly_indices']),
                "anomaly_values_summary": {
                    "min": float(min(primary_result['anomaly_values'])) if primary_result['anomaly_values'] else 0,
                    "max": float(max(primary_result['anomaly_values'])) if primary_result['anomaly_values'] else 0,
                    "mean": float(np.mean(primary_result['anomaly_values'])) if primary_result['anomaly_values'] else 0
                }
            },
            "statistics": {
                "original_mean": float(results["original"].mean()),
                "original_std": float(results["original"].std()),
                "original_min": float(results["original"].min()),
                "original_max": float(results["original"].max()),
                "cleaned_mean": float(results["cleaned"].mean()),
                "cleaned_std": float(results["cleaned"].std()),
                "cleaned_min": float(results["cleaned"].min()),
                "cleaned_max": float(results["cleaned"].max()),
                "mean_change": float(results["cleaned"].mean() - results["original"].mean()),
                "std_change": float(results["cleaned"].std() - results["original"].std())
            }
        }
        
        report["summary"]["total_anomalies_detected"] += primary_result['anomaly_count']
        if results["strategy"] != "keep":
            report["summary"]["total_anomalies_cleaned"] += primary_result['anomaly_count']
        
        report["summary"]["cleaning_strategies_used"][col] = results["strategy"]
        
        # Collect unique methods used
        for method in results['anomaly_results'].keys():
            if method not in report["summary"]["methods_used"]:
                report["summary"]["methods_used"].append(method)
    
    # Calculate averages for per-method summary
    for method in report["summary"]["per_method_summary"]:
        method_summary = report["summary"]["per_method_summary"][method]
        if method_summary["columns_affected"]:
            method_summary["avg_anomalies_per_column"] = (
                method_summary["total_anomalies"] / len(method_summary["columns_affected"])
            )
    
    return report

# =========================
# MAIN ANALYSIS FUNCTION
# =========================

def main():
    """Main anomaly detection and cleaning function."""
    print("🔍 Starting IMPROVED anomaly detection and data cleaning...")
    print(f"📁 Data path: {DATA_PATH}")
    
    # Step 1: Load and prepare data
    print("\n Loading dataset...")
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
    
    print(f"✅ Available columns: {available_cols}")
    
    # Step 2: Sample data for analysis
    sample_df = sample_data_for_analysis(lf, ts_col, available_cols, SAMPLE_SIZE)
    
    # Convert to pandas for analysis
    df_pandas = sample_df.to_pandas()
    df_pandas = df_pandas.set_index(ts_col).sort_index()
    
    print(f"📈 Analysis dataset shape: {df_pandas.shape}")
    
    # Step 3: Anomaly Detection and Cleaning
    print("\n🔍 Detecting anomalies using multiple methods...")
    cleaning_results = {}
    
    for col in available_cols:
        if col not in df_pandas.columns:
            continue
            
        print(f"  📊 Processing {col}...")
        
        # Get cleaning strategy
        strategy = CLEANING_STRATEGIES.get(col, "interpolate")
        
        # Detect anomalies using multiple methods
        anomaly_results = detect_anomalies_combined(df_pandas[col])
        
        # Use combined results for cleaning
        primary_result = anomaly_results.get('combined', 
                                           next(iter(anomaly_results.values())) 
                                           if anomaly_results else None)
        
        if primary_result is None:
            continue
        
        primary_anomaly_mask = primary_result['anomaly_mask']
        
        print(f"    🚨 Total anomalies detected: {primary_result['anomaly_count']} ({primary_result['anomaly_count']/len(df_pandas)*100:.2f}%)")
        
        # Apply cleaning strategy
        original_series = df_pandas[col].copy()
        cleaned_series = original_series.copy()
        
        if strategy == "remove":
            # Remove anomalous values (set to NaN)
            cleaned_series[primary_anomaly_mask] = np.nan
        elif strategy == "interpolate":
            # Interpolate anomalous values
            cleaned_series = interpolate_anomalies(original_series, primary_anomaly_mask, method="linear")
        elif strategy == "keep":
            # Keep original values
            pass
        else:
            print(f"    ⚠️ Unknown strategy '{strategy}', keeping original values")
        
        # Store results
        cleaning_results[col] = {
            "strategy": strategy,
            "anomaly_results": anomaly_results,
            "primary_anomaly_mask": primary_anomaly_mask,
            "original": original_series,
            "cleaned": cleaned_series
        }
        
        # Create plots
        print(f"    📊 Creating plots...")
        
        # Anomaly visualization
        anomaly_plot_path = OUTPUT_DIR / f"anomalies_{col.replace(' ', '_').replace('-', '_')}.png"
        plot_anomalies_timeseries(df_pandas.reset_index(), ts_col, col, anomaly_results, anomaly_plot_path)
        
        # Distribution plots
        distribution_plot_path = OUTPUT_DIR / f"distribution_{col.replace(' ', '_').replace('-', '_')}.png"
        plot_anomaly_distribution(df_pandas.reset_index(), col, anomaly_results, distribution_plot_path)
        
        # Cleaning comparison
        if strategy != "keep":
            comparison_plot_path = OUTPUT_DIR / f"cleaning_comparison_{col.replace(' ', '_').replace('-', '_')}.png"
            plot_cleaning_comparison(df_pandas.reset_index(), ts_col, col, 
                                   original_series, cleaned_series, primary_anomaly_mask, comparison_plot_path)
    
    # Step 4: Generate comprehensive report
    print("\n📋 Generating detailed cleaning report...")
    dataset_info = {
        "total_rows": len(df_pandas),
        "columns_analyzed": available_cols,
        "sample_size": SAMPLE_SIZE
    }
    
    cleaning_report = generate_detailed_cleaning_report(cleaning_results, dataset_info)
    
    # Save detailed cleaning log
    with open(CLEANING_LOG_FILE, "w") as f:
        json.dump(cleaning_report, f, indent=2, default=str)
    
    print(f"📋 Detailed cleaning log saved: {CLEANING_LOG_FILE}")
    
    # Step 5: Summary
    print(f"\n✅ IMPROVED anomaly detection and cleaning complete!")
    print(f"📈 Plots saved to: {OUTPUT_DIR}")
    
    # Print detailed summary
    print("\n📋 DETAILED CLEANING SUMMARY:")
    print(f"  • Columns analyzed: {len(cleaning_results)}")
    print(f"  • Total anomalies detected: {cleaning_report['summary']['total_anomalies_detected']}")
    print(f"  • Total anomalies cleaned: {cleaning_report['summary']['total_anomalies_cleaned']}")
    print(f"  • Methods used: {', '.join(cleaning_report['summary']['methods_used'])}")
    
    print("\n�� Per-method summary:")
    for method, summary in cleaning_report['summary']['per_method_summary'].items():
        print(f"  • {method.upper()}: {summary['total_anomalies']} total anomalies, "
              f"{len(summary['columns_affected'])} columns affected, "
              f"avg {summary['avg_anomalies_per_column']:.1f} per column")
    
    print("\n Per-column results:")
    for col, results in cleaning_results.items():
        strategy = results["strategy"]
        primary_result = results['anomaly_results'].get('combined', {})
        anomalies = primary_result.get('anomaly_count', 0)
        methods = list(results["anomaly_results"].keys())
        print(f"  • {col}: {anomalies} anomalies, strategy='{strategy}', methods={methods}")
    
    print(f"\n📋 Detailed cleaning log: {CLEANING_LOG_FILE}")

if __name__ == "__main__":
    main()