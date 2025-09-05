import polars as pl
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "cleaned_dataset.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "plots" / "anomaly_cleaning"
OUTPUT_DIR.mkdir(exist_ok=True)

# Improved anomaly detection parameters
Z_SCORE_THRESHOLD = 4.0  # Increased for power data
IQR_FACTOR = 3.0  # More conservative for power systems
MAD_THRESHOLD = 3.5  # Reduced sensitivity

# Domain-specific cleaning strategies
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
    "outlet_temperature": "remove_extremes_then_interpolate"
}

def load_and_preprocess_data():
    """Load and preprocess the power data"""
    df = pl.scan_parquet(DATA_PATH)
    
    # Rename to lowercase for consistency
    df = df.rename({col: col.lower() for col in df.columns})
    
    # Convert timestamp and ensure proper formatting
    df = df.with_columns(
        pl.col("timestamp").str.strptime(pl.Datetime, "%Y/%m/%d %H:%M:%S", strict=False)
    )
    
    return df.collect()
def detect_sentinel_values(series, sentinel_values=[-999999.0]):
    """Detect and flag sentinel/placeholder values"""
    if isinstance(series, pd.Series):
        mask = series.isin(sentinel_values)
    else:  # assume Polars
        mask = series.is_in(sentinel_values)
    
    return {
        "mask": mask,
        "count": mask.sum(),
        "indices": series[mask].index.tolist() if isinstance(series, pd.Series) else None,
        "values": series[mask].tolist()
    }


def improved_anomaly_detection(series, method="combined"):
    """Improved anomaly detection for power data"""
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
        anomaly_mask_clean = z_mask & iqr_mask  # strict: both must agree
    
    full_mask = pd.Series(False, index=series.index)
    full_mask[sentinel_mask] = True
    full_mask.loc[clean_series.index] = anomaly_mask_clean
    
    return {
        "mask": full_mask,
        "count": full_mask.sum(),
        "sentinel_info": sentinel_info,
        "statistical_anomalies": anomaly_mask_clean.sum()
    }


def apply_cleaning_strategy(series, anomaly_mask, strategy):
    """Apply the appropriate cleaning strategy"""
    if strategy == "keep":
        return series.copy()
    elif strategy == "remove":
        cleaned = series.copy()
        cleaned[anomaly_mask] = np.nan
        return cleaned
    elif strategy == "remove_extremes_then_interpolate":
        cleaned = series.copy()
        cleaned[anomaly_mask] = np.nan

        # Set datetime index
        if not isinstance(cleaned.index, pd.DatetimeIndex):
            cleaned.index = pd.to_datetime(cleaned.index, errors='coerce')

        # Interpolate with time if possible
        if cleaned.index.is_monotonic_increasing:
            cleaned = cleaned.interpolate(method='time')
        else:
            cleaned = cleaned.interpolate(method='linear')

        return cleaned

    elif strategy == "interpolate":
        cleaned = series.copy()
        cleaned[anomaly_mask] = np.nan   # <-- remove anomalies/sentinels first
        return cleaned.interpolate(method='linear')

def create_diagnostic_plots(df, column, anomaly_info, cleaned_series, output_path):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    orig_series = df[column].replace(-999999.0, np.nan)
    valid_mask = orig_series.notna()
    timestamps_filtered = df['timestamp'][valid_mask]
    orig_series_filtered = orig_series[valid_mask]
    
    anomaly_mask_filtered = anomaly_info['mask'][valid_mask]

    cleaned_series_filtered = cleaned_series.dropna()
    
    # Time series with anomalies
    ax1.plot(timestamps_filtered, orig_series_filtered, 'b-', alpha=0.7, label='Original')
    ax1.plot(timestamps_filtered[anomaly_mask_filtered], 
             orig_series_filtered[anomaly_mask_filtered], 
             'ro', alpha=0.7, label='Anomalies')
    ax1.set_title(f'Time Series with Anomalies - {column}')
    ax1.legend()
    
    # Distribution plot with clipping to avoid outlier domination
    clip_lower = np.percentile(orig_series_filtered, 1)
    clip_upper = np.percentile(orig_series_filtered, 99)
    ax2.hist(orig_series_filtered.clip(clip_lower, clip_upper), bins=50, alpha=0.7, density=True, label='Original (clipped)')
    ax2.hist(cleaned_series_filtered.clip(clip_lower, clip_upper), bins=50, alpha=0.7, density=True, label='Cleaned (clipped)')
    ax2.set_title(f'Distribution (1-99 percentile clipped) - {column}')
    ax2.legend()
    
    # QQ plot
    stats.probplot(orig_series_filtered, dist="norm", plot=ax3)
    ax3.set_title(f'Q-Q Plot - {column}')
    
    # Boxplot comparison
    plot_data = [orig_series_filtered, cleaned_series_filtered]
    ax4.boxplot(plot_data, labels=['Original', 'Cleaned'])
    ax4.set_title(f'Boxplot Comparison - {column}')
    ax4.set_ylabel('Power Value')
    ax4.set_xlabel('Dataset')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()



TARGET_COLUMNS = list(CLEANING_STRATEGIES.keys()) # all the columns

def main_cleaning_pipeline():
    print("Starting power data cleaning pipeline...")
    
    df = load_and_preprocess_data()
    cleaning_results = {}
    cleaned_values_dict = {}
    cleaning_results["dataset_size"] = {
        "num_rows": df.height,
        "num_columns": len(df.columns)
    }
    
    
    for column in TARGET_COLUMNS:
        if column not in df.columns:
            print(f"Skipping {column}, not found in dataset")
            continue
        
        print(f"Processing {column}...")
        
        anomaly_info = improved_anomaly_detection(df[column].to_pandas())  # convert Polars → Pandas
        print(f"  Detected {anomaly_info['count']} anomalies "
              f"({anomaly_info['sentinel_info']['count']} sentinel values)")
        
        # Apply cleaning
        strategy = CLEANING_STRATEGIES.get(column, "interpolate")
        cleaned_series = apply_cleaning_strategy(
            df[column].to_pandas(), anomaly_info["mask"], strategy
        )
        
        # Plots
        plot_path = OUTPUT_DIR / f"{column}_cleaning_diagnostics.png"
        create_diagnostic_plots(df.to_pandas(), column, anomaly_info, cleaned_series, plot_path)
        
        # Store stats
        cleaning_results[column] = {
            "original_stats": {
                "mean": float(df[column].mean()),
                "std": float(df[column].std()),
                "min": float(df[column].min()),
                "max": float(df[column].max())
            },
            "cleaned_stats": {
                "mean": float(cleaned_series.mean()),
                "std": float(cleaned_series.std()),
                "min": float(cleaned_series.min()),
                "max": float(cleaned_series.max())
            },
            "anomaly_info": {
                "total_anomalies": int(anomaly_info["count"]),
                "sentinel_values": int(anomaly_info["sentinel_info"]["count"]),
                "statistical_anomalies": int(anomaly_info["statistical_anomalies"])
            },
            "cleaning_strategy": strategy
        }
        cleaned_values_dict[column] = cleaned_series.head(25).tolist()
    
    # Save results
    results_path = OUTPUT_DIR / "cleaning_results.json"
    output_txt_path = OUTPUT_DIR / "first_25_cleaned_values.txt"
    with open(results_path, "w") as f:
        json.dump(cleaning_results, f, indent=2)
    
    print(f"Cleaning complete! Results saved to {results_path}")

    with open(output_txt_path, "w") as f:
        for col, values in cleaned_values_dict.items():
            f.write(f"Column: {col}\n")
            f.write("-" * (len(col) + 8) + "\n")
            for i, val in enumerate(values, 1):
                f.write(f"{i:2d}: {val}\n")
            f.write("\n")

    print(f"First 25 cleaned values saved to {output_txt_path}")

    return cleaning_results

if __name__ == "__main__":
    results = main_cleaning_pipeline()