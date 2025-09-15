import polars as pl
import numpy as np
import json
import logging
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# =====================
# CONFIGURATION
# =====================
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "combined.parquet"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "plots" / "forecasting"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = OUTPUT_DIR / "forecast_ready_v6.json"
RESOLUTION = "15m"
TARGET_COLS = ["load_kw", "pv_kw", "wind_kw"]
SEASONAL_PERIOD = 96  # For 15-minute intervals, typical period is 96 for daily seasonality

# =====================
# HELPERS
# =====================
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / np.clip(denominator, 1e-6, None)
    return np.mean(diff) * 100

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-6, None))) * 100

def evaluate_baseline(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "smape": float(smape(y_true, y_pred)),
        "mape": float(mape(y_true, y_pred))
    }

def add_calendar_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns([pl.col("timestamp").dt.hour().alias("hour_of_day"),
                          pl.col("timestamp").dt.weekday().alias("day_of_week"),
                          ((pl.col("timestamp").dt.month() - 1) // 3 + 1).alias("quarter"),
                          pl.col("timestamp").dt.month().alias("month"),
                          (pl.col("timestamp").dt.weekday() >= 5).cast(pl.Int8).alias("is_weekend")])
    df = df.with_columns([np.sin(2 * np.pi * pl.col("hour_of_day") / 24).alias("hour_sin"),
                          np.cos(2 * np.pi * pl.col("hour_of_day") / 24).alias("hour_cos"),
                          np.sin(2 * np.pi * pl.col("day_of_week") / 7).alias("dow_sin"),
                          np.cos(2 * np.pi * pl.col("day_of_week") / 7).alias("dow_cos")])
    return df

def add_lag_features(df: pl.DataFrame, cols, lags=[1, 2, 96]) -> pl.DataFrame:
    for col in cols:
        for lag in lags:
            df = df.with_columns(pl.col(col).shift(lag).alias(f"{col}_t-{lag}"))
    return df

def add_rolling_features(df: pl.DataFrame, cols, windows=[4, 8, 96]) -> pl.DataFrame:
    for col in cols:
        for w in windows:
            df = df.with_columns([pl.col(col).rolling_mean(window_size=w).alias(f"{col}roll_mean{w}"),
                                  pl.col(col).rolling_std(window_size=w).alias(f"{col}roll_std{w}")])
    return df

def clean_negative_values(df: pl.DataFrame, cols):
    for col in cols:
        df = df.with_columns(pl.when(pl.col(col) < 0).then(0).otherwise(pl.col(col)).alias(col))
    return df

def seasonal_decomposition(pdf: pd.DataFrame, col="load_kw", period=96) -> pd.DataFrame:
    result = seasonal_decompose(pdf[col], model="additive", period=period, extrapolate_trend="freq")
    pdf[f"{col}_trend"] = result.trend.fillna(method="bfill").fillna(method="ffill")
    pdf[f"{col}_seasonal"] = result.seasonal
    pdf[f"{col}_residual"] = result.resid.fillna(0)
    return pdf

# =====================
# FORECASTING MODELS
# =====================
def forecast_with_exponential_smoothing(train, test, seasonal_period=96):
    """Forecast using Holt-Winters Exponential Smoothing"""
    model = ExponentialSmoothing(train, trend="add", seasonal="mul", seasonal_periods=seasonal_period).fit()
    return model.forecast(len(test))

# =====================
# PLOTTING FUNCTIONALITIES
# =====================
def plot_interactive_timeseries(pdf, columns, output_dir: Path, title="Time Series Forecasting"):
    """Generate interactive time series plots with Plotly."""
    for col in columns:
        fig = go.Figure()

        # Plot the actual time series
        fig.add_trace(go.Scatter(x=pdf["timestamp"], y=pdf[col], mode='lines', name=f'{col} Actual', line=dict(color='blue')))
        
        # If forecasting model is available, plot the predicted values
        if f'{col}_forecast' in pdf.columns:
            fig.add_trace(go.Scatter(x=pdf["timestamp"], y=pdf[f'{col}_forecast'], mode='lines', name=f'{col} Forecast', line=dict(color='red', dash='dash')))

        # Customizations for the plot
        fig.update_layout(
            title=f"{col} Over Time ({title})",
            xaxis_title="Timestamp",
            yaxis_title=col,
            template="plotly_dark",
            showlegend=True,
            autosize=True
        )
        fig.write_html(output_dir / f"{col}_interactive_timeseries.html")
        logging.info(f"✅ Interactive plot saved to {output_dir / f'{col}_interactive_timeseries.html'}")

def plot_seasonal_decomposition(pdf, target_cols, output_dir: Path, period=96):
    """Plot the seasonal decomposition (trend, seasonal, residuals) for each target column."""
    for col in target_cols:
        if f"{col}_trend" in pdf.columns:
            fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

            # Trend
            axs[0].plot(pdf["timestamp"], pdf[f"{col}_trend"], label='Trend', color='orange')
            axs[0].set_title(f"{col} Trend")
            axs[0].set_ylabel('Trend')
            axs[0].grid(True)

            # Seasonal Component
            axs[1].plot(pdf["timestamp"], pdf[f"{col}_seasonal"], label='Seasonality', color='green')
            axs[1].set_title(f"{col} Seasonal Component")
            axs[1].set_ylabel('Seasonality')
            axs[1].grid(True)

            # Residuals
            axs[2].plot(pdf["timestamp"], pdf[f"{col}_residual"], label='Residuals', color='red')
            axs[2].set_title(f"{col} Residuals")
            axs[2].set_ylabel('Residuals')
            axs[2].grid(True)

            plt.xlabel('Timestamp')
            plt.tight_layout()
            plt.savefig(output_dir / f"{col}_seasonal_decomposition.png")
            logging.info(f"✅ Seasonal decomposition saved to {output_dir / f'{col}_seasonal_decomposition.png'}")
            plt.close(fig)

# =====================
# MAIN PIPELINE
# =====================
def main():
    logging.info("=== Time Series Forecasting Pipeline (v6) ===")

    # Load dataset
    df = pl.scan_parquet(DATA_PATH).collect()
    df = df.with_columns(
        pl.col("timestamp").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
    )

    # Downsample to 15-minute intervals
    df_resampled = df.sort("timestamp").group_by_dynamic("timestamp", every=RESOLUTION).agg([
        pl.mean("grid_load_demand").alias("load_kw"),
        pl.mean("solar_pv_output").alias("pv_kw"),
        pl.mean("wind_power_output").alias("wind_kw")
    ])

    # Feature Engineering
    df_resampled = add_calendar_features(df_resampled)
    df_resampled = add_lag_features(df_resampled, TARGET_COLS)
    df_resampled = add_rolling_features(df_resampled, TARGET_COLS)
    df_resampled = clean_negative_values(df_resampled, TARGET_COLS)

    # Convert to pandas for seasonal decomposition
    df_resampled_pdf = df_resampled.to_pandas()

    # Apply Seasonal Decomposition
    for col in TARGET_COLS:
        df_resampled_pdf = seasonal_decomposition(df_resampled_pdf, col)

    # Forecasting (Using Holt-Winters Exponential Smoothing)
    forecast_results = {}
    for col in TARGET_COLS:
        train, test = df_resampled_pdf[col][:-96], df_resampled_pdf[col][-96:]
        forecast = forecast_with_exponential_smoothing(train, test)
        df_resampled_pdf[f'{col}_forecast'] = np.concatenate([train, forecast])

        # Store the results
        forecast_results[col] = evaluate_baseline(test, forecast)

    # Save results to JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(forecast_results, f, indent=4)

    logging.info(f"✅ Forecast results saved to {OUTPUT_JSON}")

    # Plot the results
    plot_interactive_timeseries(df_resampled_pdf, TARGET_COLS, OUTPUT_DIR)
    plot_seasonal_decomposition(df_resampled_pdf, TARGET_COLS, OUTPUT_DIR)

if __name__ == "__main__":
    main()
