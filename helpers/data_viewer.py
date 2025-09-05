from pathlib import Path
import polars as pl
import pandas as pd
import streamlit as st
import time

# Paths
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
PARQUET_FILE = OUTPUT_DIR / "cleaned_power_dataset.parquet"
FEATHER_FILE = OUTPUT_DIR / "cleaned_power_dataset.feather" 


CHUNK_SIZE = 10000

def load_dataset():
    start = time.time()

    if PARQUET_FILE.exists():
        st.write(f"Lazy loading from Parquet: `{PARQUET_FILE.name}`")
        lf = pl.scan_parquet(PARQUET_FILE)
        total_rows = lf.select(pl.count()).collect()[0, 0]
        st.success(f"Ready to load PARQUET with {total_rows:,} rows (lazy) in {time.time() - start:.2f} sec.")
        return "polars_lazy", lf, total_rows

    elif FEATHER_FILE.exists():
        st.write(f"Loading from Feather (Polars): `{FEATHER_FILE.name}`")
        df = pl.read_ipc(FEATHER_FILE)  # or pl.read_feather(FEATHER_FILE) depending on version
        st.success(f"Loaded FEATHER with shape: {df.shape} in {time.time() - start:.2f} sec.")
        return "polars_df", df, df.height


    else:
        st.warning("No .feather or .parquet file found in the specified path.")
        return None, None, 0

def main():
    st.set_page_config(layout="wide")
    st.title("📊 Polars Dataset Viewer (.feather / .parquet)")

    data_type, data_source, total_rows = load_dataset()

    if data_type is None:
        return

    start_row = st.number_input(
        "Start row",
        min_value=0,
        max_value=max(0, total_rows - CHUNK_SIZE),
        value=0,
        step=CHUNK_SIZE,
    )
    end_row = min(start_row + CHUNK_SIZE, total_rows)
    st.markdown(f"### Showing rows {start_row:,} to {end_row:,} of {total_rows:,}")

    if data_type == "polars_df":
        # eager Polars DataFrame, slice using row selection
        chunk_df = data_source[start_row:end_row]
        st.dataframe(chunk_df.to_pandas(), use_container_width=True, height=700)
    elif data_type == "polars_lazy":
        # lazy Polars LazyFrame, slice then collect
        chunk_df = data_source.slice(start_row, CHUNK_SIZE).collect().to_pandas()
        st.dataframe(chunk_df, use_container_width=True, height=700)

if __name__ == "__main__":
    main()
