# merges multiple csv files into one


import polars as pl
import glob

# Step 1: Find all CSV files in current folder
files = glob.glob("../data/raw/*.csv")
print("Found files:", files)

# Step 1b: Column consistency check
columns_per_file = {}
for file in files:
    cols = pl.read_csv(file, n_rows=0).columns  # read only headers
    columns_per_file[file] = cols

base_cols = list(columns_per_file.values())[0]
for f, cols in columns_per_file.items():
    if cols != base_cols:
        print(f"Column mismatch in {f}")
        print(f"Expected: {base_cols}")
        print(f"Found   : {cols}")

# Step 1c: Schema (data type) check
for file in files:
    schema = pl.read_csv(file, n_rows=0).schema
    print(f"{file} schema: {schema}")


# Step 2: Read each CSV and keep row counts
row_counts = {}
dfs = []
for file in files:
    df = pl.read_csv(file)
    dfs.append(df)
    row_counts[file] = df.height  # number of rows

# Step 3: Combine them vertically (stacked)
combined = pl.concat(dfs, how="vertical")

# Step 4: Save to one CSV
# combined.write_csv("../data/combined.csv")
combined.write_parquet("../data/combined.parquet")  # parquet is better for performance
combined.write_ipc("../data/combined.feather")  # feather is better for memory

# Step 5: Verification
expected_rows = sum(row_counts.values())
actual_rows = combined.height

print("\nRow counts per file:")
for f, r in row_counts.items():
    print(f"{f}: {r} rows")

print(f"\nExpected total rows: {expected_rows}")
print(f"Combined file rows: {actual_rows}")

if expected_rows == actual_rows:
    print("No row loss detected")
else:
    print("Row mismatch! Check your files.")

# Optional: check for duplicate rows
dupes = combined.filter(combined.is_duplicated())
if dupes.height > 0:
    print(f"\nDuplicate rows found: {dupes.height}")
    print(dupes.head())  # preview first few duplicates
else:
    print("No duplicate rows detected")



# Step 6: Missing data check
null_counts = combined.null_count()
print("\nNull counts per column:")
print(null_counts)

# Final summary
print("\nMerge completed successfully")
print(f"Files merged: {len(files)}")
print(f"Total rows: {combined.height}")
print(f"Columns: {combined.columns}")
