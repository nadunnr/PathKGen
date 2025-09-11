import pandas as pd
from pathlib import Path

# Path to your CSV
datapath = Path("/home/nadun/hdd/QUILT1M")
csv_path = datapath / "quilt_1M_lookup.csv"
img_path = datapath / "quilt_1m"

# Count .jpg files (case-insensitive, recursive)
jpg_count = sum(1 for _ in img_path.rglob("*.jpg"))

print(f"Number of .jpg images: {jpg_count}")

# # Read first 5 rows
# chunk_iter = pd.read_csv(csv_path, chunksize=5)
# five_rows = next(chunk_iter)

# # Select the first row
# first_row = five_rows.iloc[2]

# # Convert the first row to dict and print key-value pairs
# row_dict = first_row.to_dict()

# for key, value in row_dict.items():
#     print(f"{key}: {value}")