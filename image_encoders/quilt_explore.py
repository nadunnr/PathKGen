import pandas as pd
from pathlib import Path
import os
import numpy as np
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Path to your CSV
# datapath = Path(config["datasets"]["quilt"])
# csv_path = datapath/ "quilt_1M_lookup.csv"
# img_path = Path(datapath["images"])

# Path to your CSV
datapath = Path("/home/nadun/hdd/QUILT1M")
csv_path = datapath / "quilt_1M_lookup.csv"
img_path = datapath / "quilt_1m"

# # Read first 5 rows
# chunk_iter = pd.read_csv(csv_path, chunksize=100)
# five_rows = next(chunk_iter)

columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
print("Columns:", columns)


head_sample = pd.read_csv(csv_path, nrows=5)
print("\nFirst 5 Rows:")
print(head_sample)

# Select the first row
first_row = head_sample.iloc[4]

# Convert the first row to dict and print key-value pairs
row_dict = first_row.to_dict()

for key, value in row_dict.items():
    print(f"{key}: {value}")

info = pd.read_csv(csv_path, nrows=1000)  # small chunk to infer types
print("\nColumn Info (from first 1000 rows):")
print(info.info())