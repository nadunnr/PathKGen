import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLASS_TO_REMOVE = "Testicular Germ Cell Tumors"

csv_files = [
    os.path.join(BASE_DIR, "variant_extracted_test.csv"),
    os.path.join(BASE_DIR, "variant_extracted_train.csv"),
]

for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    original_count = len(df)
    df = df[df["final_class"] != CLASS_TO_REMOVE]
    removed_count = original_count - len(df)
    df.to_csv(csv_path, index=False)
    print(f"{os.path.basename(csv_path)}: removed {removed_count} row(s), {len(df)} row(s) remaining.")
