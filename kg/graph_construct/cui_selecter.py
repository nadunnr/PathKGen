import matplotlib.pyplot as plt
from collections import Counter
import csv

all_cuis = []

with open("/home/nadun/wd/vlm_histo/data_pre-processing/extracted_cuis/cui_csvs/Breast Invasive Carcinoma.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        cuis = row['cuis'].split(";")  # split multiple CUIs
        all_cuis.extend(cuis)

cui_counter = Counter(all_cuis)

# Get top N most frequent CUIs
top_n = 10
top_cuis = cui_counter.most_common(top_n)
labels, counts = zip(*top_cuis)
numbered_labels = [f"{idx + 1}. {label}" for idx, label in enumerate(labels)]

plt.figure(figsize=(12,6))
plt.bar(numbered_labels, counts, color='skyblue')
plt.xticks(rotation=90)
plt.xlabel("CUI")
plt.ylabel("Frequency")
plt.title(f"Top {top_n} Most Frequent CUIs Across WSIs")
plt.tight_layout()

# Save the figure as PNG
plt.savefig("top_cuis_histogram.png", dpi=300)
plt.close() 
