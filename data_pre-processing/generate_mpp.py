import os
import openslide
import pandas as pd

wsi_dir = "/home/nadun/wd/datasets/tcga_wsis"
output_csv = "/home/nadun/wd/datasets/tcga_wsis/slide_list.csv"
default_missing_mpp = 0.25 # An educated guess

results = []

print(f"Generating TRIDENT custom list from: {wsi_dir}")

# Iterate and Extract Metadata
for filename in os.listdir(wsi_dir):
    if filename.endswith(".svs"):
        slide_path = os.path.join(wsi_dir, filename)
        try:
            slide = openslide.OpenSlide(slide_path)
            
            # Extract standard MPP tag
            mpp = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
            
            # If MPP is missing, assign default
            if mpp is None:
                mpp = default_missing_mpp
                print(f"Assigned default {default_missing_mpp} to: {filename}")
            
            # Trident requires 'wsi' (with extension) and 'mpp' columns
            results.append({
                "wsi": filename,
                "mpp": float(mpp)
            })
            slide.close()
        except Exception as e:
            print(f"Skipping {filename} due to error: {e}")

# 3. Save exactly in TRIDENT format
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)

print("-" * 30)
print(f"SUCCESS: {output_csv} generated with {len(df)} slides.")
print("-" * 30)