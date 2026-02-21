import pandas as pd
import os

def merge_primary_site(mapping_csv, label_csv, output_csv):
    print(f"Processing: {label_csv}")
    
    # Load mapping
    mapping_df = pd.read_csv(mapping_csv)
    # tcga_id, report_content, project_name, primary_site, disease_type
    mapping = mapping_df[['tcga_id', 'primary_site']].drop_duplicates()
    
    # Load labels
    label_df = pd.read_csv(label_csv)
    # tcga_id, final_class, cui
    
    # Merge
    merged_df = label_df.merge(mapping, on='tcga_id', how='left')
    
    # Reorder columns to include primary_site (tcga_id, final_class, cui, primary_site)
    # The user said "include primary sitet init" which might mean just including it.
    
    # Save to parent folder
    merged_df.to_csv(output_csv, index=False)
    print(f"Saved to: {output_csv}")
    return merged_df.head()

# File paths
train_mapping = "/home/nadun/wd/datasets/wsi_bench/dataset/report_dataset_train.csv"
test_mapping = "/home/nadun/wd/datasets/wsi_bench/dataset/report_dataset_test.csv"

train_labels = "/home/nadun/wd/vlm_histo/concept-head/labelling/labels_train/variant_extracted_train.csv"
test_labels = "/home/nadun/wd/vlm_histo/concept-head/labelling/labels_test/variant_extracted_test.csv"

train_output = "/home/nadun/wd/vlm_histo/concept-head/labelling/labels_train/variant_train_with_site.csv"
test_output = "/home/nadun/wd/vlm_histo/concept-head/labelling/labels_test/variant_test_with_site.csv"

# Execute
merge_primary_site(train_mapping, train_labels, train_output)
merge_primary_site(test_mapping, test_labels, test_output)
