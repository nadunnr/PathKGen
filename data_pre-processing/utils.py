import glob
import json
import os 
import pandas as pd


def features_for_histology(directory, labels_file, output_dir):
    histologies = {}
    labels = pd.read_csv(labels_file)
    json_files = glob.glob(os.path.join(directory, "*.json"))
    for json_file in json_files:         
        with open(json_file, 'r') as f:
            json_data = json.load(f)
            for file in json_data:
                file_name = file['tcga_id']
                matching_rows = labels[labels['tcga_id'] == file_name]  # Find the disease type by csv
                if len(matching_rows) == 0:
                    continue
                disease_type = matching_rows['final_class'].values[0]   # If histologies dictionary has not that key, add it
                if disease_type not in histologies:
                    histologies[disease_type] = []
                # Append tumor features for this tcga_id
                histologies[disease_type].append({
                    'tcga_id': file_name,
                    'tumor_features': file['extraction']['tumor_features']
                })     

    for histology_name, samples in histologies.items():
        output_file = os.path.join(output_dir, f"{histology_name}.json")
        with open(output_file, 'w') as f:
            json.dump(samples, f, indent=2)
        print(f"Saved: {output_file} ({len(samples)} samples)")     

if __name__ == "__main__":
    directory = "/home/nadun/wd/datasets/wsi_bench/extracted_features/jsons/"
    labels_file = '/home/nadun/wd/vlm_histo/concept-head/labelling/labels_train/variant_extracted_train.csv'
    output_dir = "/home/nadun/wd/datasets/wsi_bench/extracted_features/outputs"
    features_for_histology(directory, labels_file, output_dir)
