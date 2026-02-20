import json
import pandas as pd

TRAIN_SET = "/home/nadun/wd/datasets/wsi_bench/dataset/WSI-Bench-train_Report.json"
TEST_SET = "/home/nadun/wd/datasets/wsi_bench/dataset/WSI-Bench-test_Report.json"

histo_train_path = '/home/nadun/wd/datasets/wsi_bench/dataset/histology_prediction_train.csv'
histo_test_path = '/home/nadun/wd/datasets/wsi_bench/dataset/histology_prediction_test.csv'

knowledge_train_set = "/home/nadun/wd/datasets/wsi_bench/dataset/WSI-Bench-train_Report_knowledge.json"
knowledge_test_set = "/home/nadun/wd/datasets/wsi_bench/dataset/WSI-Bench-test_Report_knowledge.json"

knowledge_path = '/home/nadun/wd/vlm_histo/kg/graph_construct/histology_descriptions_with_features.json'




def enrich_dataset(report_json_path, histo_csv_path, output_json_path, knowledge_dict):
    """
    Enrich the dataset with knowledge descriptions.
    
    Args:
        report_json_path: Path to the report JSON file (train/test_Report.json)
        histo_csv_path: Path to the histology CSV file (histology_prediction_train/test.csv)
        output_json_path: Path to save the enriched knowledge JSON file
        knowledge_dict: Dictionary mapping histology type to knowledge description
    """
    
    # Load the report JSON
    with open(report_json_path, 'r') as f:
        reports = json.load(f)
    
    # Load the histology CSV
    histo_df = pd.read_csv(histo_csv_path)
    
    # Create a mapping of tcga_id to histology
    tcga_to_histo = {}
    for _, row in histo_df.iterrows():
        tcga_to_histo[row['tcga_id']] = row['predicted_label'].strip()
    
    # Create an empty dictionary to store knowledge-enriched data
    knowledge_enriched = {}
    
    # Process each report entry
    for entry in reports:
        tcga_id = entry.get('tcga_id')
        
        if tcga_id is None:
            print(f"Warning: Entry has no tcga_id: {entry.get('id')}")
            continue
        
        # Create a copy of the entry to add knowledge
        enriched_entry = entry.copy()
        
        # Get histology from the CSV (fallback to project_name if not found)
        histology = tcga_to_histo.get(tcga_id, entry.get('project_name', 'Unknown'))
        
        # Look up knowledge description
        if histology in knowledge_dict:
            knowledge = knowledge_dict[histology]
        else:
            # If exact match not found, try to find a close match
            print(f"Warning: No exact match found for histology '{histology}' of TCGA ID {tcga_id}")
            knowledge = None
        
        # Add knowledge key to the entry
        enriched_entry['knowledge'] = knowledge
        enriched_entry['histology'] = histology  # Also add the histology for reference
        
        # Store using tcga_id as key in the dictionary
        knowledge_enriched[tcga_id] = enriched_entry
    
    # Save the enriched data
    with open(output_json_path, 'w') as f:
        json.dump(knowledge_enriched, f, indent=2)
    
    print(f"Successfully enriched {len(knowledge_enriched)} entries")
    print(f"Saved to: {output_json_path}")
    
    return knowledge_enriched


def main():
    # Load knowledge descriptions
    with open(knowledge_path, 'r') as f:
        knowledge_dict = json.load(f)
    
    print(f"Loaded {len(knowledge_dict)} knowledge descriptions")
    print("\n" + "="*80)
    
    # Process training set
    print("\nProcessing Training Set...")
    print(f"Input: {TRAIN_SET}")
    print(f"Output: {knowledge_train_set}")
    enrich_dataset(TRAIN_SET, histo_train_path, knowledge_train_set, knowledge_dict)
    
    print("\n" + "="*80)
    
    # Process test set
    print("\nProcessing Test Set...")
    print(f"Input: {TEST_SET}")
    print(f"Output: {knowledge_test_set}")
    enrich_dataset(TEST_SET, histo_test_path, knowledge_test_set, knowledge_dict)
    
    print("\n" + "="*80)
    print("\nAll datasets enriched successfully!")


if __name__ == "__main__":
    main()
