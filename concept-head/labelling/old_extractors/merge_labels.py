"""
Extract and merge labeled dataset from separate CSV files.

Combines histology, site, grade, and variant labels per TCGA ID.
"""

import csv
import os
from pathlib import Path
from collections import defaultdict


def extract_dataset(labels_dir, output_file):
    """
    Extract and merge labels from separate CSV files.
    
    Args:
        labels_dir: Directory containing the label CSV files
        output_file: Path to output merged CSV
        
    Returns:
        List of merged label dictionaries
    """
    labels_dir = Path(labels_dir)
    
    # Dictionary to store merged data: tcga_id -> {fields}
    merged_data = defaultdict(dict)
    
    # Read histology (base - all slides should have this)
    with open(labels_dir / "histology_extracted.csv", 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tcga_id = row['tcga_id']
            merged_data[tcga_id]['tcga_id'] = tcga_id
            merged_data[tcga_id]['histology_name'] = row['histology']
            merged_data[tcga_id]['histology_cui'] = row['cui']
    
    # Read site
    with open(labels_dir / "site_extracted.csv", 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tcga_id = row['tcga_id']
            if tcga_id in merged_data:
                merged_data[tcga_id]['site_name'] = row['site']
                merged_data[tcga_id]['site_cui'] = row['cui']
    
    # Read grade
    with open(labels_dir / "grade_extracted.csv", 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tcga_id = row['tcga_id']
            if tcga_id in merged_data:
                merged_data[tcga_id]['grade_name'] = row['grade']
                merged_data[tcga_id]['grade_cui'] = row['cui']
    
    # Read variants
    with open(labels_dir / "variants_extracted.csv", 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tcga_id = row['tcga_id']
            if tcga_id in merged_data:
                merged_data[tcga_id]['variant_name'] = row['variant']
                merged_data[tcga_id]['variant_cui'] = row['cui']
    
    # Convert to list and fill missing values
    merged_list = []
    for tcga_id, data in sorted(merged_data.items()):
        # Ensure all fields exist
        merged_list.append({
            'tcga_id': data.get('tcga_id', ''),
            'histology_name': data.get('histology_name', ''),
            'histology_cui': data.get('histology_cui', ''),
            'variant_name': data.get('variant_name', ''),
            'variant_cui': data.get('variant_cui', ''),
            'site_name': data.get('site_name', ''),
            'site_cui': data.get('site_cui', ''),
            'grade_name': data.get('grade_name', ''),
            'grade_cui': data.get('grade_cui', '')
        })
    
    # Write merged CSV
    fieldnames = [
        'tcga_id',
        'histology_name', 'histology_cui',
        'variant_name', 'variant_cui',
        'site_name', 'site_cui',
        'grade_name', 'grade_cui'
    ]
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_list)
    
    print(f"✅ Extracted {len(merged_list)} samples")
    print(f"   Saved to: {output_file}")
    
    # Calculate statistics
    unique_histologies = len(set(row['histology_name'] for row in merged_list if row['histology_name']))
    unique_sites = len(set(row['site_name'] for row in merged_list if row['site_name']))
    unique_grades = len(set(row['grade_name'] for row in merged_list if row['grade_name']))
    with_variants = sum(1 for row in merged_list if row['variant_name'])
    without_variants = len(merged_list) - with_variants
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total samples: {len(merged_list)}")
    print(f"   Unique histologies: {unique_histologies}")
    print(f"   Unique sites: {unique_sites}")
    print(f"   Unique grades: {unique_grades}")
    print(f"   Samples with variants: {with_variants}")
    print(f"   Samples without variants: {without_variants}")
    
    # Show sample rows
    print(f"\n📋 Sample rows:")
    for i, row in enumerate(merged_list[:5]):
        print(f"   {i+1}. {row['tcga_id'][:50]}...")
        print(f"      Histology: {row['histology_name']} ({row['histology_cui']})")
        if row['variant_name']:
            print(f"      Variant: {row['variant_name']} ({row['variant_cui']})")
        print(f"      Site: {row['site_name']} ({row['site_cui']})")
        print(f"      Grade: {row['grade_name']} ({row['grade_cui']})")
        print()
    
    return merged_list


def main():
    """Extract datasets for test and train splits."""
    
    base_dir = Path("/home/nadun/wd/vlm_histo/concept-head/labelling")
    
    # Extract test set
    print("="*80)
    print("EXTRACTING TEST SET")
    print("="*80)
    test_df = extract_dataset(
        labels_dir=base_dir / "labels_test",
        output_file=base_dir / "labels_test" / "merged_labels.csv"
    )
    
    # Extract train set if exists
    train_dir = base_dir / "labels_train"
    if train_dir.exists():
        print("\n" + "="*80)
        print("EXTRACTING TRAIN SET")
        print("="*80)
        
        # Check if all required files exist
        required_files = [
            "histology_extracted.csv",
            "site_extracted.csv", 
            "grade_extracted.csv",
            "variants_extracted.csv"
        ]
        
        if all((train_dir / f).exists() for f in required_files):
            train_df = extract_dataset(
                labels_dir=train_dir,
                output_file=train_dir / "merged_labels.csv"
            )
        else:
            print("⚠️  Train set files not complete, skipping")
    
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
