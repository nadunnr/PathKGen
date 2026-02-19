import json
import os
import argparse
import random
from collections import Counter

def write_stats(data, output_log_file, title="Statistics"):
    total_count = len(data)
    
    with open(output_log_file, 'a') as f: # Append mode to keep history if called multiple times
        f.write(f"--- {title} ---\n")
        f.write(f"Total Data Points: {total_count}\n\n")
        
        if total_count == 0:
            f.write("No data points.\n\n")
            return

        # Helper for printing a section
        def print_section(title, items, count_map):
            f.write(f"{title}:\n")
            max_len = len("Name")
            if items:
                max_len = max(len(str(x)) for x in items)
            
            f.write(f"{'Name':<{max_len+2}} {'Count':<10} {'Percentage':<10}\n")
            f.write("-" * (max_len + 2 + 10 + 10 + 2) + "\n")
            
            for item, count in count_map.most_common():
                percentage = (count / total_count) * 100
                f.write(f"{str(item):<{max_len+2}} {count:<10} {percentage:<10.2f}%\n")
            f.write(f"Total entries: {len(count_map)}\n\n")

        fields_to_log = [
            ('task', 'Task Statistics'),
            ('project', 'Project Statistics'),
            ('project_name', 'Project Name Statistics'),
            ('primary_site', 'Primary Site Statistics'),
            ('disease_type', 'Disease Type Statistics')
        ]

        for field, log_title in fields_to_log:
            values = [str(item.get(field, 'N/A')) for item in data]
            counts = Counter(values)
            print_section(log_title, list(counts.keys()), counts)

def filter_wsi_bench(input_file, output_file, tasks, feature_dir, log_file, split_ratio, seed):
    print(f"I: Loading data from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    initial_count = len(data)
    print(f"I: Initial count: {initial_count}")
    
    # Initialize log file
    with open(log_file, 'w') as f:
        f.write(f"Filtering Log for {input_file}\n")
        f.write("========================================\n\n")
    
    # 1. Filter by Task
    if tasks:
        print(f"I: Filtering by tasks: {tasks}")
        data = [item for item in data if item.get('task') in tasks]
        print(f"I: Count after task filter: {len(data)}")
        
    # 2. Filter by Feature Existence
    if feature_dir:
        print(f"I: Filtering by feature existence in {feature_dir}...")
        filtered_data = []
        found_features = 0
        missing_features = 0
        
        # Get set of available files for faster lookup (assuming filenames are unique)
        # Assuming filename format: {tcga_id}.h5
        available_files = set(os.listdir(feature_dir))
        
        for item in data:
            tcga_id = item.get('tcga_id')
            if not tcga_id:
                missing_features += 1
                continue
                
            expected_filename = f"{tcga_id}.h5"
            
            if expected_filename in available_files:
                item['feature_path'] = os.path.join(feature_dir, expected_filename)
                filtered_data.append(item)
                found_features += 1
            else:
                missing_features += 1
        
        data = filtered_data
        print(f"I: Count after feature filter: {len(data)} (Found: {found_features}, Missing: {missing_features})")

    # Write stats after filtering
    write_stats(data, log_file, title="Post-Filtering Statistics")

    # Construct suffix based on tasks and split ratio
    task_suffix = f"_{'_'.join(tasks)}" if tasks else ""
    split_suffix = f"_split{split_ratio}" if split_ratio > 0 else ""
    
    base, ext = os.path.splitext(output_file)
    
    # 3. Handle Splitting
    if split_ratio > 0:
        print(f"I: Splitting data with test ratio {split_ratio} (seed={seed})...")
        random.seed(seed)
        random.shuffle(data)
        
        split_idx = int(len(data) * (1 - split_ratio))
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        # New filenames with suffixes
        train_output = f"{base}{task_suffix}{split_suffix}_train{ext}"
        test_output = f"{base}{task_suffix}{split_suffix}_test{ext}"
        
        print(f"I: Saving train set to {train_output} ({len(train_data)} samples)...")
        with open(train_output, 'w') as f:
            json.dump(train_data, f, indent=4)
            
        print(f"I: Saving test set to {test_output} ({len(test_data)} samples)...")
        with open(test_output, 'w') as f:
            json.dump(test_data, f, indent=4)
            
        write_stats(train_data, log_file, title="Train Split Statistics")
        write_stats(test_data, log_file, title="Test Split Statistics")
        
    else:
        # Update output_file name with task suffix if any
        final_output = f"{base}{task_suffix}{ext}"
        
        print(f"I: Saving filtered data to {final_output}...")
        with open(final_output, 'w') as f:
            json.dump(data, f, indent=4)

    print(f"I: Done. Log written to {log_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and split WSI-Bench dataset.")
    parser.add_argument("--input_file", help="Path to input processed JSON")
    parser.add_argument("--output_file", help="Path to save filtered JSON (or prefix for splits)")
    parser.add_argument("--tasks", nargs='+', help="List of tasks to keep")
    parser.add_argument("--feature_dir", help="Path to feature directory (.h5 files)")
    parser.add_argument("--log_file", help="Path for statistics logging")
    parser.add_argument("--split_ratio", type=float, default=0.0, help="Test set ratio (0.0 to 1.0). Default 0.0 (no split)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    
    args = parser.parse_args()

    # Preset paths (Defaults)
    input_train_file = "/home/nadun/wd/datasets/wsi_bench/dataset/WSI-Bench-train-formatted.json"
    input_test_file = "/home/nadun/wd/datasets/wsi_bench/dataset/WSI-Bench-test-formatted.json"
    
    output_train_file = "/home/nadun/wd/datasets/wsi_bench/dataset/WSI-Bench-train.json"
    output_test_file = "/home/nadun/wd/datasets/wsi_bench/dataset/WSI-Bench-test.json"
    
    # Defaults for other args
    default_train_log_file = "/home/nadun/wd/datasets/wsi_bench/dataset/WSI-Bench-train-filter-log.txt"
    default_test_log_file = "/home/nadun/wd/datasets/wsi_bench/dataset/WSI-Bench-test-filter-log.txt"
    default_train_feature_dir = "/home/nadun/wd/datasets/wsi_bench/titan_features/train/20x_512px_0px_overlap/slide_features_titan"
    default_test_feature_dir = "/home/nadun/wd/datasets/wsi_bench/titan_features/test/20x_512px_0px_overlap/slide_features_titan"
    default_tasks = ["Report"]

    filter_wsi_bench(
        input_file=input_train_file,
        output_file=output_train_file,
        tasks=args.tasks if args.tasks else default_tasks,
        feature_dir=args.feature_dir if args.feature_dir else default_train_feature_dir,
        log_file=args.log_file if args.log_file else default_train_log_file,
        split_ratio=0,
        seed=args.seed
    )

    filter_wsi_bench(
        input_file=input_test_file,
        output_file=output_test_file,
        tasks=args.tasks if args.tasks else default_tasks,
        feature_dir=args.feature_dir if args.feature_dir else default_test_feature_dir,
        log_file=args.log_file if args.log_file else default_test_log_file,
        split_ratio=0,
        seed=args.seed
    )
