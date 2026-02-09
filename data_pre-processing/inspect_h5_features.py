#!/usr/bin/env python3
"""
Inspect H5 feature files in a directory
Check available keys and their shapes
"""

import os
import sys
import h5py
from pathlib import Path
from collections import defaultdict

def inspect_h5_files(directory, max_files=None):
    """
    Inspect all H5 files in a directory
    
    Args:
        directory: Path to directory containing .h5 files
        max_files: Maximum number of files to inspect (None = all)
    """
    # Find all H5 files
    h5_files = list(Path(directory).rglob("*.h5"))
    
    if not h5_files:
        print(f"No .h5 files found in {directory}")
        return
    
    print(f"Found {len(h5_files)} H5 files in {directory}")
    print("=" * 80)
    
    # Limit files if requested
    if max_files:
        h5_files = h5_files[:max_files]
        print(f"Inspecting first {max_files} files...\n")
    else:
        print(f"Inspecting all {len(h5_files)} files...\n")
    
    # Track key statistics across all files
    key_stats = defaultdict(lambda: {"count": 0, "shapes": defaultdict(int), "dtypes": set()})
    
    # Inspect each file
    # count number of files with each key
    key_counts = defaultdict(int)
    for h5_path in h5_files:
        with h5py.File(h5_path, 'r') as f:
            keys = list(f.keys())
            for key in keys:
                key_counts[key] += 1

    # print one file's keys and shapes
    with h5py.File(h5_files[0], 'r') as f:
        keys = list(f.keys())
        for key in keys:
            data = f[key]
            shape = data.shape
            dtype = data.dtype
            print(f"'{key}': shape={shape}, dtype={dtype}")
    
    print("Key counts:", key_counts)
    
    # Print summary statistics
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    for key, stats in sorted(key_stats.items()):
        print(f"\nKey: '{key}'")
        print(f"  Found in: {stats['count']}/{len(h5_files)} files ({100*stats['count']/len(h5_files):.1f}%)")
        print(f"  Shapes:")
        for shape, count in sorted(stats['shapes'].items(), key=lambda x: -x[1]):
            print(f"    {shape}: {count} files ({100*count/len(h5_files):.1f}%)")
        print(f"  Dtypes: {', '.join(sorted(stats['dtypes']))}")


if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("Usage: python inspect_h5_features.py <directory> [max_files]")
    #     print("\nExample:")
    #     print("  python inspect_h5_features.py /path/to/features")
    #     print("  python inspect_h5_features.py /path/to/features 10  # inspect first 10 files")
    #     sys.exit(1)
    
    directory = "/home/nadun/wd/datasets/wsi_bench/processed/20x_512px_0px_overlap/slide_features_titan"
    
    if not os.path.exists(directory):
        print(f"Error: Directory not found: {directory}")
        sys.exit(1)
    
    inspect_h5_files(directory)
