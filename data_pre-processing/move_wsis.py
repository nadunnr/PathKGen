#!/usr/bin/env python3
"""
Safe script to MOVE all nested WSI files to a single parent directory.
This uses a two-phase approach for safety:
1. Move files (fast if on same filesystem)
2. Log all operations for easy rollback if needed

Source: /home/nadun/wd/datasets/tcga_report_histgen/primary_site_splits/breast_wsis
Destination: /home/nadun/wd/datasets/tcga_report_histgen/breast_wsi_dataset
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

# Configuration
SOURCE_DIR = Path("/home/sandun/nadun/lung_wsi")
DEST_DIR = Path("/home/sandun/nadun/lung_wsis")
LOG_FILE = Path("/home/sandun/nadun/lung_wsis/move_log.json")

def main():
    # Create destination directory if it doesn't exist
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all .svs files recursively
    wsi_files = list(SOURCE_DIR.rglob("*.svs"))
    
    print(f"Found {len(wsi_files)} WSI files to move")
    print(f"Source: {SOURCE_DIR}")
    print(f"Destination: {DEST_DIR}")
    print()
    
    # Check if they're on the same filesystem
    source_stat = os.stat(SOURCE_DIR)
    dest_stat = os.stat(DEST_DIR)
    same_filesystem = source_stat.st_dev == dest_stat.st_dev
    
    if same_filesystem:
        print("✓ Source and destination are on the same filesystem")
        print("  → Moving will be FAST (just updates directory entries)")
    else:
        print("⚠ Source and destination are on DIFFERENT filesystems")
        print("  → Moving will require copying then deleting (slower)")
    print()
    
    # Track all moves for safety
    move_log = {
        "timestamp": datetime.now().isoformat(),
        "source": str(SOURCE_DIR),
        "destination": str(DEST_DIR),
        "same_filesystem": same_filesystem,
        "moves": []
    }
    
    # Move files with progress bar
    success_count = 0
    failed_moves = []
    
    for wsi_file in tqdm(wsi_files, desc="Moving WSI files", unit="file"):
        filename = wsi_file.name
        dest_path = DEST_DIR / filename
        
        try:
            # Move the file (shutil.move handles both same/different filesystem)
            shutil.move(str(wsi_file), str(dest_path))
            
            # Log the successful move
            move_log["moves"].append({
                "filename": filename,
                "original_path": str(wsi_file),
                "new_path": str(dest_path),
                "status": "success"
            })
            success_count += 1
            
        except Exception as e:
            failed_moves.append({
                "filename": filename,
                "original_path": str(wsi_file),
                "error": str(e)
            })
            print(f"\n✗ Failed to move {filename}: {e}")
    
    # Save the log
    with open(LOG_FILE, 'w') as f:
        json.dump(move_log, f, indent=2)
    
    # Verify the move
    moved_files = list(DEST_DIR.glob("*.svs"))
    
    print(f"\n✓ Done!")
    print(f"  Files successfully moved: {success_count}")
    print(f"  Files in destination: {len(moved_files)}")
    print(f"  Failed moves: {len(failed_moves)}")
    if failed_moves:
        print(f"\n⚠ Failed files:")
        for fail in failed_moves:
            print(f"    - {fail['filename']}: {fail['error']}")
    
    print(f"\n📝 Move log saved to: {LOG_FILE}")
    print(f"   (You can use this to restore files if needed)")
    
    # Clean up empty directories (optional)
    response = input("\nDo you want to remove empty source directories? (y/n): ")
    if response.lower() == 'y':
        removed_dirs = 0
        for subdir in SOURCE_DIR.iterdir():
            if subdir.is_dir():
                try:
                    if not any(subdir.iterdir()):  # Check if empty
                        subdir.rmdir()
                        removed_dirs += 1
                except Exception as e:
                    print(f"Could not remove {subdir.name}: {e}")
        print(f"✓ Removed {removed_dirs} empty directories")

if __name__ == "__main__":
    main()
