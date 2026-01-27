#!/usr/bin/env python3
"""
Match test video labels by comparing first frame images between test set and archive.
Uses image hash comparison for matching.
"""

import zipfile
import hashlib
import os
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Paths
PROJECT_DIR = Path(__file__).parent.parent
ACTION_VIDEO_ZIP = PROJECT_DIR / "action-video.zip"
ARCHIVE_ZIP = PROJECT_DIR / "archive.zip"
OUTPUT_DIR = PROJECT_DIR / "data"
OUTPUT_FILE = OUTPUT_DIR / "test_labels.json"

def get_image_hash(data: bytes) -> str:
    """Get MD5 hash of image data."""
    return hashlib.md5(data).hexdigest()

def extract_first_images_from_zip(zip_path: Path, pattern: str, desc: str) -> dict:
    """
    Extract first image (10000.jpg) from each folder matching pattern.
    Returns dict: {folder_path: image_hash}
    """
    folder_hashes = {}
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Get all files matching pattern with 10000.jpg
        matching_files = [f for f in zf.namelist() if pattern in f and f.endswith('/10000.jpg')]
        
        print(f"Found {len(matching_files)} folders in {zip_path.name}")
        
        for file_path in tqdm(matching_files, desc=desc):
            try:
                data = zf.read(file_path)
                img_hash = get_image_hash(data)
                # Extract folder path (parent of 10000.jpg)
                folder = str(Path(file_path).parent)
                folder_hashes[folder] = img_hash
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    return folder_hashes

def main():
    print("=" * 60)
    print("Matching Test Labels by First Frame Comparison")
    print("=" * 60)
    
    # Step 1: Extract hashes from test set
    print("\n[1/3] Extracting first images from test set...")
    test_hashes = extract_first_images_from_zip(
        ACTION_VIDEO_ZIP, 
        "data/test/", 
        "Test set"
    )
    print(f"  → Found {len(test_hashes)} test videos")
    
    # Step 2: Extract hashes from archive (original dataset)
    print("\n[2/3] Extracting first images from archive (this may take a while)...")
    archive_hashes = extract_first_images_from_zip(
        ARCHIVE_ZIP,
        "HMDB51/",
        "Archive"
    )
    print(f"  → Found {len(archive_hashes)} archive videos")
    
    # Step 3: Match hashes and get labels
    print("\n[3/3] Matching test videos to original labels...")
    
    # Build reverse lookup: hash -> (folder_path, class_name)
    hash_to_archive = {}
    for folder, h in archive_hashes.items():
        # Extract class from path: HMDB51/{class}/{video_name}
        parts = folder.split('/')
        if len(parts) >= 2:
            class_name = parts[1]  # HMDB51/{class}/...
            hash_to_archive[h] = (folder, class_name)
    
    # Match test videos
    test_labels = {}
    matched = 0
    unmatched = []
    
    for test_folder, test_hash in tqdm(test_hashes.items(), desc="Matching"):
        # Extract test ID from path: data/test/{id}
        test_id = Path(test_folder).name
        
        if test_hash in hash_to_archive:
            archive_folder, class_name = hash_to_archive[test_hash]
            test_labels[test_id] = {
                "class": class_name,
                "original_folder": archive_folder
            }
            matched += 1
        else:
            unmatched.append(test_id)
            test_labels[test_id] = {
                "class": "UNKNOWN",
                "original_folder": None
            }
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(test_labels, f, indent=2)
    
    # Also save as CSV for easy viewing
    csv_file = OUTPUT_DIR / "test_labels.csv"
    with open(csv_file, 'w') as f:
        f.write("id,class\n")
        for test_id in sorted(test_labels.keys(), key=lambda x: int(x)):
            f.write(f"{test_id},{test_labels[test_id]['class']}\n")
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total test videos: {len(test_hashes)}")
    print(f"Matched: {matched}")
    print(f"Unmatched: {len(unmatched)}")
    if unmatched:
        print(f"Unmatched IDs: {unmatched[:10]}...")
    print(f"\nLabels saved to:")
    print(f"  - {OUTPUT_FILE}")
    print(f"  - {csv_file}")
    
    # Print class distribution
    class_counts = defaultdict(int)
    for info in test_labels.values():
        class_counts[info['class']] += 1
    
    print(f"\nClass distribution ({len(class_counts)} classes):")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {cls}: {count}")
    if len(class_counts) > 10:
        print(f"  ... and {len(class_counts) - 10} more classes")

if __name__ == "__main__":
    main()
