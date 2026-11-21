"""
Data Cleaning, Deduplication, Standardization, and Stratified Splitting Pipeline
- Identifies and discards corrupted / unreadable / duplicate files.
- Converts images into RGB format.
- Resizes images to 224x224 pixels.
- Splits into 70% Train / 15% Validation / 15% Test with stratified sampling.
- Guarantees zero data leakage across splits.
"""

import os
import sys
import shutil
import hashlib
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from image_deduplication import (
    PHASH_HAMMING_THRESHOLD,
    SSIM_THRESHOLD,
    detect_near_duplicate_pairs,
)

def compute_file_hash(filepath, block_size=65536):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read(block_size)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(block_size)
    return hasher.hexdigest()

def clean_and_split_dataset(
    dataset_root,
    output_dir="data/cleaned",
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    target_size=(224, 224),
    seed=42
):
    print(f"[Cleaning Pipeline] Source dataset: {dataset_root}")
    print(f"[Cleaning Pipeline] Output directory: {output_dir}")
    print(f"[Cleaning Pipeline] Split ratios -> Train: {train_ratio*100:.0f}%, Val: {val_ratio*100:.0f}%, Test: {test_ratio*100:.0f}%")
    
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    raw_files = []
    
    for root, dirs, files in os.walk(dataset_root):
        for f in files:
            if Path(f).suffix.lower() in valid_extensions:
                raw_files.append(os.path.join(root, f))
                
    print(f"[Cleaning Pipeline] Found {len(raw_files)} raw candidate image files.")
    
    # 1. Deduplication and validation
    seen_hashes = {}
    valid_samples = []
    corrupted_count = 0
    duplicate_count = 0
    
    for fpath in raw_files:
        rel = os.path.relpath(fpath, dataset_root)
        parts = Path(rel).parts
        
        # Determine class label
        if len(parts) >= 2 and parts[0].lower() in ["train", "test", "val", "validation"]:
            label = parts[1].lower()
        else:
            label = parts[0].lower()
            
        # Verify image integrity
        try:
            with Image.open(fpath) as img:
                img.verify()
            with Image.open(fpath) as img:
                img.load()
        except Exception as e:
            corrupted_count += 1
            continue
            
        # Check hash
        try:
            fhash = compute_file_hash(fpath)
            if fhash in seen_hashes:
                duplicate_count += 1
                continue
            seen_hashes[fhash] = fpath
            valid_samples.append((fpath, label))
        except Exception:
            corrupted_count += 1
            continue
            
    print(f"[Cleaning Pipeline] Validation complete:")
    print(f"  - Valid unique samples: {len(valid_samples)}")
    print(f"  - Deduplicated files removed: {duplicate_count}")
    print(f"  - Corrupted files skipped: {corrupted_count}")

    near_duplicate_pairs = detect_near_duplicate_pairs([path for path, _ in valid_samples])
    near_duplicate_paths = {pair["duplicate"] for pair in near_duplicate_pairs}
    valid_samples = [
        sample for sample in valid_samples
        if sample[0] not in near_duplicate_paths
    ]
    print(f"  - High-confidence near-duplicate files removed: {len(near_duplicate_paths)}")
    
    # Organize by class
    class_groups = defaultdict(list)
    for path, label in valid_samples:
        class_groups[label].append(path)
        
    for label, paths in class_groups.items():
        print(f"  - Class '{label}': {len(paths)} unique samples")
        
    # Prepare output directories
    for split in ["train", "val", "test"]:
        for label in class_groups.keys():
            class_dir = Path(output_dir) / split / label
            class_dir.mkdir(parents=True, exist_ok=True)
            for existing_file in class_dir.glob("*.jpg"):
                existing_file.unlink()
            
    split_manifest = {
        "train": defaultdict(int),
        "val": defaultdict(int),
        "test": defaultdict(int)
    }
    
    rng = np.random.RandomState(seed)
    
    for label, paths in class_groups.items():
        shuffled = paths.copy()
        rng.shuffle(shuffled)
        
        n_total = len(shuffled)
        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))
        # Test gets the remainder to ensure exact sum = n_total
        n_test = n_total - n_train - n_val
        
        train_paths = shuffled[:n_train]
        val_paths = shuffled[n_train:n_train + n_val]
        test_paths = shuffled[n_train + n_val:]
        
        splits_dict = {
            "train": train_paths,
            "val": val_paths,
            "test": test_paths
        }
        
        for split_name, split_files in splits_dict.items():
            for idx, src_path in enumerate(split_files):
                dest_filename = f"{label}_{idx:05d}.jpg"
                dest_path = os.path.join(output_dir, split_name, label, dest_filename)
                
                # Convert to RGB and resize to 224x224
                with Image.open(src_path) as img:
                    rgb_img = img.convert("RGB")
                    resized_img = rgb_img.resize(target_size, Image.Resampling.LANCZOS)
                    resized_img.save(dest_path, "JPEG", quality=95)
                    
                split_manifest[split_name][label] += 1
                
    print("\n[Cleaning Pipeline] Splitting and image processing complete:")
    for split_name, counts in split_manifest.items():
        total_split = sum(counts.values())
        print(f"  - {split_name.capitalize()} ({total_split} images): {dict(counts)}")
        
    manifest_path = os.path.join(output_dir, "cleaning_split_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "target_size": target_size,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "splits": {k: dict(v) for k, v in split_manifest.items()},
            "exact_duplicate_files_removed": duplicate_count,
            "near_duplicate_files_removed": len(near_duplicate_paths),
            "near_duplicate_pairs": near_duplicate_pairs,
            "near_duplicate_detection": {
                "phash_hamming_max": PHASH_HAMMING_THRESHOLD,
                "ssim_min": SSIM_THRESHOLD
            }
        }, f, indent=4)
        
    return split_manifest

if __name__ == "__main__":
    from dataset_analysis import find_dataset_root
    dataset_root = find_dataset_root()
    clean_and_split_dataset(dataset_root)
