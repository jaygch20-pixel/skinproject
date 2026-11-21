"""
Dataset Inspection and Analysis Module
Automatically identifies dataset folder, analyzes images, detects corrupt/duplicate/invalid files,
checks dimensions/formats, and generates visual charts and comprehensive reports.
"""

import os
import sys
import glob
import hashlib
import json
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from image_deduplication import (
    PHASH_HAMMING_THRESHOLD,
    SSIM_THRESHOLD,
    detect_near_duplicate_pairs,
)

def find_dataset_root(search_dir="."):
    """Automatically find dataset root containing subfolders with image files."""
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    candidates = []
    
    # Priority check known patterns
    for candidate in [
        os.path.join(search_dir, "archive", "archive", "melanoma_cancer_dataset"),
        os.path.join(search_dir, "archive", "melanoma_cancer_dataset"),
        os.path.join(search_dir, "data", "original"),
        os.path.join(search_dir, "archive"),
    ]:
        if os.path.exists(candidate):
            candidates.append(candidate)
            
    for root, dirs, files in os.walk(search_dir):
        # Ignore git, node_modules, results, brain
        if any(ignored in root for ignored in [".git", "node_modules", ".gemini", "results", "brain"]):
            continue
        img_count = sum(1 for f in files if Path(f).suffix.lower() in valid_extensions)
        if img_count > 10:
            candidates.append(root)
            
    # Select the candidate that has train/test or class subfolders
    for c in candidates:
        subdirs = [d for d in os.listdir(c) if os.path.isdir(os.path.join(c, d))]
        if "train" in subdirs and "test" in subdirs:
            return os.path.abspath(c)
        if any(cat in [s.lower() for s in subdirs] for cat in ["benign", "malignant", "melanoma"]):
            return os.path.abspath(c)
            
    if candidates:
        return os.path.abspath(candidates[0])
    raise FileNotFoundError("Could not locate skin disease image dataset.")

def compute_file_hash(filepath, block_size=65536):
    """Compute SHA-256 hash of file content to detect duplicates."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read(block_size)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(block_size)
    return hasher.hexdigest()

def analyze_dataset(dataset_root, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join("backend", "results"), exist_ok=True)
    
    print(f"[Dataset Analysis] Found dataset root at: {dataset_root}")
    
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_files = []
    
    for root, dirs, files in os.walk(dataset_root):
        for f in files:
            ext = Path(f).suffix.lower()
            if ext in valid_extensions or ext in {".txt", ".csv", ".json", ".xml"}:
                all_files.append(os.path.join(root, f))
                
    print(f"[Dataset Analysis] Total files found: {len(all_files)}")
    
    classes_found = set()
    split_counts = defaultdict(lambda: defaultdict(int))
    class_total_counts = defaultdict(int)
    corrupted_files = []
    unreadable_files = []
    invalid_format_files = []
    
    hash_to_files = defaultdict(list)
    image_metadata = []
    
    formats_counter = Counter()
    dimensions_counter = Counter()
    aspect_ratios = []
    file_sizes_kb = []
    
    sample_images = defaultdict(list)
    
    for fpath in all_files:
        ext = Path(fpath).suffix.lower()
        if ext not in valid_extensions:
            invalid_format_files.append(fpath)
            continue
            
        # Determine split and class
        rel_path = os.path.relpath(fpath, dataset_root)
        parts = Path(rel_path).parts
        
        split = "all"
        cls_name = "unknown"
        if len(parts) >= 2:
            if parts[0].lower() in ["train", "test", "val", "validation"]:
                split = parts[0].lower()
                cls_name = parts[1].lower()
            else:
                cls_name = parts[0].lower()
        elif len(parts) == 1:
            cls_name = "unclassified"
            
        classes_found.add(cls_name)
        split_counts[split][cls_name] += 1
        class_total_counts[cls_name] += 1
        
        # Test readability and corruption
        try:
            with Image.open(fpath) as img:
                img.verify() # verify integrity
            with Image.open(fpath) as img:
                img.load() # load actual pixel data
                w, h = img.size
                mode = img.mode
                fmt = img.format or ext.replace(".", "").upper()
                
                formats_counter[fmt] += 1
                dimensions_counter[(w, h)] += 1
                aspect_ratios.append(round(w / h, 2))
                size_kb = os.path.getsize(fpath) / 1024.0
                file_sizes_kb.append(size_kb)
                
                if len(sample_images[cls_name]) < 5:
                    sample_images[cls_name].append(fpath)
                    
                image_metadata.append({
                    "path": fpath,
                    "class": cls_name,
                    "split": split,
                    "width": w,
                    "height": h,
                    "mode": mode,
                    "format": fmt,
                    "size_kb": size_kb
                })
        except Exception as e:
            corrupted_files.append({"path": fpath, "error": str(e)})
            continue
            
        # Check duplicate hash
        try:
            fhash = compute_file_hash(fpath)
            hash_to_files[fhash].append(fpath)
        except Exception as e:
            unreadable_files.append({"path": fpath, "error": str(e)})
            
    # Find duplicates
    duplicate_groups = {h: paths for h, paths in hash_to_files.items() if len(paths) > 1}
    total_duplicate_files = sum(len(paths) - 1 for paths in duplicate_groups.values())
    exact_unique_paths = [paths[0] for paths in hash_to_files.values()]
    near_duplicate_pairs = detect_near_duplicate_pairs(exact_unique_paths)
    
    print("\n--- Summary of Analysis ---")
    print(f"Classes: {sorted(list(classes_found))}")
    print(f"Total Images: {len(image_metadata)}")
    print(f"Total Classes Breakdown: {dict(class_total_counts)}")
    print(f"Split Breakdown: {json.dumps(split_counts, indent=2)}")
    print(f"Corrupted Images: {len(corrupted_files)}")
    print(f"Unreadable Images: {len(unreadable_files)}")
    print(f"Invalid Extension Files: {len(invalid_format_files)}")
    print(f"Exact Duplicate Images Detected: {total_duplicate_files} across {len(duplicate_groups)} groups")
    print(f"High-confidence Near-Duplicate Images Detected: {len(near_duplicate_pairs)} pairs")
    print(f"Image Formats: {dict(formats_counter)}")
    print(f"Most common dimensions: {dimensions_counter.most_common(5)}")
    
    # Generate Visualizations
    sns.set_theme(style="whitegrid")
    
    # 1. Number of images in each class before cleaning
    plt.figure(figsize=(8, 5))
    classes = sorted(list(class_total_counts.keys()))
    counts = [class_total_counts[c] for c in classes]
    colors = ['#2E86AB', '#D32F2F'] if len(classes) == 2 else sns.color_palette("mako", len(classes))
    bars = plt.bar(classes, counts, color=colors, width=0.5, edgecolor='black', linewidth=1)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 50, f'{yval:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.title("Original Dataset: Class Distribution Before Cleaning", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Skin Disease Category", fontsize=12, fontweight='bold')
    plt.ylabel("Number of Images", fontsize=12, fontweight='bold')
    plt.ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    chart1_path = os.path.join(output_dir, "class_distribution_before_cleaning.png")
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    
    # 2. Dataset Distribution (Pie Chart with breakdown)
    plt.figure(figsize=(7, 7))
    plt.pie(counts, labels=[f"{c.capitalize()}\n({n:,} images)" for c, n in zip(classes, counts)],
            autopct='%1.1f%%', startangle=140, colors=colors,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2},
            textprops={'fontsize': 12, 'fontweight': 'bold'})
    plt.title("Dataset Distribution Proportion", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    chart2_path = os.path.join(output_dir, "dataset_distribution.png")
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    
    # 3. Image Dimensions and Formats Visualization
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Formats bar chart
    fmts = list(formats_counter.keys())
    fmt_counts = [formats_counter[f] for f in fmts]
    axes[0].bar(fmts, fmt_counts, color='#3A86FF', edgecolor='black', width=0.4)
    axes[0].set_title("Image Formats Detected", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Count", fontsize=11)
    for i, v in enumerate(fmt_counts):
        axes[0].text(i, v + 20, str(v), ha='center', fontweight='bold')
        
    # Dimension scatter / distribution
    dims_labels = [f"{w}x{h}" for (w, h), cnt in dimensions_counter.most_common(5)]
    dims_vals = [cnt for (w, h), cnt in dimensions_counter.most_common(5)]
    axes[1].barh(dims_labels, dims_vals, color='#48CAE4', edgecolor='black')
    axes[1].set_title("Top 5 Image Dimensions (Width x Height)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Count", fontsize=11)
    for i, v in enumerate(dims_vals):
        axes[1].text(v + 10, i, f"{v:,}", va='center', fontweight='bold')
        
    plt.tight_layout()
    chart3_path = os.path.join(output_dir, "image_dimensions_and_formats.png")
    plt.savefig(chart3_path, dpi=300)
    plt.close()
    
    # 4. Sample Images from each category
    fig, axes = plt.subplots(len(classes), 5, figsize=(15, 3 * len(classes)))
    if len(classes) == 1:
        axes = np.expand_dims(axes, 0)
    for row_idx, cls in enumerate(classes):
        samples = sample_images[cls]
        for col_idx in range(5):
            ax = axes[row_idx, col_idx]
            if col_idx < len(samples):
                img = Image.open(samples[col_idx]).convert("RGB")
                ax.imshow(img)
                ax.set_title(f"{cls.capitalize()} sample {col_idx+1}", fontsize=10)
            ax.axis('off')
    plt.suptitle("Sample Dataset Images by Skin Disease Category", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    chart4_path = os.path.join(output_dir, "sample_images_per_category.png")
    plt.savefig(chart4_path, dpi=300)
    plt.close()
    
    # Copy charts to backend/results as well for API availability
    for p in [chart1_path, chart2_path, chart3_path, chart4_path]:
        dest = os.path.join("backend", "results", os.path.basename(p))
        with open(p, "rb") as src_f, open(dest, "wb") as dst_f:
            dst_f.write(src_f.read())
            
    # Save Report Data JSON
    report_data = {
        "dataset_root": dataset_root,
        "classes": sorted(list(classes_found)),
        "class_total_counts": dict(class_total_counts),
        "split_counts": {k: dict(v) for k, v in split_counts.items()},
        "total_images": len(image_metadata),
        "corrupted_images_count": len(corrupted_files),
        "corrupted_files": corrupted_files[:20],
        "unreadable_files_count": len(unreadable_files),
        "invalid_format_files_count": len(invalid_format_files),
        "duplicate_groups_count": len(duplicate_groups),
        "duplicate_images_total": total_duplicate_files,
        "duplicate_samples": {h: paths for h, paths in list(duplicate_groups.items())[:10]},
        "near_duplicate_pairs_count": len(near_duplicate_pairs),
        "near_duplicate_detection": {
            "phash_hamming_max": PHASH_HAMMING_THRESHOLD,
            "ssim_min": SSIM_THRESHOLD
        },
        "near_duplicate_samples": near_duplicate_pairs[:20],
        "formats": dict(formats_counter),
        "top_dimensions": {f"{w}x{h}": cnt for (w, h), cnt in dimensions_counter.most_common(5)},
        "average_file_size_kb": round(float(np.mean(file_sizes_kb)), 2) if file_sizes_kb else 0,
        "median_file_size_kb": round(float(np.median(file_sizes_kb)), 2) if file_sizes_kb else 0
    }
    
    with open(os.path.join(output_dir, "dataset_analysis_report.json"), "w") as f:
        json.dump(report_data, f, indent=4)
    with open(os.path.join("backend", "results", "dataset_analysis_report.json"), "w") as f:
        json.dump(report_data, f, indent=4)
        
    # Generate Markdown Report
    md_report = f"""# Skin Disease Dataset Inspection & Analysis Report

**Generated Automatically by CV Pipeline**
**Dataset Path:** `{dataset_root}`

## 1. Executive Summary
- **Identified Categories/Classes:** {', '.join([c.capitalize() for c in sorted(classes_found)])}
- **Total Image Samples Found:** {len(image_metadata):,}
- **Corrupted / Damaged Files:** {len(corrupted_files)}
- **Unreadable Files:** {len(unreadable_files)}
- **Invalid Extensions:** {len(invalid_format_files)}
- **Exact Duplicate Images:** {total_duplicate_files:,} duplicates across {len(duplicate_groups):,} hash clusters.
- **High-confidence Near-Duplicate Pairs:** {len(near_duplicate_pairs):,} (perceptual hash plus SSIM)

## 2. Class Distribution (Raw / Before Cleaning)
| Disease Category | Total Count | Proportion | Original Splits |
|---|---|---|---|
"""
    for cls in sorted(classes_found):
        cnt = class_total_counts[cls]
        pct = (cnt / len(image_metadata) * 100) if image_metadata else 0
        splits_detail = ", ".join([f"{s}: {split_counts[s][cls]:,}" for s in split_counts if cls in split_counts[s]])
        md_report += f"| **{cls.capitalize()}** | {cnt:,} | {pct:.1f}% | {splits_detail} |\n"
        
    md_report += f"""
## 3. Image Dimensions and Technical Characteristics
- **Predominant Dimensions:** {', '.join([f'{w}x{h} ({cnt:,} images)' for (w, h), cnt in dimensions_counter.most_common(3)])}
- **File Formats:** {', '.join([f'{fmt} ({cnt:,})' for fmt, cnt in formats_counter.items()])}
- **Average File Size:** {report_data['average_file_size_kb']} KB

## 4. Preprocessing & Cleaning Recommendations
1. **Deduplication:** Remove exact and high-confidence near-duplicate images before assigning training, validation, and test splits.
2. **Standardization:** Rescale all images uniformly to 224x224 RGB.
3. **Class Balancing Strategy:** Address benign/malignant imbalance via controlled minority augmentation in the training set only.

---
*Generated by Antigravity CV Engine*
"""
    with open(os.path.join(output_dir, "dataset_analysis_report.md"), "w") as f:
        f.write(md_report)
        
    print(f"[Dataset Analysis] Analysis complete! Reports and visualizations saved to '{output_dir}'.")
    return report_data

if __name__ == "__main__":
    dataset_root = find_dataset_root()
    analyze_dataset(dataset_root)
