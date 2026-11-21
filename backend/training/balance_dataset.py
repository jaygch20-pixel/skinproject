"""
Dataset Class Balancing Module
Applies controlled dermatological data augmentation to the minority class in the training split.
Preserves validation and testing sets unchanged.
Generates before-and-after class distribution graphs and computes balanced loss weights.
"""

import os
import shutil
import json
import random
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageEnhance

def augment_skin_image(img):
    """
    Apply clinically realistic skin lesion augmentation:
    - Random horizontal flip
    - Subtle rotation (-15 to +15 deg) with reflective padding or edge clamp
    - Gentle brightness variation (0.92 to 1.08)
    - Gentle contrast variation (0.92 to 1.08)
    """
    # 1. Flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
    # 2. Subtle rotation
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(180, 150, 140))
    
    # 3. Brightness
    factor = random.uniform(0.92, 1.08)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(factor)
    
    # 4. Contrast
    factor_c = random.uniform(0.92, 1.08)
    enhancer_c = ImageEnhance.Contrast(img)
    img = enhancer_c.enhance(factor_c)
    
    return img

def balance_training_set(
    cleaned_dir="data/cleaned",
    balanced_dir="data/balanced",
    results_dir="results",
    seed=42
):
    random.seed(seed)
    np.random.seed(seed)
    
    os.makedirs(balanced_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join("backend", "results"), exist_ok=True)
    
    train_cleaned = os.path.join(cleaned_dir, "train")
    if not os.path.exists(train_cleaned):
        raise FileNotFoundError(f"Cleaned train folder not found at: {train_cleaned}")
        
    classes = [d for d in os.listdir(train_cleaned) if os.path.isdir(os.path.join(train_cleaned, d))]
    classes.sort()
    
    # Count images in each class in train
    before_counts = {}
    train_files = {}
    for cls in classes:
        cls_dir = os.path.join(train_cleaned, cls)
        files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        before_counts[cls] = len(files)
        train_files[cls] = files
        
    print(f"[Class Balancing] Cleaned Training Set counts before balancing: {before_counts}")
    max_count = max(before_counts.values())
    min_count = min(before_counts.values())
    imbalance_ratio = max_count / max(min_count, 1)
    
    print(f"[Class Balancing] Max count: {max_count}, Min count: {min_count}, Ratio: {imbalance_ratio:.2f}")
    
    # Prepare balanced train directory
    for split in ["train", "val", "test"]:
        for cls in classes:
            class_dir = Path(balanced_dir) / split / cls
            class_dir.mkdir(parents=True, exist_ok=True)
            for existing_file in class_dir.iterdir():
                if existing_file.is_file() and existing_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    existing_file.unlink()
            
    # Copy Validation and Test sets directly and completely untouched
    for split in ["val", "test"]:
        src_split = os.path.join(cleaned_dir, split)
        dst_split = os.path.join(balanced_dir, split)
        for cls in classes:
            src_cls = os.path.join(src_split, cls)
            dst_cls = os.path.join(dst_split, cls)
            for f in os.listdir(src_cls):
                shutil.copy2(os.path.join(src_cls, f), os.path.join(dst_cls, f))
                
    after_counts = {}
    
    # Process train split
    for cls in classes:
        src_files = train_files[cls]
        dst_cls = os.path.join(balanced_dir, "train", cls)
        
        # First copy all existing original cleaned images
        for f in src_files:
            shutil.copy2(f, os.path.join(dst_cls, os.path.basename(f)))
            
        needed = max_count - len(src_files)
        if needed > 0:
            print(f"[Class Balancing] Augmenting minority class '{cls}' by +{needed} realistic synthetic variations...")
            # Sample with replacement from existing images and apply gentle augmentation
            aug_idx = 0
            while aug_idx < needed:
                pick = random.choice(src_files)
                with Image.open(pick) as img:
                    aug_img = augment_skin_image(img.convert("RGB"))
                    aug_filename = f"{cls}_aug_{aug_idx:05d}.jpg"
                    aug_img.save(os.path.join(dst_cls, aug_filename), "JPEG", quality=95)
                aug_idx += 1
                
        after_counts[cls] = max_count
        
    print(f"[Class Balancing] Balanced Training Set counts: {after_counts}")
    
    # Calculate Class Weights
    total_samples = sum(before_counts.values())
    num_classes = len(classes)
    class_weights = {}
    for cls, cnt in before_counts.items():
        # sklearn style: N / (K * n_c)
        class_weights[cls] = round(total_samples / (num_classes * cnt), 4)
    print(f"[Class Balancing] Computed Class Weights for Loss Function: {class_weights}")
    
    # Generate Before-and-After Class Distribution Graph
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(classes))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, [before_counts[c] for c in classes], width, label='Before Balancing (Cleaned)', color='#5C6BC0', edgecolor='black')
    rects2 = ax.bar(x + width/2, [after_counts[c] for c in classes], width, label='After Balancing (Augmented)', color='#26A69A', edgecolor='black')
    
    ax.set_ylabel('Number of Training Images', fontsize=12, fontweight='bold')
    ax.set_title('Training Dataset Class Distribution: Before vs After Balancing', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in classes], fontsize=11, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(0, max_count * 1.18)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:,}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
            
    autolabel(rects1)
    autolabel(rects2)
    plt.tight_layout()
    
    chart_path = os.path.join(results_dir, "before_and_after_class_balancing.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    
    # Copy to backend/results
    shutil.copy2(chart_path, os.path.join("backend", "results", "before_and_after_class_balancing.png"))
    
    # Save Report
    report = {
        "classes": classes,
        "before_balancing": before_counts,
        "after_balancing": after_counts,
        "augmented_samples_added": {cls: after_counts[cls] - before_counts[cls] for cls in classes},
        "class_weights": class_weights,
        "imbalance_ratio_before": round(imbalance_ratio, 4)
    }
    with open(os.path.join(results_dir, "balancing_report.json"), "w") as f:
        json.dump(report, f, indent=4)
    with open(os.path.join("backend", "results", "balancing_report.json"), "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"[Class Balancing] Pipeline finished. Visualizations saved to '{chart_path}'.")
    return report

if __name__ == "__main__":
    balance_training_set()
