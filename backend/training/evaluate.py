"""
Comprehensive Model Evaluation Module
Evaluates trained ResNet18 model on test dataset.
Computes Accuracy, Precision, Recall/Sensitivity, Specificity, F1-Score, Confusion Matrix, ROC-AUC.
Generates publication-quality visualizations and persistent reports.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
)

def evaluate_model(
    model_path="backend/models/best_resnet18.pth",
    test_dir="data/balanced/test",
    labels_path="backend/models/class_labels.json",
    results_dir="results",
    batch_size=32,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    print(f"[Evaluation] Evaluating on device: {device}")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join("backend", "results"), exist_ok=True)
    
    # Load labels
    with open(labels_path, "r") as f:
        labels_meta = json.load(f)
    class_names = labels_meta["classes"]
    class_to_idx = labels_meta["class_to_idx"]
    idx_to_class = labels_meta.get("idx_to_class", {str(v): k for k, v in class_to_idx.items()})
    
    # Load model
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, len(class_names))
    )
    
    if not os.path.exists(model_path):
        # Fallback to alternative model path
        alt_path = os.path.join("backend", "models", "resnet18_skin_model.pth")
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
            
    print(f"[Evaluation] Loading checkpoint: {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # Preprocessing
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"[Evaluation] Test images: {len(test_dataset)} across classes: {test_dataset.classes}")
    
    all_targets = []
    all_predictions = []
    all_probabilities = []
    
    softmax = nn.Softmax(dim=1)
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = softmax(outputs)
            _, preds = outputs.max(1)
            
            all_targets.extend(targets.cpu().numpy().tolist())
            all_predictions.extend(preds.cpu().numpy().tolist())
            all_probabilities.extend(probs.cpu().numpy().tolist())
            
    y_true = np.array(all_targets)
    y_pred = np.array(all_predictions)
    y_prob = np.array(all_probabilities)
    
    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    prec_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Specificity and Sensitivity per class
    # For binary: 0=benign, 1=malignant
    per_class_metrics = {}
    if len(class_names) == 2:
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        per_class_metrics[class_names[0]] = {
            "sensitivity_recall": round(float(tn / (tn + fp)), 4) if (tn + fp) > 0 else 0.0,
            "specificity": round(float(tp / (tp + fn)), 4) if (tp + fn) > 0 else 0.0,
            "precision": round(float(tn / (tn + fn)), 4) if (tn + fn) > 0 else 0.0,
            "f1": round(float(2*tn / (2*tn + fp + fn)), 4) if (2*tn + fp + fn) > 0 else 0.0,
            "support": int(tn + fp)
        }
        per_class_metrics[class_names[1]] = {
            "sensitivity_recall": round(float(sensitivity), 4),
            "specificity": round(float(specificity), 4),
            "precision": round(float(tp / (tp + fp)), 4) if (tp + fp) > 0 else 0.0,
            "f1": round(float(2*tp / (2*tp + fp + fn)), 4) if (2*tp + fp + fn) > 0 else 0.0,
            "support": int(tp + fn)
        }
        binary_auc = roc_auc_score(y_true, y_prob[:, 1])
    else:
        sensitivity = rec_macro
        specificity = 0.0
        binary_auc = 0.0
        
    clf_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    
    print("\n--- Test Evaluation Results ---")
    print(f"Overall Accuracy: {acc*100:.2f}%")
    print(f"Macro Precision: {prec_macro*100:.2f}% | Recall: {rec_macro*100:.2f}% | F1-Score: {f1_macro*100:.2f}%")
    if len(class_names) == 2:
        print(f"Malignant Sensitivity (Recall): {sensitivity*100:.2f}%")
        print(f"Malignant Specificity: {specificity*100:.2f}%")
        print(f"ROC-AUC Score: {binary_auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    # 1. Confusion Matrix Heatmap
    sns.set_theme(style="white")
    plt.figure(figsize=(7, 6))
    cm_labels = [c.capitalize() for c in class_names]
    
    # Annotations with counts and percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]:,}\n({cm_percent[i, j]*100:.1f}%)"
            
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=True,
                xticklabels=cm_labels, yticklabels=cm_labels,
                annot_kws={"size": 12, "weight": "bold"})
    plt.title("Confusion Matrix on Test Set", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Predicted Disease Class", fontsize=12, fontweight='bold')
    plt.ylabel("Ground Truth Class", fontsize=12, fontweight='bold')
    plt.tight_layout()
    cm_chart = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_chart, dpi=300)
    plt.close()
    
    # 2. ROC Curves
    plt.figure(figsize=(8, 6))
    if len(class_names) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='#1565C0', lw=2.5, label=f"ROC curve (AUC = {roc_auc_val:.3f})")
    else:
        for i, c in enumerate(class_names):
            y_bin = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
            roc_auc_val = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"{c.capitalize()} (AUC = {roc_auc_val:.3f})")
            
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label="Chance (AUC = 0.500)")
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_chart = os.path.join(results_dir, "roc_curve.png")
    plt.savefig(roc_chart, dpi=300)
    plt.close()
    
    # Copy charts to backend/results
    import shutil
    shutil.copy2(cm_chart, os.path.join("backend", "results", "confusion_matrix.png"))
    shutil.copy2(roc_chart, os.path.join("backend", "results", "roc_curve.png"))
    
    # Save JSON metrics
    metrics_data = {
        "overall_accuracy": round(float(acc), 4),
        "macro_precision": round(float(prec_macro), 4),
        "macro_recall": round(float(rec_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_precision": round(float(prec_weighted), 4),
        "weighted_recall": round(float(rec_weighted), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "roc_auc": round(float(binary_auc), 4),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": per_class_metrics,
        "classification_report": clf_report,
        "test_samples_count": len(test_dataset)
    }
    
    with open(os.path.join(results_dir, "model_evaluation_metrics.json"), "w") as f:
        json.dump(metrics_data, f, indent=4)
    with open(os.path.join("backend", "results", "model_evaluation_metrics.json"), "w") as f:
        json.dump(metrics_data, f, indent=4)

    config_paths = [
        os.path.join("backend", "models", "preprocessing_config.json"),
        os.path.join("models", "preprocessing_config.json")
    ]
    for config_path in config_paths:
        config = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        config.update({
            "test_accuracy": round(float(acc), 4),
            "test_sensitivity": round(float(sensitivity), 4),
            "test_specificity": round(float(specificity), 4),
            "test_roc_auc": round(float(binary_auc), 4),
            "evaluation_dataset": os.path.normpath(test_dir)
        })
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        
    # Generate Evaluation Summary Markdown
    md_summary = f"""# ResNet18 Skin Disease Assessment Model Evaluation Report

**Evaluation Dataset:** Test split (`data/balanced/test`)  
**Total Test Samples:** {len(test_dataset):,}

## 1. Key Performance Indicators
- **Overall Accuracy:** {acc*100:.2f}%
- **Sensitivity / Recall:** {sensitivity*100:.2f}%
- **Specificity:** {specificity*100:.2f}%
- **Precision (Macro):** {prec_macro*100:.2f}%
- **F1-Score (Macro):** {f1_macro*100:.2f}%
- **ROC Area Under Curve (AUC):** {binary_auc:.4f}

## 2. Confusion Matrix
| Actual \\ Predicted | Benign | Malignant | Total |
|---|---|---|---|
| **Benign** | {cm[0, 0]:,} | {cm[0, 1]:,} | {cm[0].sum():,} |
| **Malignant** | {cm[1, 0]:,} | {cm[1, 1]:,} | {cm[1].sum():,} |
| **Total** | {cm[:, 0].sum():,} | {cm[:, 1].sum():,} | {len(test_dataset):,} |

## 3. Per-Class Classification Report
| Class | Precision | Recall / Sensitivity | Specificity | F1-Score | Support |
|---|---|---|---|---|---|
| **Benign** | {clf_report['benign']['precision']*100:.2f}% | {clf_report['benign']['recall']*100:.2f}% | {per_class_metrics.get('benign', {}).get('specificity', 0)*100:.2f}% | {clf_report['benign']['f1-score']*100:.2f}% | {clf_report['benign']['support']:,} |
| **Malignant** | {clf_report['malignant']['precision']*100:.2f}% | {clf_report['malignant']['recall']*100:.2f}% | {per_class_metrics.get('malignant', {}).get('specificity', 0)*100:.2f}% | {clf_report['malignant']['f1-score']*100:.2f}% | {clf_report['malignant']['support']:,} |

---
*Generated by Model Evaluation Pipeline*
"""
    with open(os.path.join(results_dir, "evaluation_summary_report.md"), "w") as f:
        f.write(md_summary)
        
    print(f"[Evaluation] Evaluation complete! Results saved to '{results_dir}'.")
    return metrics_data

if __name__ == "__main__":
    evaluate_model()
