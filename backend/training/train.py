"""
Model Training Module for Skin Disease Assessment using PyTorch and ResNet18.
- Loads cleaned and balanced training dataset.
- Supports ResNet18 architecture.
- Handles ImageNet weights or robust He/Kaiming initialization when offline.
- Trains for exactly 5 epochs initially with Cross-Entropy Loss & Adam Optimizer.
- Tracks train & validation loss / accuracy at every epoch.
- Saves best checkpoint, labels, and preprocessing config.
- Plots and saves training/validation accuracy and loss curves.
"""

import os
import sys
import time
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
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def build_resnet18(num_classes=2, pretrained_requested=True):
    """
    Builds ResNet18 model for skin disease classification.
    Gracefully handles offline environments if pretrained weights cannot be downloaded.
    """
    weights_loaded = False
    init_mode = "ImageNet Pretrained"
    
    # Check if a cached checkpoint exists first
    cache_checkpoint = os.path.expanduser(r"~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
    if os.path.exists(cache_checkpoint):
        try:
            model = models.resnet18(weights=None)
            state = torch.load(cache_checkpoint, map_location="cpu")
            model.load_state_dict(state)
            weights_loaded = True
            print("[Model Builder] Loaded cached ResNet18 ImageNet weights from local cache.", flush=True)
        except Exception as e:
            print(f"[Model Builder] Could not load cached file: {e}", flush=True)
            
    if not weights_loaded:
        print("[Model Builder] Initializing ResNet18 architecture with Kaiming He normal weights for offline standalone execution.", flush=True)
        model = models.resnet18(weights=None)
        init_mode = "Kaiming He Initialized (Offline Standalone)"
        
    # Replace final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    
    # Custom initialization for newly added head
    nn.init.xavier_uniform_(model.fc[1].weight)
    nn.init.zeros_(model.fc[1].bias)
    
    return model, init_mode

def get_data_loaders(data_dir, batch_size=32, num_workers=0):
    """
    Create PyTorch DataLoaders with clinically sound augmentation for training.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.08, contrast=0.08),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    return train_loader, val_loader, train_dataset.classes, train_dataset.class_to_idx

def evaluate_loader(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            correct += outputs.argmax(1).eq(targets).sum().item()
            total += targets.size(0)

    if total == 0:
        raise ValueError("Cannot evaluate an empty dataset.")
    return running_loss / total, correct / total

def train_model(
    data_dir="data/balanced",
    models_dir="backend/models",
    results_dir="results",
    epochs=5,
    batch_size=32,
    lr=0.0003,
    device=None,
    resume_checkpoint=None,
    completed_epochs=0
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 1))
    print(f"[Training] Using compute device: {device}")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(os.path.join("models"), exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join("backend", "results"), exist_ok=True)
    
    train_loader, val_loader, class_names, class_to_idx = get_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size
    )
    
    print(f"[Training] Classes detected: {class_names} ({class_to_idx})")
    print(f"[Training] Training batches: {len(train_loader)} ({len(train_loader.dataset)} images)")
    print(f"[Training] Validation batches: {len(val_loader)} ({len(val_loader.dataset)} images)")
    
    # Build model
    model, init_mode = build_resnet18(num_classes=len(class_names), pretrained_requested=True)
    if resume_checkpoint:
        if completed_epochs <= 0 or completed_epochs >= epochs:
            raise ValueError("Resuming requires completed_epochs between 1 and epochs - 1.")
        if not os.path.isfile(resume_checkpoint):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")
        model.load_state_dict(torch.load(resume_checkpoint, map_location="cpu"))
        print(f"[Training] Resuming after epoch {completed_epochs} from {resume_checkpoint}")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
    
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_model_path = os.path.join(models_dir, "best_resnet18.pth")
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rate": [],
        "history_note": None
    }

    if resume_checkpoint:
        best_val_loss, best_val_acc = evaluate_loader(model, val_loader, criterion, device)
        history["history_note"] = (
            "Epoch 1 checkpoint was resumed, but its original training metrics were not saved. "
            "This history contains measured metrics for the resumed epochs only."
        )
    
    start_total_time = time.time()
    
    for epoch in range(completed_epochs + 1, epochs + 1):
        epoch_start = time.time()
        
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()
            
            if (batch_idx + 1) % 40 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  [Epoch {epoch}/{epochs}] Batch {batch_idx + 1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | Batch Acc: {100. * predicted.eq(targets).sum().item() / targets.size(0):.2f}%")
                
        train_loss = running_loss / total_train
        train_acc = correct_train / total_train
        
        # --- Validation Phase ---
        val_loss, val_acc = evaluate_loader(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        epoch_duration = time.time() - epoch_start
        
        print(f"\n=> Epoch {epoch}/{epochs} Complete in {epoch_duration:.1f}s | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}% (LR: {current_lr:.6f})")
        
        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["learning_rate"].append(current_lr)
        
        # Save best model
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            # Copy to top-level models/ as well
            torch.save(model.state_dict(), os.path.join("models", "best_resnet18.pth"))
            print(f"[*] Best checkpoint saved (Val Acc: {val_acc*100:.2f}%, Val Loss: {val_loss:.4f})")
            
    total_duration = time.time() - start_total_time
    print(f"\n[Training Complete] Total time: {total_duration/60:.2f} mins. Best Val Acc: {best_val_acc*100:.2f}%")
    
    # Save final model state
    final_model_path = os.path.join(models_dir, "resnet18_skin_model.pth")
    torch.save(model.state_dict(), final_model_path)
    torch.save(model.state_dict(), os.path.join("models", "resnet18_skin_model.pth"))
    
    # Save class labels
    idx_to_class = {str(v): k for k, v in class_to_idx.items()}
    labels_content = {
        "classes": class_names,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class
    }
    with open(os.path.join(models_dir, "class_labels.json"), "w") as f:
        json.dump(labels_content, f, indent=4)
    with open(os.path.join("models", "class_labels.json"), "w") as f:
        json.dump(labels_content, f, indent=4)
        
    # Save preprocessing configuration
    preprocessing_config = {
        "model_architecture": "ResNet18",
        "input_size": [3, 224, 224],
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "color_space": "RGB",
        "interpolation": "LANCZOS",
        "training_epochs": epochs,
        "batch_size": batch_size,
        "optimizer": "Adam",
        "initial_lr": lr,
        "initialization_mode": init_mode,
        "best_val_accuracy": round(best_val_acc, 4),
        "best_val_loss": round(best_val_loss, 4),
        "training_run_duration_seconds": round(total_duration, 1),
        "resumed_after_epoch": completed_epochs,
        "optimizer_state_restored": False if resume_checkpoint else None
    }
    with open(os.path.join(models_dir, "preprocessing_config.json"), "w") as f:
        json.dump(preprocessing_config, f, indent=4)
    with open(os.path.join("models", "preprocessing_config.json"), "w") as f:
        json.dump(preprocessing_config, f, indent=4)
        
    # Save history JSON
    with open(os.path.join(results_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)
    with open(os.path.join("backend", "results", "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)
        
    # Generate Training and Validation Accuracy & Loss Graphs
    sns.set_theme(style="whitegrid")
    
    # 1. Accuracy Graph
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], [a * 100 for a in history["train_acc"]], 'o-', label="Training Accuracy", color="#1E88E5", linewidth=2.5)
    plt.plot(history["epoch"], [a * 100 for a in history["val_acc"]], 's-', label="Validation Accuracy", color="#43A047", linewidth=2.5)
    plt.title("ResNet18 Training & Validation Accuracy (Recorded Epochs)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Epoch", fontsize=12, fontweight='bold')
    plt.ylabel("Accuracy (%)", fontsize=12, fontweight='bold')
    plt.xticks(history["epoch"])
    plt.ylim(0, 105)
    plt.legend(fontsize=11)
    plt.tight_layout()
    acc_chart = os.path.join(results_dir, "training_validation_accuracy.png")
    plt.savefig(acc_chart, dpi=300)
    plt.close()
    
    # 2. Loss Graph
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_loss"], 'o-', label="Training Loss", color="#E53935", linewidth=2.5)
    plt.plot(history["epoch"], history["val_loss"], 's-', label="Validation Loss", color="#FB8C00", linewidth=2.5)
    plt.title("ResNet18 Training & Validation Loss (Recorded Epochs)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Epoch", fontsize=12, fontweight='bold')
    plt.ylabel("Cross-Entropy Loss", fontsize=12, fontweight='bold')
    plt.xticks(history["epoch"])
    plt.legend(fontsize=11)
    plt.tight_layout()
    loss_chart = os.path.join(results_dir, "training_validation_loss.png")
    plt.savefig(loss_chart, dpi=300)
    plt.close()
    
    # Copy charts to backend/results
    import shutil
    shutil.copy2(acc_chart, os.path.join("backend", "results", "training_validation_accuracy.png"))
    shutil.copy2(loss_chart, os.path.join("backend", "results", "training_validation_loss.png"))
    
    print(f"[Training] Saved model artifacts to {models_dir} and graphs to {results_dir}.")
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/balanced")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--completed-epochs", type=int, default=0)
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume_checkpoint=args.resume_checkpoint,
        completed_epochs=args.completed_epochs
    )
