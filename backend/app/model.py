"""
Model Loading and Management Module
Maintains cached singleton instance of trained ResNet18 model,
class labels, and preprocessing hyperparameters.
"""

import os
import json
import torch
import torch.nn as nn
from torchvision import models

class SkinModelManager:
    _instance = None
    
    def __init__(self, models_dir="backend/models"):
        self.models_dir = models_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names = ["benign", "malignant"]
        self.class_to_idx = {"benign": 0, "malignant": 1}
        self.idx_to_class = {"0": "benign", "1": "malignant"}
        self.config = {}
        self.is_loaded = False
        
    @classmethod
    def get_instance(cls, models_dir="backend/models"):
        if cls._instance is None:
            cls._instance = cls(models_dir)
            cls._instance.load_model()
        return cls._instance
        
    def load_model(self):
        # 1. Load labels if present
        labels_path = os.path.join(self.models_dir, "class_labels.json")
        if not os.path.exists(labels_path):
            labels_path = os.path.join("models", "class_labels.json")
            
        if os.path.exists(labels_path):
            try:
                with open(labels_path, "r") as f:
                    meta = json.load(f)
                    self.class_names = meta.get("classes", self.class_names)
                    self.class_to_idx = meta.get("class_to_idx", self.class_to_idx)
                    self.idx_to_class = meta.get("idx_to_class", {str(v): k for k, v in self.class_to_idx.items()})
            except Exception as e:
                print(f"[Model Manager] Warning: could not parse {labels_path}: {e}")
                
        # 2. Load config if present
        config_path = os.path.join(self.models_dir, "preprocessing_config.json")
        if not os.path.exists(config_path):
            config_path = os.path.join("models", "preprocessing_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    self.config = json.load(f)
            except Exception as e:
                print(f"[Model Manager] Warning: could not parse {config_path}: {e}")
                
        # 3. Build model architecture
        num_classes = len(self.class_names)
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
        
        # 4. Find checkpoint
        candidates = [
            os.path.join(self.models_dir, "best_resnet18.pth"),
            os.path.join(self.models_dir, "resnet18_skin_model.pth"),
            os.path.join("models", "best_resnet18.pth"),
            os.path.join("models", "resnet18_skin_model.pth"),
        ]
        
        loaded = False
        for checkpoint_path in candidates:
            if os.path.exists(checkpoint_path):
                try:
                    print(f"[Model Manager] Loading weights from: {checkpoint_path}")
                    state_dict = torch.load(checkpoint_path, map_location=self.device)
                    model.load_state_dict(state_dict)
                    loaded = True
                    break
                except Exception as e:
                    print(f"[Model Manager] Could not load checkpoint from {checkpoint_path}: {e}")
                    
        self.model = model.to(self.device)
        self.model.eval()
        self.is_loaded = loaded
        
        # Warmup forward pass
        try:
            dummy_input = torch.zeros((1, 3, 224, 224), dtype=torch.float32, device=self.device)
            with torch.no_grad():
                _ = self.model(dummy_input)
            print("[Model Manager] Model loaded and warmed up successfully.")
        except Exception as e:
            print(f"[Model Manager] Warmup pass notice: {e}")
            
    def get_info(self):
        """Return diagnostic and structural model information."""
        return {
            "model_name": "ResNet18-SkinDisease-Classifier",
            "architecture": "ResNet18 (Residual Neural Network)",
            "pretrained_weights": self.config.get("initialization_mode", "ResNet18 (Kaiming He Initialized)"),
            "input_resolution": "224x224 RGB",
            "device": str(self.device),
            "classes": self.class_names,
            "num_classes": len(self.class_names),
            "training_epochs": self.config.get("training_epochs", 5),
            "is_loaded": self.is_loaded,
            "config": self.config
        }
