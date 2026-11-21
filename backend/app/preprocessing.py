"""
Image Preprocessing and Validation Module
Handles image decoding (file uploads and base64 webcam frames), format validation,
color normalization, and tensor preparation for ResNet18.
"""

import io
import re
import base64
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB
TARGET_SIZE = (224, 224)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# PyTorch transformation pipeline
eval_transform = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

def decode_image_bytes(image_bytes: bytes) -> Image.Image:
    """Decode raw bytes into PIL Image with verification."""
    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise ValueError(f"Image file exceeds maximum allowable size of {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB.")
    if len(image_bytes) < 100:
        raise ValueError("Image file is empty or too small.")
        
    try:
        stream = io.BytesIO(image_bytes)
        img = Image.open(stream)
        img.verify()
        
        # Re-open after verify
        stream.seek(0)
        img = Image.open(stream)
        img.load()
        return img.convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to read image: corrupted or invalid image format. Error: {str(e)}")

def decode_base64_image(base64_str: str) -> Image.Image:
    """Decode a base64 encoded image string (e.g., from HTML canvas/camera feed)."""
    if not base64_str:
        raise ValueError("Empty image data provided.")
        
    # Strip data URI header if present
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
        
    try:
        raw_bytes = base64.b64decode(base64_str)
        return decode_image_bytes(raw_bytes)
    except Exception as e:
        raise ValueError(f"Invalid base64 image data: {str(e)}")

def preprocess_for_model(img: Image.Image) -> torch.Tensor:
    """Preprocess PIL image into a normalized 4D PyTorch tensor (1, C, H, W)."""
    rgb_img = img.convert("RGB")
    tensor = eval_transform(rgb_img)
    return tensor.unsqueeze(0)

def get_preprocessed_preview(img: Image.Image) -> Image.Image:
    """Return a denormalized 224x224 preview of the exact tensor content sent to the CNN."""
    tensor = eval_transform(img.convert("RGB"))
    mean = torch.tensor(NORM_MEAN).view(3, 1, 1)
    std = torch.tensor(NORM_STD).view(3, 1, 1)
    denormalized = tensor * std + mean
    pixels = (denormalized.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")
