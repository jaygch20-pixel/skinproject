"""
Prediction and Assessment Pipeline
Coordinates model inference, confidence calculation, computer vision
lesion localization, and generation of AR assessment metadata.
"""

from typing import Dict, Any, Union
from PIL import Image
import torch
import torch.nn as nn

from app.model import SkinModelManager
from app.preprocessing import decode_image_bytes, decode_base64_image, preprocess_for_model, get_preprocessed_preview
from app.utils.cv_ar import analyze_skin_and_lesion, pil_to_png_data_url, pil_to_jpeg_data_url

MEDICAL_DESCRIPTIONS = {
    "benign": {
        "title": "Model-assigned category: benign",
        "description": "The model assigned this image to the benign category in its training dataset. This result is not a diagnosis and cannot rule out skin cancer.",
        "risk_level": "Not clinically assessed",
        "recommendation": "Do not use this result to decide whether a lesion is safe. Consult a qualified dermatologist about any new, changing, bleeding, or concerning spot."
    },
    "malignant": {
        "title": "Model-assigned category: malignant",
        "description": "The model assigned this image to the malignant category in its training dataset. This result does not confirm melanoma or any other diagnosis.",
        "risk_level": "Not clinically assessed",
        "recommendation": "Please arrange prompt review with a qualified dermatologist. Do not delay professional assessment or make treatment decisions based on this model output."
    }
}

class ModelUnavailableError(RuntimeError):
    """Raised when inference is requested without a trained checkpoint."""

class PredictionService:
    def __init__(self):
        self.model_manager = SkinModelManager.get_instance()
        self.softmax = nn.Softmax(dim=1)
        
    def predict(self, image_input: Union[bytes, str, Image.Image]) -> Dict[str, Any]:
        """
        Accepts raw bytes, base64 string, or PIL Image.
        Executes model inference and CV/AR processing.
        """
        # 1. Resolve to PIL Image
        if isinstance(image_input, bytes):
            pil_img = decode_image_bytes(image_input)
        elif isinstance(image_input, str):
            pil_img = decode_base64_image(image_input)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
            
        # 2. Preprocess for deep learning model
        tensor = preprocess_for_model(pil_img).to(self.model_manager.device)
        
        # 3. Model Forward Pass
        if not self.model_manager.is_loaded:
            self.model_manager.load_model()
        if not self.model_manager.is_loaded or self.model_manager.model is None:
            raise ModelUnavailableError("No trained model checkpoint is available; prediction is disabled.")
            
        model = self.model_manager.model
        with torch.no_grad():
            outputs = model(tensor)
            probs = self.softmax(outputs).squeeze(0).cpu().numpy()
            
        class_names = self.model_manager.class_names
        
        # Build probability map
        probabilities = {}
        for idx, name in enumerate(class_names):
            probabilities[name] = round(float(probs[idx]), 4)
            
        top_idx = int(torch.argmax(outputs, dim=1).item())
        predicted_class = class_names[top_idx]
        confidence = float(probs[top_idx])
        
        # 4. Computer Vision Skin and Lesion AR Analysis
        cv_result = analyze_skin_and_lesion(
            pil_image=pil_img,
            predicted_class=predicted_class,
            confidence=confidence
        )
        preprocessed_preview = get_preprocessed_preview(pil_img)
        segmentation_visualizations = {
            "original_image": pil_to_png_data_url(pil_img),
            "preprocessed_cnn_input": pil_to_png_data_url(preprocessed_preview),
            "binary_mask": cv_result["binary_mask_data_url"],
            "boundary_overlay": cv_result["boundary_overlay_data_url"],
            "segmented_image": cv_result["segmented_image_data_url"]
        }
        
        desc_info = MEDICAL_DESCRIPTIONS.get(predicted_class, {
            "title": f"Category: {predicted_class.capitalize()}",
            "description": f"Model classified image as {predicted_class}.",
            "risk_level": "Evaluation Required",
            "recommendation": "Consult a physician for clinical assessment."
        })
        
        return {
            "status": "success",
            "prediction_status": "completed",
            "predicted_category": predicted_class,
            "confidence_score": round(confidence, 4),
            "confidence_percentage": round(confidence * 100, 2),
            "class_probabilities": probabilities,
            "assessment_summary": {
                "category_title": desc_info["title"],
                "clinical_description": desc_info["description"],
                "risk_profile": desc_info["risk_level"],
                "recommendation": desc_info["recommendation"]
            },
            "ar_cv_overlay": cv_result,
            "segmentation_visualizations": segmentation_visualizations,
            "medical_safety_disclaimer": (
                "IMPORTANT MEDICAL NOTICE: This system is an academic research and engineering prototype "
                "developed for educational and technological demonstration purposes. It is NOT an FDA/CE cleared "
                "medical diagnostic tool. Model confidence represents statistical pattern matching, NOT diagnostic certainty. "
                "Never rely solely on automated algorithms for health decisions. Always seek immediate guidance from a "
                "qualified, board-certified dermatologist."
            )
        }
