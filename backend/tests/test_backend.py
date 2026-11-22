"""
End-to-End Test Suite for Skin Disease Assessment & AR Backend
"""

import os
import sys
import io
import json
import base64
import pytest
from PIL import Image
from PIL import ImageDraw
import numpy as np

# Add backend to path
sys.path.insert(0, os.path.abspath("backend"))
sys.path.insert(0, os.path.abspath("backend/training"))

from fastapi.testclient import TestClient
from app.main import app
from app.model import SkinModelManager
from app.prediction import ModelUnavailableError, PredictionService
from app.utils.cv_ar import analyze_skin_and_lesion
from image_deduplication import detect_near_duplicate_pairs

client = TestClient(app)

def test_near_duplicate_detection(tmp_path):
    source = Image.new("RGB", (128, 128), (185, 140, 125))
    draw = ImageDraw.Draw(source)
    draw.ellipse((35, 40, 95, 92), fill=(65, 38, 29))
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    source.save(first, quality=95)
    source.save(second, quality=88)

    pairs = detect_near_duplicate_pairs([str(first), str(second)])
    assert len(pairs) == 1
    assert pairs[0]["ssim"] >= 0.98

def test_cv_candidates_keep_multiple_interior_regions_separate():
    image = Image.new("RGB", (320, 240), (190, 145, 130))
    draw = ImageDraw.Draw(image)
    draw.ellipse((45, 50, 105, 110), fill=(65, 40, 30))
    draw.ellipse((205, 125, 265, 185), fill=(65, 40, 30))

    result = analyze_skin_and_lesion(image)

    assert result["detected"] is True
    assert len(result["contours"]) >= 2
    assert result["localization_supported"] is False
    assert result["bounding_box"]["x"] > 1
    assert result["bounding_box"]["x"] + result["bounding_box"]["width"] < image.width - 1
    for data_url in (
        result["binary_mask_data_url"],
        result["boundary_overlay_data_url"],
        result["segmented_image_data_url"]
    ):
        encoded_image = data_url.split(",", 1)[1]
        rendered = Image.open(io.BytesIO(base64.b64decode(encoded_image)))
        assert rendered.size == image.size

def test_cv_candidates_do_not_mark_uniform_skin():
    result = analyze_skin_and_lesion(Image.new("RGB", (320, 240), (190, 145, 130)))

    assert result["detected"] is False
    assert result["contours"] == []
    assert result["localization_supported"] is False

def test_prediction_fails_when_trained_checkpoint_is_unavailable(monkeypatch):
    class UnavailableManager:
        is_loaded = False
        model = None
        device = "cpu"

        def load_model(self):
            pass

    monkeypatch.setattr(SkinModelManager, "get_instance", lambda: UnavailableManager())
    service = PredictionService()

    with pytest.raises(ModelUnavailableError):
        service.predict(Image.new("RGB", (224, 224), color=(200, 150, 130)))

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    print("[PASS] GET /api/health verified:", data)

def test_model_info_endpoint():
    response = client.get("/api/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["architecture"] == "ResNet18 (Residual Neural Network)"
    assert data["num_classes"] == 2
    assert "benign" in data["classes"]
    assert "malignant" in data["classes"]
    print("[PASS] GET /api/model-info verified:", data["architecture"], "Classes:", data["classes"])

def test_predict_documents_both_input_formats():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    content = response.json()["paths"]["/api/predict"]["post"]["requestBody"]["content"]
    assert "multipart/form-data" in content
    assert "application/json" in content

def test_predict_with_file_upload():
    # Create test skin patch image
    img = Image.new("RGB", (224, 224), color=(220, 180, 160))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    
    response = client.post(
        "/api/predict",
        files={"file": ("test_lesion.jpg", buf, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["predicted_category"] in ["benign", "malignant", "uncertain"]
    assert 0.0 <= data["confidence_score"] <= 1.0
    assert "class_probabilities" in data
    assert "ar_cv_overlay" in data
    assert "bounding_box" in data["ar_cv_overlay"]
    assert "ar_overlay_data_url" in data["ar_cv_overlay"]
    visualizations = data["segmentation_visualizations"]
    assert set(visualizations) == {
        "original_image",
        "preprocessed_cnn_input",
        "binary_mask",
        "boundary_overlay",
        "segmented_image"
    }
    assert all(value.startswith("data:image/") for value in visualizations.values())
    assert "medical_safety_disclaimer" in data
    print(f"[PASS] POST /api/predict (File Upload) verified: Predicted {data['predicted_category']} with {data['confidence_percentage']}% confidence.")

def test_predict_with_base64():
    import base64
    img = Image.new("RGB", (224, 224), color=(200, 150, 130))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64_str = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
    
    response = client.post(
        "/api/predict",
        json={"image": b64_str}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    print(f"[PASS] POST /api/predict (Base64 Camera Frame) verified: Predicted {data['predicted_category']} ({data['confidence_percentage']}%).")

def test_predict_can_skip_visualization_payload_for_live_camera():
    import base64
    img = Image.new("RGB", (320, 180), color=(200, 150, 130))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64_str = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

    response = client.post(
        "/api/predict",
        json={"image": b64_str, "include_visualizations": False}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["segmentation_visualizations"] is None
    assert data["ar_cv_overlay"]["localization_supported"] is False
    assert data["ar_cv_overlay"]["detected"] is False

def test_plain_image_returns_uncertain():
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/api/predict",
        files={"file": ("plain.jpg", buf, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["uncertain"] is True
    assert data["predicted_category"] == "uncertain"
    assert data["ar_cv_overlay"]["detected"] is False
    assert data["uncertainty_reason"]

def test_dataset_benign_and_malignant_samples_return_real_predictions():
    cases = [
        ("data/balanced/test/benign/benign_00000.jpg", "benign"),
        ("data/balanced/test/benign/benign_00001.jpg", "benign"),
        ("data/balanced/test/malignant/malignant_00000.jpg", "malignant"),
        ("data/balanced/test/malignant/malignant_00001.jpg", "malignant")
    ]
    for path, _label in cases:
        if not os.path.exists(path):
            continue
        with open(path, "rb") as image_file:
            response = client.post(
                "/api/predict",
                files={"file": (os.path.basename(path), image_file, "image/jpeg")}
            )
        assert response.status_code == 200
        data = response.json()
        assert data["raw_predicted_category"] in ["benign", "malignant"]
        assert data["confidence_score"] == round(data["confidence_score"], 4)

def test_predict_validation_error():
    # Empty file
    response = client.post("/api/predict", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert response.status_code in [400, 422]
    print("[PASS] POST /api/predict validation error handling confirmed.")

def test_chart_endpoint():
    response = client.get("/api/results/charts/confusion_matrix.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    print("[PASS] GET /api/results/charts/confusion_matrix.png verified.")

if __name__ == "__main__":
    print("=== Running DermaVision AR Test Suite ===")
    test_health_endpoint()
    test_model_info_endpoint()
    test_predict_with_file_upload()
    test_predict_with_base64()
    test_predict_validation_error()
    test_chart_endpoint()
    print("=== All Tests Passed Successfully! ===")
