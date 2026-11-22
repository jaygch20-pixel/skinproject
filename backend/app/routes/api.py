"""
API Routes for Skin Disease Assessment & AR Computer Vision Service
Provides endpoints for health checking, model metadata, image predictions, and evaluation metrics.
"""

import json
import logging
import os
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError
from starlette.datastructures import UploadFile

from app.model import SkinModelManager
from app.prediction import ModelUnavailableError, PredictionService
from app.preprocessing import MAX_FILE_SIZE_BYTES

router = APIRouter(prefix="/api", tags=["Skin Disease CV & AR"])
prediction_service = PredictionService()
logger = logging.getLogger(__name__)

class PredictBase64Request(BaseModel):
    image: str = Field(..., min_length=1, description="Base64 encoded skin image or data URL from camera")
    include_visualizations: bool = True

@router.get("/health")
def get_health():
    """Health check endpoint providing service & model state."""
    manager = SkinModelManager.get_instance()
    return {
        "status": "healthy",
        "service": "Augmented Reality Skin Disease Assessment API",
        "version": "1.0.0",
        "model_loaded": manager.is_loaded,
        "device": str(manager.device)
    }

@router.get("/model-info")
def get_model_info():
    """Retrieves deep learning model architecture, training configuration, and limitations."""
    manager = SkinModelManager.get_instance()
    info = manager.get_info()
    
    # Check if evaluation report exists
    eval_path = os.path.join("results", "model_evaluation_metrics.json")
    if not os.path.exists(eval_path):
        eval_path = os.path.join("backend", "results", "model_evaluation_metrics.json")
        
    eval_metrics = None
    if os.path.exists(eval_path):
        import json
        with open(eval_path, "r") as f:
            eval_metrics = json.load(f)
            
    return {
        **info,
        "evaluation_metrics": eval_metrics,
        "limitations": [
            "Trained for educational and research prototyping only.",
            "Not certified for clinical or diagnostic medical use.",
            "Requires adequate dermatological lighting and clear focus without heavy glare.",
            "Visual features are classified statistically; final diagnosis requires histology and biopsy by a dermatologist."
        ]
    }

@router.post(
    "/predict",
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "file": {"type": "string", "format": "binary"},
                            "include_visualizations": {"type": "boolean", "default": True},
                        },
                        "required": ["file"],
                    }
                },
                "application/json": {"schema": PredictBase64Request.model_json_schema()},
            }
        }
    },
)
async def predict_image(request: Request):
    """
    Accepts an uploaded image file or base64 frame from webcam/mobile camera.
    Returns predicted disease category, confidence, probabilities, and AR CV overlay metadata.
    """
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    include_visualizations = True
    try:
        if content_type == "multipart/form-data":
            form = await request.form()
            include_visualizations = str(form.get("include_visualizations", "true")).lower() == "true"
            upload = form.get("file")
            if not isinstance(upload, UploadFile):
                raise HTTPException(status_code=400, detail="Multipart request must include an image file in the 'file' field.")
            image_input = await upload.read(MAX_FILE_SIZE_BYTES + 1)
            if not image_input:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            if len(image_input) > MAX_FILE_SIZE_BYTES:
                raise HTTPException(status_code=413, detail="Image file exceeds the 15 MB upload limit.")
        elif content_type == "application/json":
            try:
                body = await request.json()
            except json.JSONDecodeError as error:
                raise HTTPException(status_code=400, detail="Request body must contain valid JSON.") from error
            try:
                payload = PredictBase64Request.model_validate(body)
            except ValidationError as error:
                raise HTTPException(status_code=422, detail=error.errors()) from error
            image_input = payload.image
            include_visualizations = payload.include_visualizations
        else:
            raise HTTPException(
                status_code=415,
                detail="Use multipart/form-data with a 'file' field or application/json with an 'image' field.",
            )

        result = prediction_service.predict(
            image_input,
            include_visualizations=include_visualizations
        )
        return JSONResponse(status_code=200, content=result)
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except ModelUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        logger.exception("Prediction request failed.")
        raise HTTPException(status_code=500, detail="Prediction failed due to an internal server error.") from error

@router.get("/results/metrics")
def get_evaluation_metrics():
    """Returns the latest saved evaluation metrics and test report."""
    eval_path = os.path.join("results", "model_evaluation_metrics.json")
    if not os.path.exists(eval_path):
        eval_path = os.path.join("backend", "results", "model_evaluation_metrics.json")
    if os.path.exists(eval_path):
        import json
        with open(eval_path, "r") as f:
            return json.load(f)
    return {"status": "pending", "message": "Evaluation in progress."}

@router.get("/results/charts/{chart_name}")
def get_chart(chart_name: str):
    """Serves generated visualization charts (confusion matrix, ROC, loss/accuracy curves)."""
    # Sanitize chart_name
    safe_name = os.path.basename(chart_name)
    candidates = [
        os.path.join("results", safe_name),
        os.path.join("backend", "results", safe_name)
    ]
    for p in candidates:
        if os.path.exists(p):
            return FileResponse(p, media_type="image/png")
    raise HTTPException(status_code=404, detail=f"Chart '{safe_name}' not found.")
