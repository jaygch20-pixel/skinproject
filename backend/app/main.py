"""
Main FastAPI Application Entrypoint
Augmented Reality for Early Skin Disease Assessment Using Computer Vision
Serves API endpoints, generated analytical charts, and the interactive React frontend.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.routes.api import router as api_router
from app.model import SkinModelManager

# Paths (relative to skinproject root where uvicorn is started)
FRONTEND_DIST = os.path.join("frontend", "dist")
FRONTEND_INDEX = os.path.join(FRONTEND_DIST, "index.html")
RESULTS_DIR = "results"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize and load model
    print("[Server Startup] Initializing SkinModelManager...")
    manager = SkinModelManager.get_instance()
    print(f"[Server Startup] Model loaded: {manager.is_loaded} on device: {manager.device}")
    yield
    # Shutdown
    print("[Server Shutdown] Cleaning up resources...")

app = FastAPI(
    title="Augmented Reality for Early Skin Disease Assessment Using Computer Vision",
    description="Medical CV and interactive Augmented Reality backend for skin disease assessment.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend dev-server integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. API Routes (highest priority)
app.include_router(api_router)

# 2. Mount results charts under /static/results
if os.path.exists(RESULTS_DIR):
    app.mount("/static/results", StaticFiles(directory=RESULTS_DIR), name="results-static")

# 3. Mount compiled React frontend assets (CSS/JS bundles)
frontend_assets = os.path.join(FRONTEND_DIST, "assets")
if os.path.exists(frontend_assets):
    app.mount("/assets", StaticFiles(directory=frontend_assets), name="frontend-assets")

# 4. Serve React SPA index.html at root
@app.get("/", include_in_schema=False)
def serve_root():
    if os.path.exists(FRONTEND_INDEX):
        return FileResponse(FRONTEND_INDEX, media_type="text/html")
    return {
        "project": "Augmented Reality for Early Skin Disease Assessment Using Computer Vision",
        "status": "online",
        "docs": "/docs",
        "api_endpoints": {
            "health": "/api/health",
            "model_info": "/api/model-info",
            "predict": "/api/predict",
            "metrics": "/api/results/metrics"
        }
    }

# 5. Catch-all SPA fallback for client-side React routes (must be LAST)
@app.get("/{full_path:path}", include_in_schema=False)
def serve_frontend_spa(full_path: str):
    # Let API, docs, openapi routes 404 through FastAPI
    if full_path.startswith(("api/", "docs", "redoc", "openapi.json", "static/")):
        raise HTTPException(status_code=404, detail="API endpoint not found.")
    # Serve SPA for all other routes (client-side navigation)
    if os.path.exists(FRONTEND_INDEX):
        return FileResponse(FRONTEND_INDEX, media_type="text/html")
    return {"message": "Frontend not found. Build the React app first."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
