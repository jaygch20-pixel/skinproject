# Augmented Reality for Early Skin Disease Assessment Using Computer Vision

**DermaVision AR** is an academic research prototype combining a deep residual convolutional neural network (**ResNet18**), experimental image processing (**OpenCV**), and a browser-based camera overlay. It is not a clinical screening or diagnostic tool.

---

## ⚠️ Important Medical Safety & Ethical Disclaimer

> **ACADEMIC & RESEARCH PROTOTYPE ONLY**  
> This software is an engineering demonstration and academic research prototype. It is **NOT** a certified medical diagnostic device and has **NOT** been evaluated or cleared by the FDA, CE, or any health authority.  
> 
> - **Statistical Confidence vs. Diagnostic Certainty:** Model confidence scores represent statistical pattern matching against training data, **not** clinical diagnostic certainty.
> - **No Medical Advice:** Do not use this tool to make treatment decisions or delay professional healthcare.
> - **Dermatologist Consultation:** Any suspicious skin lesion, irregular mole, or cutaneous change must be promptly evaluated in person by a qualified, board-certified dermatologist with dermoscopy and histological biopsy.

---

## 🌟 Key Features

1. **Automated Dataset Inspection & Hygiene:**
   - Automatic discovery of raw dataset partitions.
   - SHA-256 exact deduplication and conservative perceptual-hash/SSIM near-duplicate checks before splitting.
   - Complete integrity checks for unreadable/corrupted files.
   - Standardized 224×224 RGB formatting.

2. **Rigorous Stratified Splitting & Class Balancing:**
   - 70% Training / 15% Validation / 15% Test stratified partition.
   - Balanced training set using restrained image augmentations (subtle rotation, horizontal flip, brightness adjustment).
   - Validation and test splits preserved 100% untouched.

3. **Deep Residual Neural Network (ResNet18):**
   - 18 convolutional layers with identity shortcut connections to learn subtle dermatoscopic patterns.
   - Robust Kaiming He normal weight initialization for offline standalone execution.
   - Trained with Adam optimizer and StepLR scheduling.
   - Full evaluation metrics: Accuracy, Sensitivity (Recall), Specificity, Precision, F1-Score, Confusion Matrix, and ROC-AUC curves.

4. **Experimental Computer Vision (OpenCV):**
   - Heuristic image processing using color thresholds and Otsu thresholding.
   - Optional approximate contour and center-zone overlays on an analyzed image.
   - Overlay geometry is not validated lesion segmentation or a clinical measurement.

5. **Browser Camera Overlay:**
   - Webcam and mobile camera preview with capture controls.
   - AR-style HTML5 Canvas overlay with a static-image fallback.
   - Approximate computer-vision results are shown on the captured image, not presented as live lesion tracking.
   - Camera access uses browser permissions and requires HTTPS or localhost.

6. **Full-Stack Web Architecture:**
   - **Backend:** FastAPI with modular routers, cached model singleton, base64 and multipart file support, and Swagger/OpenAPI documentation.
   - **Frontend:** Responsive React application with Vite, Axios API integration, and a healthcare blue/slate aesthetic.

---

## 📁 Project Directory Structure

```
skinproject/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application entrypoint & static mounting
│   │   ├── model.py             # ResNet18 model manager and cached singleton
│   │   ├── prediction.py        # Prediction pipeline and non-diagnostic guidance
│   │   ├── preprocessing.py     # Image decoding, validation, and PyTorch normalization
│   │   ├── routes/
│   │   │   └── api.py           # REST endpoints (/health, /model-info, /predict, /charts)
│   │   └── utils/
│   │       └── cv_ar.py         # Approximate geometry for captured images
│   ├── models/
│   │   ├── best_resnet18.pth    # Best performing model checkpoint
│   │   ├── class_labels.json    # Class label mappings
│   │   └── preprocessing_config.json # Hyperparameter configuration
│   ├── results/                 # Evaluation charts & metrics JSON
│   ├── training/
│   │   ├── dataset_analysis.py  # Dataset inspection, deduplication, and visual reporting
│   │   ├── image_deduplication.py # Exact and near-duplicate detection
│   │   ├── clean_and_split.py   # Stratified 70/15/15 splitting & 224x224 RGB conversion
│   │   ├── balance_dataset.py   # Minority class augmentation for training split
│   │   ├── train.py             # ResNet18 PyTorch training loop
│   │   └── evaluate.py          # Comprehensive test set evaluation & ROC generation
│   └── requirements.txt         # Python dependencies
├── archive/archive/
│   └── melanoma_cancer_dataset/ # Original dataset (never modified)
├── data/
│   ├── cleaned/                 # Cleaned, standardized 224x224 train/val/test splits
│   └── balanced/                # Balanced training set with untouched val/test
├── frontend/
│   ├── dist/
│   │   └── index.html           # Standalone production SPA bundle (offline-ready)
│   ├── src/
│   │   ├── components/          # Navbar, Footer, MedicalDisclaimer
│   │   ├── pages/               # Home, Assessment, ARExperience, Results, ModelInfo, AboutProject
│   │   ├── services/            # API integration client
│   │   ├── styles/              # Healthcare CSS styling
│   │   ├── App.jsx              # Main React component
│   │   └── main.jsx             # React entrypoint
│   └── package.json             # Frontend package configuration
├── results/                     # Persistent publication-quality visual charts
└── README.md                    # Project documentation
```

---

## 📊 Dataset & Model Performance Summary

- **Total Analyzed Raw Images:** 10,605
- **Cleaned Unique Samples:** 10,571 (22 exact duplicates and 12 high-confidence near-duplicate images excluded)
- **Classes:** `benign` (5,489 unique) | `malignant` (5,082 unique)
- **Partitions:**
  - **Train (70%):** 7,399 cleaned images (balanced to 3,842 benign & 3,842 malignant)
  - **Validation (15%):** 1,585 images (823 benign, 762 malignant)
  - **Test (15%):** 1,587 images (824 benign, 763 malignant)
- **Model Checkpoint:** `backend/models/best_resnet18.pth`
  - **Best Validation Accuracy:** 90.16% (five-epoch run; epoch 1 continued from the available checkpoint)
  - **Independent Test Results:** 89.48% accuracy, 88.73% malignant sensitivity, 90.17% specificity, ROC-AUC 0.9650
  - **Training History:** Epochs 2–5 are recorded; epoch 1 metrics were unavailable from the interrupted run and are not reconstructed.

---

## 🚀 Exact Commands to Run the Project in VS Code

### 1. Prerequisites
- Python 3.10+ installed
- Node.js v18+ (optional, for frontend dev server)

### 2. Backend Setup & Startup
Open a terminal in VS Code at the project root (`c:\Users\hp\Desktop\skinproject`):

```bash
# 1. Install required Python packages
python -m pip install -r backend/requirements.txt

# 2. Start the FastAPI backend server
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
```

The backend will start at:
- **Application URL:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive API Documentation (Swagger):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check:** [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

### 3. Accessing the Interactive Application
The full interactive React frontend is served automatically by the backend at:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

You can navigate between:
1. **Home:** Application overview and educational workflow.
2. **Skin Assessment:** Upload a skin photo or capture one with the camera.
3. **AR Experience:** Capture a still image and view approximate computer-vision geometry; this is not live lesion tracking.
4. **Results:** Non-diagnostic statistical classification and explanatory information; consult a dermatologist for medical advice.
5. **Model & Evaluation:** Architecture specifications, confusion matrix, ROC curves, and training/validation loss & accuracy charts.
6. **About Project:** Scientific methodology and technology stack.

---

## 🔄 Re-running Training & Evaluation Pipelines (Optional)

If you wish to re-execute any stage of the pipeline independently:

```bash
# 1. Inspect original dataset & generate analysis visualizations
python backend/training/dataset_analysis.py

# 2. Clean, deduplicate, and split into 70/15/15
python backend/training/clean_and_split.py

# 3. Apply controlled minority class balancing
python backend/training/balance_dataset.py

# 4. Train ResNet18 model for 5 epochs
python backend/training/train.py --batch-size 64 --epochs 5

# 5. Evaluate model on test set & generate ROC/Confusion Matrix
python backend/training/evaluate.py
```

---

## 🔬 Scientific Methodology

### The ABCDE Criteria for Skin Lesions
The application educates users on the clinical signs of melanoma:
- **A - Asymmetry:** One half does not match the other half.
- **B - Border:** Edges are irregular, notched, scalloped, or blurred.
- **C - Color:** Color variation across the lesion (brown, black, pink, blue, red).
- **D - Diameter:** Lesions larger than 6mm (approx. pencil eraser).
- **E - Evolving:** Any mole changing in size, shape, color, or symptoms (bleeding, itching).

*Developed as an academic contribution to Computer Vision and Augmented Reality in Healthcare.*
