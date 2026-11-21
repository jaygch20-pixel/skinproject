import React from "react";
import MedicalDisclaimer from "../components/MedicalDisclaimer";

export default function AboutProject() {
  return (
    <div>
      <MedicalDisclaimer />

      {/* Hero Header */}
      <div className="card">
        <h2 className="card-title">ℹ️ About The Project</h2>
        <h3 style={{ fontSize: "20px", color: "var(--primary-hover)", marginBottom: "12px" }}>
          Augmented Reality for Early Skin Disease Assessment Using Computer Vision
        </h3>
        <p style={{ fontSize: "15px", color: "var(--text-main)", lineHeight: 1.7, marginBottom: "20px" }}>
          This academic prototype explores image classification and browser camera overlays for skin-image datasets. It is not a
          screening tool, does not diagnose skin conditions, and should not be used to assess a lesion or decide whether to seek care.
        </p>

        <div className="grid-3">
          <div style={{ background: "#f8fafc", padding: "16px", borderRadius: "8px", border: "1px solid var(--border)" }}>
            <strong style={{ color: "var(--primary)" }}>Objective 1: Model Evaluation</strong>
            <p style={{ fontSize: "13px", color: "var(--text-muted)", marginTop: "4px" }}>
              Explore two-category image classification and report measured results on a held-out dataset.
            </p>
          </div>
          <div style={{ background: "#f8fafc", padding: "16px", borderRadius: "8px", border: "1px solid var(--border)" }}>
            <strong style={{ color: "var(--secondary)" }}>Objective 2: Interactive AR</strong>
            <p style={{ fontSize: "13px", color: "var(--text-muted)", marginTop: "4px" }}>
              Display an educational camera overlay without requiring a native app; model output is not a diagnosis.
            </p>
          </div>
          <div style={{ background: "#f8fafc", padding: "16px", borderRadius: "8px", border: "1px solid var(--border)" }}>
            <strong style={{ color: "#d97706" }}>Objective 3: Public Education</strong>
            <p style={{ fontSize: "13px", color: "var(--text-muted)", marginTop: "4px" }}>
              Provide general educational material and encourage professional dermatological assessment when appropriate.
            </p>
          </div>
        </div>
      </div>

      {/* Methodology Section */}
      <div className="card">
        <h3 className="card-title">🔬 Methodological Details</h3>

        <div style={{ marginBottom: "20px" }}>
          <h4 style={{ fontSize: "16px", color: "var(--text-main)", marginBottom: "6px" }}>
            1. Computer Vision Pipeline (PyTorch & OpenCV)
          </h4>
          <p style={{ fontSize: "14px", color: "var(--text-muted)", lineHeight: 1.6 }}>
            The image classifier uses a customized <strong>ResNet18</strong> architecture. Input images undergo RGB conversion and resizing
            to 224×224 pixels, followed by PyTorch normalization. OpenCV applies heuristic color and intensity thresholding to produce an
            optional approximate contour. That contour is not validated lesion segmentation, and its geometry is not a clinical measurement.
          </p>
        </div>

        <div style={{ marginBottom: "20px" }}>
          <h4 style={{ fontSize: "16px", color: "var(--text-main)", marginBottom: "6px" }}>
            2. Augmented Reality Engine (WebXR & HTML5 Canvas Overlay)
          </h4>
          <p style={{ fontSize: "14px", color: "var(--text-muted)", lineHeight: 1.6 }}>
            The browser can capture webcam and mobile camera frames via <code>getUserMedia</code>. An HTML5 canvas renders an AR-style
            overlay; image analysis is performed after capture and displayed on the still image. Camera availability depends on browser
            permissions and a secure context (HTTPS or localhost).
          </p>
        </div>

        <div>
          <h4 style={{ fontSize: "16px", color: "var(--text-main)", marginBottom: "6px" }}>
            3. Data Hygiene & Class Balancing
          </h4>
          <p style={{ fontSize: "14px", color: "var(--text-muted)", lineHeight: 1.6 }}>
            The preprocessing pipeline removes exact duplicates and high-confidence perceptual matches before making class-wise 70%/15%/15%
            splits. Augmentation is applied only to training images; validation and test images are kept unchanged.
          </p>
        </div>
      </div>

      {/* Technology Stack Grid */}
      <div className="card">
        <h3 className="card-title">💻 Complete Technology Stack</h3>
        <div className="grid-2">
          <div>
            <h4 style={{ fontSize: "14px", fontWeight: 700, marginBottom: "8px", color: "var(--primary)" }}>
              Backend & Machine Learning
            </h4>
            <ul style={{ paddingLeft: "18px", fontSize: "13px", color: "var(--text-main)", lineHeight: 1.8 }}>
              <li><strong>Python 3.13:</strong> High-performance core backend runtime.</li>
              <li><strong>PyTorch 2.14:</strong> Deep neural network modeling and ResNet18 training.</li>
              <li><strong>Torchvision:</strong> Image transformations, data loaders, and tensor normalization.</li>
              <li><strong>OpenCV (cv2):</strong> Color segmentation, morphological filtering, and contour extraction.</li>
              <li><strong>FastAPI & Uvicorn:</strong> Asynchronous high-throughput REST API.</li>
              <li><strong>Scikit-Learn:</strong> Stratified splitting, confusion matrix, ROC-AUC, and KPI computation.</li>
            </ul>
          </div>

          <div>
            <h4 style={{ fontSize: "14px", fontWeight: 700, marginBottom: "8px", color: "var(--secondary)" }}>
              Frontend & Augmented Reality
            </h4>
            <ul style={{ paddingLeft: "18px", fontSize: "13px", color: "var(--text-main)", lineHeight: 1.8 }}>
              <li><strong>React.js & Vite:</strong> Lightning-fast reactive component architecture.</li>
              <li><strong>HTML5 Canvas & WebRTC:</strong> Browser camera preview and still-image overlays.</li>
              <li><strong>Vanilla CSS3:</strong> Polished healthcare design system with zero bloated dependencies.</li>
              <li><strong>Axios:</strong> Multipart file and base64 asynchronous REST communication.</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
