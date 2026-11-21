import React from "react";
import MedicalDisclaimer from "../components/MedicalDisclaimer";

export default function Home({ setActiveTab }) {
  return (
    <div>
      <MedicalDisclaimer />

      {/* Hero Section */}
      <section
        style={{
          background: "linear-gradient(135deg, #0284c7 0%, #0369a1 100%)",
          color: "white",
          padding: "48px 36px",
          borderRadius: "var(--radius)",
          marginBottom: "32px",
          boxShadow: "var(--shadow-md)",
        }}
      >
        <div style={{ maxWidth: "800px" }}>
          <div
            style={{
              display: "inline-block",
              background: "rgba(255, 255, 255, 0.2)",
              padding: "4px 12px",
              borderRadius: "20px",
              fontSize: "12px",
              fontWeight: 700,
              textTransform: "uppercase",
              letterSpacing: "0.8px",
              marginBottom: "14px",
            }}
          >
            Research Prototype • Computer Vision & Augmented Reality
          </div>
          <h1 style={{ fontSize: "32px", fontWeight: 800, lineHeight: 1.25, marginBottom: "16px" }}>
            Augmented Reality for Early Skin Disease Assessment Using Computer Vision
          </h1>
          <p style={{ fontSize: "16px", opacity: 0.95, lineHeight: 1.6, marginBottom: "28px" }}>
            Combining a ResNet18 image classifier with an experimental OpenCV visual aid and browser-based camera overlays
            for an educational research prototype. It is not a diagnostic or clinical screening tool.
          </p>
          <div style={{ display: "flex", gap: "14px", flexWrap: "wrap" }}>
            <button
              className="btn"
              style={{ background: "#ffffff", color: "#0369a1", fontWeight: 700 }}
              onClick={() => setActiveTab("assessment")}
            >
              🔬 Start Skin Assessment
            </button>
            <button
              className="btn"
              style={{ background: "rgba(255, 255, 255, 0.2)", color: "#ffffff", border: "1px solid rgba(255,255,255,0.4)", fontWeight: 700 }}
              onClick={() => setActiveTab("ar")}
            >
              👓 Launch AR Camera HUD
            </button>
            <button
              className="btn"
              style={{ background: "transparent", color: "#ffffff", textDecoration: "underline" }}
              onClick={() => setActiveTab("model")}
            >
              View Model Metrics →
            </button>
          </div>
        </div>
      </section>

      {/* Technology Overview Cards */}
      <h2 style={{ fontSize: "22px", fontWeight: 700, marginBottom: "16px", color: "var(--text-main)" }}>
        Technology Architecture & Clinical Workflow
      </h2>

      <div className="grid-3" style={{ marginBottom: "32px" }}>
        <div className="card">
          <div style={{ fontSize: "32px", marginBottom: "12px" }}>🧠</div>
          <h3 style={{ fontSize: "17px", fontWeight: 700, marginBottom: "8px", color: "var(--primary)" }}>
            How Computer Vision Works
          </h3>
          <p style={{ fontSize: "14px", color: "var(--text-muted)", lineHeight: 1.5 }}>
            Images are preprocessed into standardized 224×224 RGB tensors and passed through a ResNet18 convolutional
            classifier trained on the project's two image categories. Its model output is not a clinical diagnosis.
          </p>
        </div>

        <div className="card">
          <div style={{ fontSize: "32px", marginBottom: "12px" }}>🎯</div>
          <h3 style={{ fontSize: "17px", fontWeight: 700, marginBottom: "8px", color: "var(--secondary)" }}>
            How Augmented Reality Works
          </h3>
          <p style={{ fontSize: "14px", color: "var(--text-muted)", lineHeight: 1.5 }}>
            The browser can display a device camera preview with a simple AR-style HTML canvas overlay. OpenCV processing
            runs on a captured image and produces an approximate visual aid; it does not provide validated lesion segmentation.
          </p>
        </div>

        <div className="card">
          <div style={{ fontSize: "32px", marginBottom: "12px" }}>🛡️</div>
          <h3 style={{ fontSize: "17px", fontWeight: 700, marginBottom: "8px", color: "#d97706" }}>
            Educational Model Output
          </h3>
          <p style={{ fontSize: "14px", color: "var(--text-muted)", lineHeight: 1.5 }}>
            Displays the model's category scores and educational information. It cannot determine whether a lesion is benign,
            malignant, or safe; consult a qualified dermatologist for professional assessment.
          </p>
        </div>
      </div>

      {/* Feature Highlights Grid */}
      <div className="card">
        <h3 className="card-title">Key Capabilities</h3>
        <div className="grid-2">
          <div style={{ display: "flex", gap: "12px", alignItems: "flex-start" }}>
            <span style={{ fontSize: "20px" }}>📷</span>
            <div>
              <strong style={{ display: "block", fontSize: "15px" }}>Multi-Input Flexibility</strong>
              <span style={{ fontSize: "13px", color: "var(--text-muted)" }}>
                Seamlessly toggle between high-resolution dermatoscopic image file uploads and real-time interactive webcam captures.
              </span>
            </div>
          </div>
          <div style={{ display: "flex", gap: "12px", alignItems: "flex-start" }}>
            <span style={{ fontSize: "20px" }}>📐</span>
            <div>
              <strong style={{ display: "block", fontSize: "15px" }}>Approximate Visual Overlay</strong>
              <span style={{ fontSize: "13px", color: "var(--text-muted)" }}>
                Calculates lesion area ratio, asymmetry index, compactness, and bounding boxes using deterministic OpenCV algorithms.
              </span>
            </div>
          </div>
          <div style={{ display: "flex", gap: "12px", alignItems: "flex-start" }}>
            <span style={{ fontSize: "20px" }}>📱</span>
            <div>
              <strong style={{ display: "block", fontSize: "15px" }}>Universal Browser AR Compatibility</strong>
              <span style={{ fontSize: "13px", color: "var(--text-muted)" }}>
                Engineered to operate seamlessly across both WebXR-supported hardware and standard mobile/desktop browsers via HTML5 Canvas overlays.
              </span>
            </div>
          </div>
          <div style={{ display: "flex", gap: "12px", alignItems: "flex-start" }}>
            <span style={{ fontSize: "20px" }}>⚖️</span>
            <div>
              <strong style={{ display: "block", fontSize: "15px" }}>Rigorous Stratification & Balancing</strong>
              <span style={{ fontSize: "13px", color: "var(--text-muted)" }}>
                Cleaned dataset with deduplication, stratified 70/15/15 split, and controlled clinical minority augmentation.
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
