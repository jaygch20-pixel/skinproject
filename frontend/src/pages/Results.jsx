import React from "react";
import MedicalDisclaimer from "../components/MedicalDisclaimer";

export default function Results({ analysisData, setActiveTab }) {
  if (!analysisData || !analysisData.result) {
    return (
      <div>
        <MedicalDisclaimer />
        <div className="card" style={{ textAlign: "center", padding: "48px 20px" }}>
          <div style={{ fontSize: "48px", marginBottom: "16px" }}>📋</div>
          <h2 style={{ fontSize: "20px", fontWeight: 700, marginBottom: "8px" }}>
            No Assessment Result Available Yet
          </h2>
          <p style={{ color: "var(--text-muted)", marginBottom: "24px" }}>
            Please perform a skin lesion assessment or launch the AR camera to inspect a skin region.
          </p>
          <div style={{ display: "flex", justifyContent: "center", gap: "12px" }}>
            <button className="btn btn-primary" onClick={() => setActiveTab("assessment")}>
              🔬 Go to Skin Assessment
            </button>
            <button className="btn btn-secondary" onClick={() => setActiveTab("ar")}>
              👓 Launch AR Camera HUD
            </button>
          </div>
        </div>
      </div>
    );
  }

  const { result, originalImage } = analysisData;
  const isBenign = result.predicted_category === "benign";
  const overlay = result.ar_cv_overlay || {};
  const morphology = overlay.morphology || {};

  return (
    <div>
      <MedicalDisclaimer />

      {/* Header Result Card */}
      <div
        className="card"
        style={{
          borderLeft: `6px solid ${isBenign ? "var(--benign)" : "var(--malignant)"}`,
          background: isBenign ? "#f0fdf4" : "#fef2f2",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "14px" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "8px" }}>
              <span className={`badge ${isBenign ? "badge-benign" : "badge-malignant"}`} style={{ fontSize: "14px", padding: "6px 14px" }}>
                {result.predicted_category.toUpperCase()}
              </span>
              <span style={{ fontSize: "14px", color: "var(--text-muted)", fontWeight: 600 }}>
                Status: {result.prediction_status}
              </span>
            </div>
            <h2 style={{ fontSize: "24px", fontWeight: 800, color: isBenign ? "#15803d" : "#b91c1c", marginBottom: "6px" }}>
              {result.assessment_summary?.category_title || `Model-assigned category: ${result.predicted_category}`}
            </h2>
            <p style={{ fontSize: "14px", color: "var(--text-main)", maxWidth: "700px" }}>
              {result.assessment_summary?.clinical_description}
            </p>
          </div>

          <div
            style={{
              background: "#ffffff",
              padding: "16px 24px",
              borderRadius: "8px",
              border: "1px solid var(--border)",
              textAlign: "center",
              boxShadow: "var(--shadow-sm)",
            }}
          >
            <div style={{ fontSize: "11px", textTransform: "uppercase", color: "var(--text-muted)", fontWeight: 700 }}>
              Model Confidence
            </div>
            <div style={{ fontSize: "32px", fontWeight: 800, color: isBenign ? "var(--benign)" : "var(--malignant)" }}>
              {result.confidence_percentage}%
            </div>
            <div style={{ fontSize: "11px", color: "var(--text-light)" }}>Statistical Match</div>
          </div>
        </div>
      </div>

      {/* Visual Inspection Comparison */}
      <div className="card">
        <h3 className="card-title">🔍 Image & Approximate Visual Aid</h3>
        <div className="grid-2" style={{ marginTop: "16px" }}>
          {/* Original Image */}
          <div>
            <h4 style={{ fontSize: "14px", color: "var(--text-muted)", marginBottom: "8px" }}>
              Uploaded or captured image
            </h4>
            <div style={{ width: "100%", height: "280px", borderRadius: "8px", overflow: "hidden", background: "#0f172a", border: "1px solid var(--border)" }}>
              <img
                src={originalImage}
                alt="Original Sample"
                style={{ width: "100%", height: "100%", objectFit: "contain" }}
              />
            </div>
          </div>

          {/* AR OpenCV Annotated Overlay */}
          <div>
            <h4 style={{ fontSize: "14px", color: "var(--text-muted)", marginBottom: "8px" }}>
              Approximate OpenCV overlay (not validated lesion segmentation)
            </h4>
            <div style={{ width: "100%", height: "280px", borderRadius: "8px", overflow: "hidden", background: "#0f172a", border: "1px solid var(--border)" }}>
              <img
                src={overlay.ar_overlay_data_url || originalImage}
                alt="Approximate OpenCV visual aid"
                style={{ width: "100%", height: "100%", objectFit: "contain" }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Probability Distribution & Morphometrics */}
      <div className="grid-2">
        {/* Class Probabilities */}
        <div className="card">
          <h3 className="card-title">📊 Model Score Distribution</h3>
          <p style={{ fontSize: "13px", color: "var(--text-muted)", marginBottom: "16px" }}>
            Model scores across training categories. These are not diagnostic probabilities:
          </p>
          {Object.entries(result.class_probabilities || {}).map(([cls, prob]) => {
            const isTarget = cls === result.predicted_category;
            const pct = (prob * 100).toFixed(1);
            return (
              <div key={cls} style={{ marginBottom: "14px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "14px", fontWeight: 600 }}>
                  <span style={{ textTransform: "capitalize" }}>{cls}</span>
                  <span style={{ color: isTarget ? (cls === "benign" ? "var(--benign)" : "var(--malignant)") : "var(--text-muted)" }}>
                    {pct}%
                  </span>
                </div>
                <div className="progress-bar-bg">
                  <div
                    className="progress-bar-fill"
                    style={{
                      width: `${pct}%`,
                      background: cls === "benign" ? "var(--benign)" : "var(--malignant)",
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>

        {/* Experimental OpenCV estimates */}
        <div className="card">
          <h3 className="card-title">📐 Approximate OpenCV Visual Aid</h3>
          <p style={{ fontSize: "13px", color: "var(--text-muted)", marginBottom: "16px" }}>
            Experimental image geometry only; this is not validated lesion segmentation or a clinical measurement.
          </p>
          <div style={{ display: "flex", flexDirection: "column", gap: "10px", fontSize: "14px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", borderBottom: "1px solid var(--border)", paddingBottom: "6px" }}>
              <span style={{ color: "var(--text-muted)" }}>Candidate region:</span>
              <strong>{overlay.detected ? "Heuristic candidate region" : "No reliable region detected"}</strong>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", borderBottom: "1px solid var(--border)", paddingBottom: "6px" }}>
              <span style={{ color: "var(--text-muted)" }}>Contour shape heuristic:</span>
              <strong>{morphology.contour_shape_heuristic || "Unavailable"}</strong>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", borderBottom: "1px solid var(--border)", paddingBottom: "6px" }}>
              <span style={{ color: "var(--text-muted)" }}>Bounding-box aspect ratio:</span>
              <strong>{morphology.bounding_box_aspect_ratio ?? "Unavailable"}</strong>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", borderBottom: "1px solid var(--border)", paddingBottom: "6px" }}>
              <span style={{ color: "var(--text-muted)" }}>Circularity Index:</span>
              <strong>{morphology.circularity_index ?? "Unavailable"}</strong>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-muted)" }}>Candidate box maximum dimension:</span>
              <strong>{morphology.candidate_box_max_dimension_px ?? "Unavailable"}</strong>
            </div>
          </div>
        </div>
      </div>

      {/* Educational information and professional assessment */}
      <div className="card" style={{ background: isBenign ? "#f8fafc" : "#fff7ed", border: `1px solid ${isBenign ? "#cbd5e1" : "#fed7aa"}` }}>
        <h3 className="card-title" style={{ color: isBenign ? "var(--primary-hover)" : "#c2410c" }}>
          👨‍⚕️ General Information & Professional Assessment
        </h3>
        <p style={{ fontSize: "14px", marginBottom: "12px", lineHeight: 1.6 }}>
          <strong>Important:</strong> {result.assessment_summary?.recommendation}
        </p>

        <div style={{ background: "#ffffff", padding: "16px", borderRadius: "8px", border: "1px solid var(--border)", marginTop: "12px" }}>
          <h4 style={{ fontSize: "14px", fontWeight: 700, marginBottom: "8px" }}>
            The Dermatological ABCDE Melanoma Rule:
          </h4>
          <ul style={{ paddingLeft: "20px", fontSize: "13px", color: "var(--text-main)", lineHeight: 1.7 }}>
            <li><strong>A - Asymmetry:</strong> One half of the spot does not match the other half.</li>
            <li><strong>B - Border:</strong> Edges are irregular, scalloped, ragged, or poorly defined.</li>
            <li><strong>C - Color:</strong> Color is not uniform across the spot (shades of brown, black, red, white, or blue).</li>
            <li><strong>D - Diameter:</strong> Spot is larger than 6 millimeters across (about pencil eraser size), though melanomas can be smaller.</li>
            <li><strong>E - Evolving:</strong> The spot is changing in size, shape, color, elevation, or starts to bleed or itch.</li>
          </ul>
        </div>

        <div style={{ marginTop: "18px", display: "flex", gap: "12px", flexWrap: "wrap" }}>
          <button className="btn btn-primary" onClick={() => setActiveTab("assessment")}>
            🔬 Assess Another Lesion
          </button>
          <button className="btn btn-secondary" onClick={() => setActiveTab("ar")}>
            👓 Open AR Experience
          </button>
          <button className="btn btn-outline" onClick={() => setActiveTab("model")}>
            📊 View Model Performance
          </button>
        </div>
      </div>
    </div>
  );
}
