import React from "react";

export default function Footer() {
  return (
    <footer className="footer">
      <div style={{ maxWidth: "1200px", margin: "0 auto", display: "flex", flexDirection: "column", gap: "8px" }}>
        <p>
          <strong>Augmented Reality for Early Skin Disease Assessment Using Computer Vision</strong>
        </p>
        <p style={{ fontSize: "12px", color: "var(--text-light)" }}>
          Academic Research Prototype | ResNet18 Deep Residual Learning | OpenCV Spatial Lesion Localization | Browser AR HUD
        </p>
        <p style={{ fontSize: "11px", color: "#94a3b8" }}>
          Strictly for educational and research demonstration. Not cleared as a medical diagnostic tool.
        </p>
      </div>
    </footer>
  );
}
