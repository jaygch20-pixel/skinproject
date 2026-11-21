import React, { useRef, useState } from "react";
import { predictWithFile } from "../services/api";
import MedicalDisclaimer from "../components/MedicalDisclaimer";

export default function Assessment({ onAnalysisComplete, setActiveTab }) {
  const fileInputRef = useRef(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const selectFile = (file) => {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setError("Please select a valid image file (JPEG, PNG, WEBP).");
      return;
    }
    if (file.size > 15 * 1024 * 1024) {
      setError("Image file exceeds the 15 MB upload limit.");
      return;
    }
    setSelectedFile(file);
    setError(null);
    setPreviewUrl(URL.createObjectURL(file));
  };

  const handleFileChange = (event) => selectFile(event.target.files[0]);

  const runAnalysis = async () => {
    if (!selectedFile) {
      setError("Please upload an image first.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const result = await predictWithFile(selectedFile);
      onAnalysisComplete({
        result,
        originalImage: previewUrl,
        workflow: "upload"
      });
      setActiveTab("results");
    } catch (err) {
      setError(err.message || "Failed to communicate with prediction service.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <MedicalDisclaimer />
      <div className="card">
        <h2 className="card-title">Skin Lesion Image Assessment</h2>
        <p style={{ fontSize: "14px", color: "var(--text-muted)", marginBottom: "20px" }}>
          Upload a skin photograph for model analysis and five automatic segmentation visualizations.
        </p>

        {error && (
          <div style={{ background: "#fef2f2", border: "1px solid #f87171", color: "#b91c1c", padding: "12px", borderRadius: "6px", marginBottom: "16px", fontSize: "14px" }}>
            {error}
          </div>
        )}

        <div
          onDragOver={(event) => event.preventDefault()}
          onDrop={(event) => {
            event.preventDefault();
            selectFile(event.dataTransfer.files[0]);
          }}
          style={{
            border: "2px dashed var(--border-focus)",
            background: "var(--primary-50)",
            padding: "36px 20px",
            borderRadius: "var(--radius)",
            textAlign: "center",
            cursor: "pointer",
            marginBottom: "20px"
          }}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            style={{ display: "none" }}
            onChange={handleFileChange}
          />
          <div style={{ fontSize: "36px", marginBottom: "8px" }}>Upload</div>
          <p style={{ fontWeight: 600, color: "var(--primary-hover)", marginBottom: "4px" }}>
            Click to browse or drag and drop a skin image here
          </p>
          <p style={{ fontSize: "12px", color: "var(--text-muted)" }}>
            Supports JPEG, PNG, WEBP, and BMP (Max 15 MB)
          </p>
        </div>

        {previewUrl && (
          <div style={{ borderTop: "1px solid var(--border)", paddingTop: "20px" }}>
            <h3 style={{ fontSize: "16px", fontWeight: 700, marginBottom: "12px" }}>Target Image Preview</h3>
            <div style={{ display: "flex", gap: "20px", alignItems: "flex-start", flexWrap: "wrap" }}>
              <div style={{ width: "224px", height: "224px", border: "2px solid var(--border)", borderRadius: "8px", overflow: "hidden", background: "#000" }}>
                <img src={previewUrl} alt="Uploaded preview" style={{ width: "100%", height: "100%", objectFit: "contain" }} />
              </div>
              <div style={{ flex: 1, minWidth: "260px" }}>
                <p style={{ fontSize: "14px", color: "var(--text-muted)", marginBottom: "12px" }}>
                  Analysis will generate the prediction and five visualizations automatically.
                </p>
                <button className="btn btn-primary" style={{ padding: "12px 24px", fontSize: "15px" }} onClick={runAnalysis} disabled={loading}>
                  {loading ? "Running ResNet18 & OpenCV analysis…" : "Analyze Skin Lesion"}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
