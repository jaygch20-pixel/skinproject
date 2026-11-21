import React, { useState, useRef, useEffect } from "react";
import { predictWithFile, predictWithBase64 } from "../services/api";
import MedicalDisclaimer from "../components/MedicalDisclaimer";

export default function Assessment({ onAnalysisComplete, setActiveTab }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [inputMode, setInputMode] = useState("upload"); // 'upload' | 'camera'
  
  // Camera state
  const videoRef = useRef(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraStream, setCameraStream] = useState(null);

  useEffect(() => {
    return () => {
      cameraStream?.getTracks().forEach((track) => track.stop());
    };
  }, [cameraStream]);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (!file.type.startsWith("image/")) {
        setError("Please select a valid image file (JPEG, PNG, WEBP).");
        return;
      }
      setSelectedFile(file);
      setError(null);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      setSelectedFile(file);
      setError(null);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  // Camera handling
  const startCamera = async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 640 }, height: { ideal: 480 } },
      });
      setCameraStream(stream);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      setCameraActive(true);
    } catch (err) {
      setError(`Could not access camera: ${err.message}. Please allow camera permissions.`);
      setCameraActive(false);
    }
  };

  const stopCamera = () => {
    if (cameraStream) {
      cameraStream.getTracks().forEach((track) => track.stop());
      setCameraStream(null);
    }
    setCameraActive(false);
  };

  const captureFrame = () => {
    if (!videoRef.current) return;
    const canvas = document.createElement("canvas");
    canvas.width = videoRef.current.videoWidth || 640;
    canvas.height = videoRef.current.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.95);
    setPreviewUrl(dataUrl);
    setSelectedFile(null); // using base64 preview
    stopCamera();
    setInputMode("upload"); // switch back to preview mode
  };

  const runAnalysis = async () => {
    if (!selectedFile && !previewUrl) {
      setError("Please upload an image or capture a photo first.");
      return;
    }
    setLoading(true);
    setError(null);

    try {
      let result;
      if (selectedFile) {
        result = await predictWithFile(selectedFile);
      } else if (previewUrl && previewUrl.startsWith("data:")) {
        result = await predictWithBase64(previewUrl);
      } else {
        throw new Error("No image data available for analysis.");
      }

      onAnalysisComplete({
        result,
        originalImage: previewUrl,
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
        <h2 className="card-title">🔬 Skin Lesion Assessment & Capture</h2>
        <p style={{ fontSize: "14px", color: "var(--text-muted)", marginBottom: "20px" }}>
          Upload a skin photograph or use your device camera. The model output is educational only and is not a diagnosis.
        </p>

        {/* Input Mode Switcher */}
        <div style={{ display: "flex", gap: "10px", marginBottom: "20px" }}>
          <button
            className={`btn ${inputMode === "upload" ? "btn-primary" : "btn-secondary"}`}
            onClick={() => {
              setInputMode("upload");
              stopCamera();
            }}
          >
            📁 File Upload & Samples
          </button>
          <button
            className={`btn ${inputMode === "camera" ? "btn-primary" : "btn-secondary"}`}
            onClick={() => {
              setInputMode("camera");
              startCamera();
            }}
          >
            📷 Device Camera Capture
          </button>
        </div>

        {error && (
          <div style={{ background: "#fef2f2", border: "1px solid #f87171", color: "#b91c1c", padding: "12px", borderRadius: "6px", marginBottom: "16px", fontSize: "14px" }}>
            ❌ {error}
          </div>
        )}

        {/* Upload Mode */}
        {inputMode === "upload" && (
          <div>
            <div
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              style={{
                border: "2px dashed var(--border-focus)",
                background: "var(--primary-50)",
                padding: "36px 20px",
                borderRadius: "var(--radius)",
                textAlign: "center",
                cursor: "pointer",
                marginBottom: "20px",
              }}
              onClick={() => document.getElementById("file-input").click()}
            >
              <input
                id="file-input"
                type="file"
                accept="image/*"
                style={{ display: "none" }}
                onChange={handleFileChange}
              />
              <div style={{ fontSize: "36px", marginBottom: "8px" }}>📤</div>
              <p style={{ fontWeight: 600, color: "var(--primary-hover)", marginBottom: "4px" }}>
                Click to browse or drag and drop a skin image here
              </p>
              <p style={{ fontSize: "12px", color: "var(--text-muted)" }}>
                Supports standard formats: JPEG, PNG, WEBP (Max 15MB)
              </p>
            </div>

          </div>
        )}

        {/* Camera Mode */}
        {inputMode === "camera" && (
          <div style={{ textAlign: "center", marginBottom: "24px" }}>
            <div className="ar-viewport-box" style={{ maxWidth: "560px", height: "420px" }}>
              <video ref={videoRef} autoPlay playsInline muted className="ar-video" />
              <div className="ar-hud-panel">
                CAMERA: ACTIVE • RESOLUTION: 640x480
              </div>
            </div>
            <div style={{ marginTop: "14px", display: "flex", justifyContent: "center", gap: "12px" }}>
              <button className="btn btn-primary" onClick={captureFrame}>
                📸 Snap & Inspect Photo
              </button>
              <button className="btn btn-secondary" onClick={stopCamera}>
                ⏹️ Close Camera
              </button>
            </div>
          </div>
        )}

        {/* Preview Area */}
        {previewUrl && (
          <div style={{ borderTop: "1px solid var(--border)", paddingTop: "20px" }}>
            <h3 style={{ fontSize: "16px", fontWeight: 700, marginBottom: "12px" }}>
              Target Image Preview
            </h3>
            <div style={{ display: "flex", gap: "20px", alignItems: "flex-start", flexWrap: "wrap" }}>
              <div
                style={{
                  width: "224px",
                  height: "224px",
                  border: "2px solid var(--border)",
                  borderRadius: "8px",
                  overflow: "hidden",
                  background: "#000",
                }}
              >
                <img
                  src={previewUrl}
                  alt="Preview"
                  style={{ width: "100%", height: "100%", objectFit: "cover" }}
                />
              </div>

              <div style={{ flex: 1, minWidth: "260px" }}>
                <p style={{ fontSize: "14px", color: "var(--text-muted)", marginBottom: "12px" }}>
                  Image is ready for deep learning model evaluation (224×224 normalized RGB).
                </p>
                <button
                  className="btn btn-primary"
                  style={{ padding: "12px 24px", fontSize: "15px" }}
                  onClick={runAnalysis}
                  disabled={loading}
                >
                  {loading ? "⏳ Running ResNet18 & OpenCV Assessment..." : "⚡ Analyze Skin Lesion"}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
