import React, { useState, useEffect, useRef } from "react";
import { predictWithBase64, predictWithFile } from "../services/api";
import MedicalDisclaimer from "../components/MedicalDisclaimer";

export default function ARExperience({ onAnalysisComplete, setActiveTab }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const animationFrameRef = useRef(null);

  const [cameraActive, setCameraActive] = useState(false);
  const [cameraStream, setCameraStream] = useState(null);
  const [facingMode, setFacingMode] = useState("environment"); // or 'user'
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  
  // AR Overlay state
  const [arResult, setArResult] = useState(null);
  const [capturedSnapshot, setCapturedSnapshot] = useState(null);
  const [overlayMode, setOverlayMode] = useState("camera"); // 'camera' | 'uploaded'
  const [uploadedPreview, setUploadedPreview] = useState(null);

  // Start Camera
  const startCamera = async () => {
    try {
      setError(null);
      if (cameraStream) {
        cameraStream.getTracks().forEach((t) => t.stop());
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: facingMode, width: { ideal: 640 }, height: { ideal: 480 } },
      });
      setCameraStream(stream);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      setCameraActive(true);
    } catch (err) {
      setError(`Camera access error: ${err.message}. Please check browser camera permissions.`);
      setCameraActive(false);
    }
  };

  const stopCamera = () => {
    if (cameraStream) {
      cameraStream.getTracks().forEach((track) => track.stop());
      setCameraStream(null);
    }
    setCameraActive(false);
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
  };

  const toggleCameraFacing = () => {
    const nextFacing = facingMode === "environment" ? "user" : "environment";
    setFacingMode(nextFacing);
  };

  useEffect(() => {
    if (cameraActive) {
      startCamera();
    }
    return () => {
      stopCamera();
    };
  }, [facingMode]);

  // Real-time Canvas AR Loop (Drawing Reticles & Dynamic HUD)
  useEffect(() => {
    let scanLineY = 0;
    let scanDirection = 1;

    const renderLoop = () => {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext("2d");
      const width = canvas.width;
      const height = canvas.height;

      ctx.clearRect(0, 0, width, height);

      // If we have an active video stream, draw the AR targeting reticle
      if (cameraActive && video && video.readyState >= 2) {
        // Draw HUD Crosshairs & Center Brackets
        const cx = width / 2;
        const cy = height / 2;
        const targetSize = Math.min(width, height) * 0.42;

        // Bounding Bracket coordinates
        const x1 = cx - targetSize / 2;
        const y1 = cy - targetSize / 2;
        const x2 = cx + targetSize / 2;
        const y2 = cy + targetSize / 2;
        const bracketLen = 24;

        ctx.strokeStyle = "#00f0ff";
        ctx.lineWidth = 2.5;
        ctx.shadowColor = "#00f0ff";
        ctx.shadowBlur = 8;

        // Corner Brackets
        ctx.beginPath();
        // Top-left
        ctx.moveTo(x1, y1 + bracketLen); ctx.lineTo(x1, y1); ctx.lineTo(x1 + bracketLen, y1);
        // Top-right
        ctx.moveTo(x2 - bracketLen, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + bracketLen);
        // Bottom-left
        ctx.moveTo(x1, y2 - bracketLen); ctx.lineTo(x1, y2); ctx.lineTo(x1 + bracketLen, y2);
        // Bottom-right
        ctx.moveTo(x2 - bracketLen, y2); ctx.lineTo(x2, y2); ctx.lineTo(x2, y2 - bracketLen);
        ctx.stroke();

        // Center reticle marker
        ctx.beginPath();
        ctx.arc(cx, cy, 6, 0, 2 * Math.PI);
        ctx.stroke();

        // Animated Scanning Line
        scanLineY += 2 * scanDirection;
        if (scanLineY > y2) scanDirection = -1;
        if (scanLineY < y1) scanDirection = 1;

        ctx.strokeStyle = "rgba(0, 240, 255, 0.45)";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(x1 + 4, scanLineY);
        ctx.lineTo(x2 - 4, scanLineY);
        ctx.stroke();
      }

      // Draw OpenCV-detected lesion contour and bounding box if available
      if (overlayMode === "uploaded" && arResult && arResult.ar_cv_overlay) {
        const overlay = arResult.ar_cv_overlay;
        const isBenign = arResult.predicted_category === "benign";
        const themeColor = "#00f0ff";

        ctx.shadowColor = themeColor;
        ctx.shadowBlur = 12;

        // Draw Contour Polygon if detected
        if (overlay.contour_points && overlay.contour_points.length > 2) {
          ctx.strokeStyle = themeColor;
          ctx.lineWidth = 3;
          ctx.fillStyle = "rgba(0, 240, 255, 0.15)";
          
          ctx.beginPath();
          overlay.contour_points.forEach((pt, idx) => {
            const px = pt.norm_x * width;
            const py = pt.norm_y * height;
            if (idx === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
          });
          ctx.closePath();
          ctx.fill();
          ctx.stroke();
        }

        // Draw Detected Lesion Bounding Box
        if (overlay.normalized_bounding_box) {
          const nb = overlay.normalized_bounding_box;
          const bx = nb.x * width;
          const by = nb.y * height;
          const bw = nb.width * width;
          const bh = nb.height * height;

          ctx.strokeStyle = themeColor;
          ctx.lineWidth = 2;
          ctx.strokeRect(bx, by, bw, bh);

          // Corner tags
          const bLen = 14;
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.moveTo(bx, by + bLen); ctx.lineTo(bx, by); ctx.lineTo(bx + bLen, by);
          ctx.moveTo(bx + bw - bLen, by); ctx.lineTo(bx + bw, by); ctx.lineTo(bx + bw, by + bLen);
          ctx.moveTo(bx, by + bh - bLen); ctx.lineTo(bx, by + bh); ctx.lineTo(bx + bLen, by + bh);
          ctx.moveTo(bx + bw - bLen, by + bh); ctx.lineTo(bx + bw, by + bh); ctx.lineTo(bx + bw, by + bh - bLen);
          ctx.stroke();
        }
      }

      ctx.shadowBlur = 0;
      animationFrameRef.current = requestAnimationFrame(renderLoop);
    };

    animationFrameRef.current = requestAnimationFrame(renderLoop);
    return () => {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    };
  }, [cameraActive, arResult]);

  // Capture & Run Live Assessment
  const captureAndAssess = async () => {
    let base64Data;
    if (overlayMode === "camera") {
      if (!videoRef.current) return;
      const snapCanvas = document.createElement("canvas");
      snapCanvas.width = videoRef.current.videoWidth || 640;
      snapCanvas.height = videoRef.current.videoHeight || 480;
      const ctx = snapCanvas.getContext("2d");
      ctx.drawImage(videoRef.current, 0, 0, snapCanvas.width, snapCanvas.height);
      base64Data = snapCanvas.toDataURL("image/jpeg", 0.95);
    } else {
      base64Data = capturedSnapshot || uploadedPreview;
    }

    if (!base64Data) {
      setError("Please ensure camera is active or select an image.");
      return;
    }

    setIsProcessing(true);
    setError(null);
    setCapturedSnapshot(base64Data);
    setArResult(null);
    if (overlayMode === "camera") {
      setOverlayMode("uploaded");
      stopCamera();
    }

    try {
      const result = await predictWithBase64(base64Data);
      setArResult(result);
      if (onAnalysisComplete) {
        onAnalysisComplete({
          result,
          originalImage: base64Data,
        });
      }
    } catch (err) {
      setError(`AR Analysis failed: ${err.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleUploadChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setUploadedPreview(url);
      setOverlayMode("uploaded");
      setArResult(null);
      stopCamera();
      
      // Auto analyze uploaded image in AR mode
      const reader = new FileReader();
      reader.onload = async (ev) => {
        const b64 = ev.target.result;
        setCapturedSnapshot(b64);
        setIsProcessing(true);
        try {
          const res = await predictWithBase64(b64);
          setArResult(res);
          if (onAnalysisComplete) onAnalysisComplete({ result: res, originalImage: b64 });
        } catch (err) {
          setError(`AR Processing error: ${err.message}`);
        } finally {
          setIsProcessing(false);
        }
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div>
      <MedicalDisclaimer />

      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px", flexWrap: "wrap", gap: "10px" }}>
          <div>
            <h2 className="card-title" style={{ margin: 0 }}>👓 Interactive Augmented Reality HUD</h2>
            <p style={{ fontSize: "13px", color: "var(--text-muted)" }}>
            Camera preview with a capture-time, approximate OpenCV visual aid and educational model output.
            </p>
          </div>

          <div style={{ display: "flex", gap: "8px" }}>
            <button
              className={`btn ${overlayMode === "camera" ? "btn-primary" : "btn-secondary"}`}
              onClick={() => {
                setOverlayMode("camera");
                setArResult(null);
                startCamera();
              }}
            >
              📹 Live Camera
            </button>
            <button
              className={`btn ${overlayMode === "uploaded" ? "btn-primary" : "btn-secondary"}`}
              onClick={() => {
                setOverlayMode("uploaded");
                stopCamera();
                document.getElementById("ar-upload-file").click();
              }}
            >
              🖼️ Uploaded Photo
            </button>
            <input
              id="ar-upload-file"
              type="file"
              accept="image/*"
              style={{ display: "none" }}
              onChange={handleUploadChange}
            />
          </div>
        </div>

        {error && (
          <div style={{ background: "#fef2f2", border: "1px solid #f87171", color: "#b91c1c", padding: "10px", borderRadius: "6px", marginBottom: "14px", fontSize: "13px" }}>
            {error}
          </div>
        )}

        {/* Viewport Box */}
        <div className="ar-viewport-box">
          {overlayMode === "camera" ? (
            <video ref={videoRef} autoPlay playsInline muted className="ar-video" />
          ) : (
            <img
              src={capturedSnapshot || uploadedPreview}
              alt="AR View"
              style={{ width: "100%", height: "100%", objectFit: "contain", background: "#0f172a" }}
            />
          )}

          {/* AR Overlay Canvas */}
          <canvas
            ref={canvasRef}
            width={640}
            height={480}
            className="ar-canvas-overlay"
          />

          {/* AR Telemetry HUD Top-Left */}
          <div className="ar-hud-panel">
            <div style={{ color: "#00f0ff", fontWeight: "bold", marginBottom: "4px" }}>
              ⚡ AR VISUAL AID • RESEARCH
            </div>
            <div>CAMERA: {cameraActive ? "LIVE PREVIEW" : "OFF / STATIC IMAGE"}</div>
            <div>MODE: {overlayMode.toUpperCase()}</div>
            <div>CV: APPROXIMATE, CAPTURE-TIME ONLY</div>
            {arResult && (
              <div style={{ marginTop: "6px", borderTop: "1px dashed rgba(0, 240, 255, 0.4)", paddingTop: "4px" }}>
                <div>MODEL CATEGORY: <strong>{arResult.predicted_category.toUpperCase()}</strong></div>
                <div>MODEL CONFIDENCE: {(arResult.confidence_score * 100).toFixed(1)}%</div>
                <div>REGION AID: {arResult.ar_cv_overlay?.detected ? "HEURISTIC" : "UNAVAILABLE"}</div>
              </div>
            )}
          </div>

          {/* AR Controls Bottom Bar */}
          <div className="ar-hud-bottom">
            {overlayMode === "camera" && (
              <>
                {!cameraActive ? (
                  <button className="btn btn-primary" onClick={startCamera}>
                    ▶️ Start Camera
                  </button>
                ) : (
                  <>
                    <button
                      className="btn"
                      style={{ background: "#00f0ff", color: "#0f172a", fontWeight: "bold" }}
                      onClick={captureAndAssess}
                      disabled={isProcessing}
                    >
                      {isProcessing ? "⏳ Assessing..." : "🎯 Capture & Assess Lesion"}
                    </button>
                    <button className="btn btn-secondary" onClick={toggleCameraFacing} title="Flip Camera">
                      🔄 Flip
                    </button>
                    <button className="btn btn-secondary" onClick={stopCamera}>
                      ⏹️ Stop
                    </button>
                  </>
                )}
              </>
            )}

            {overlayMode === "uploaded" && (
              <button
                className="btn btn-primary"
                onClick={captureAndAssess}
                disabled={isProcessing}
              >
                {isProcessing ? "⏳ Analyzing..." : "⚡ Re-analyze captured image"}
              </button>
            )}
          </div>
        </div>

        {/* Diagnostic Assessment Banner */}
        {arResult && (
          <div
            style={{
              marginTop: "20px",
              padding: "16px",
              borderRadius: "8px",
              background: arResult.predicted_category === "benign" ? "var(--benign-bg)" : "var(--malignant-bg)",
              border: `1px solid ${arResult.predicted_category === "benign" ? "#86efac" : "#fca5a5"}`,
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              flexWrap: "wrap",
              gap: "12px",
            }}
          >
            <div>
              <span className={`badge ${arResult.predicted_category === "benign" ? "badge-benign" : "badge-malignant"}`}>
                {arResult.predicted_category}
              </span>
              <strong style={{ marginLeft: "12px", fontSize: "16px" }}>
                Model Confidence: {arResult.confidence_percentage}%
              </strong>
              <p style={{ fontSize: "13px", marginTop: "4px", color: "var(--text-main)" }}>
                {arResult.assessment_summary?.clinical_description}
              </p>
            </div>

            <button
              className="btn btn-primary"
              onClick={() => setActiveTab("results")}
            >
              View Assessment Details →
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
