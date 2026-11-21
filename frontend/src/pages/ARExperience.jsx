import React, { useEffect, useRef, useState } from "react";
import { predictWithBase64 } from "../services/api";
import MedicalDisclaimer from "../components/MedicalDisclaimer";

export default function ARExperience({ onAnalysisComplete }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const animationFrameRef = useRef(null);
  const intervalRef = useRef(null);
  const inferenceInProgressRef = useRef(false);

  const [cameraActive, setCameraActive] = useState(false);
  const [facingMode, setFacingMode] = useState("environment");
  const [error, setError] = useState(null);
  const [cameraLoading, setCameraLoading] = useState(true);
  const [arResult, setArResult] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraActive(false);
    setArResult(null);
  };

  const analyzeFrame = async () => {
    const video = videoRef.current;
    if (!video || video.readyState < 2 || inferenceInProgressRef.current) return;

    inferenceInProgressRef.current = true;
    setIsProcessing(true);

    const frameCanvas = document.createElement("canvas");
    frameCanvas.width = 224;
    frameCanvas.height = 224;
    const ctx = frameCanvas.getContext("2d");
    ctx.drawImage(video, 0, 0, 224, 224);

    try {
      const result = await predictWithBase64(frameCanvas.toDataURL("image/jpeg", 0.82));
      setArResult(result);
      if (onAnalysisComplete) {
        onAnalysisComplete({
          result,
          originalImage: frameCanvas.toDataURL("image/jpeg", 0.82),
          workflow: "live-ar"
        });
      }
    } catch (err) {
      setError(err.message || "Live frame analysis failed.");
    } finally {
      inferenceInProgressRef.current = false;
      setIsProcessing(false);
    }
  };

  const startCamera = async () => {
    stopCamera();
    setCameraLoading(true);
    setError(null);

    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Camera access is not supported by this browser.");
      setCameraLoading(false);
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: facingMode },
          width: { ideal: 640 },
          height: { ideal: 480 }
        },
        audio: false
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraActive(true);
      setCameraLoading(false);
      intervalRef.current = setInterval(analyzeFrame, 1400);
    } catch (err) {
      let message = err.message || "Camera could not be started.";
      if (err.name === "NotAllowedError") message = "Camera permission was denied. Allow camera access and restart.";
      if (err.name === "NotFoundError") message = "No camera device was found.";
      if (err.name === "NotReadableError") message = "The camera is already in use by another application.";
      setError(message);
      setCameraLoading(false);
      setCameraActive(false);
    }
  };

  useEffect(() => {
    startCamera();
    return () => {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    };
  }, []);

  useEffect(() => {
    if (cameraActive) startCamera();
  }, [facingMode]);

  useEffect(() => {
    const renderLoop = () => {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      if (!canvas) {
        animationFrameRef.current = requestAnimationFrame(renderLoop);
        return;
      }

      if (video?.readyState >= 2) {
        if (canvas.width !== video.videoWidth) canvas.width = video.videoWidth || 640;
        if (canvas.height !== video.videoHeight) canvas.height = video.videoHeight || 480;
      }

      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (cameraActive && arResult?.ar_cv_overlay?.detected) {
        const overlay = arResult.ar_cv_overlay;
        const points = overlay.contour_points || [];
        if (points.length > 2) {
          ctx.beginPath();
          points.forEach((point, index) => {
            const x = point.norm_x * canvas.width;
            const y = point.norm_y * canvas.height;
            if (index === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          });
          ctx.closePath();
          ctx.fillStyle = "rgba(0, 240, 255, 0.14)";
          ctx.fill();
          ctx.strokeStyle = "#00f0ff";
          ctx.lineWidth = 3;
          ctx.stroke();
        }
      }

      animationFrameRef.current = requestAnimationFrame(renderLoop);
    };
    animationFrameRef.current = requestAnimationFrame(renderLoop);
    return () => {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    };
  }, [cameraActive, arResult]);

  const restartWithFacingMode = () => {
    setFacingMode((current) => current === "environment" ? "user" : "environment");
  };

  return (
    <div>
      <MedicalDisclaimer />
      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px", flexWrap: "wrap", gap: "10px" }}>
          <div>
            <h2 className="card-title" style={{ margin: 0 }}>Live Augmented Reality Assessment</h2>
            <p style={{ fontSize: "13px", color: "var(--text-muted)" }}>
              Automatic live-frame classification. No capture or upload is required.
            </p>
          </div>
          <div style={{ display: "flex", gap: "8px" }}>
            {!cameraActive ? (
              <button className="btn btn-primary" onClick={startCamera}>Start Camera</button>
            ) : (
              <button className="btn btn-secondary" onClick={stopCamera}>Stop Camera</button>
            )}
            <button className="btn btn-secondary" onClick={restartWithFacingMode}>Restart / Flip</button>
          </div>
        </div>

        {error && (
          <div style={{ background: "#fef2f2", border: "1px solid #f87171", color: "#b91c1c", padding: "10px", borderRadius: "6px", marginBottom: "14px", fontSize: "13px" }}>
            {error}
          </div>
        )}

        <div className="ar-viewport-box">
          {cameraLoading && (
            <div style={{ position: "absolute", inset: 0, zIndex: 5, display: "grid", placeItems: "center", background: "#0f172a", color: "#fff" }}>
              Requesting camera access…
            </div>
          )}
          <video ref={videoRef} autoPlay playsInline muted className="ar-video" />
          <canvas ref={canvasRef} width={640} height={480} className="ar-canvas-overlay" />

          {cameraActive && arResult && (
            <div className="ar-hud-panel">
              <div style={{ color: "#00f0ff", fontWeight: "bold", marginBottom: "4px" }}>
                {isProcessing ? "ANALYZING LIVE FRAME…" : "LIVE MODEL RESULT"}
              </div>
              <div>STATUS: DETECTED</div>
              <div>MODEL CATEGORY: <strong>{arResult.predicted_category.toUpperCase()}</strong></div>
              <div>CONFIDENCE: {arResult.confidence_percentage}%</div>
              <div>REGION AID: {arResult.ar_cv_overlay?.detected ? "APPROXIMATE" : "UNAVAILABLE"}</div>
            </div>
          )}
        </div>

        {arResult && (
          <div style={{ marginTop: "16px", padding: "14px", borderRadius: 8, background: "#f8fafc", border: "1px solid var(--border)" }}>
            <strong>{arResult.predicted_category.toUpperCase()}</strong>
            <span style={{ marginLeft: 12 }}>Confidence: {arResult.confidence_percentage}%</span>
            <p style={{ marginTop: 6, fontSize: 13, color: "var(--text-muted)" }}>
              Live results are educational model estimates. Approximate boundaries are not medically verified lesion segmentation.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
