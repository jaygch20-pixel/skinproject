import React, { useEffect, useRef, useState } from "react";
import { predictWithBase64 } from "../services/api";

const FRAME_INTERVAL_MS = 500;
const OVERLAY_MAX_AGE_MS = 1200;

function smoothNearbyContour(previous, next) {
  const oldBox = previous?.normalized_bounding_box;
  const newBox = next?.normalized_bounding_box;
  const oldPoints = previous?.contour_points || [];
  const newPoints = next?.contour_points || [];
  if (!oldBox || !newBox || oldPoints.length !== newPoints.length || !newPoints.length) {
    return next;
  }

  const left = Math.max(oldBox.x, newBox.x);
  const top = Math.max(oldBox.y, newBox.y);
  const right = Math.min(oldBox.x + oldBox.width, newBox.x + newBox.width);
  const bottom = Math.min(oldBox.y + oldBox.height, newBox.y + newBox.height);
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  const union =
    oldBox.width * oldBox.height +
    newBox.width * newBox.height -
    intersection;
  if (!union || intersection / union < 0.2) return next;

  let alignedPoints = newPoints;
  let minimumDistance = Number.POSITIVE_INFINITY;
  for (const candidate of [newPoints, [...newPoints].reverse()]) {
    for (let shift = 0; shift < candidate.length; shift += 1) {
      let distance = 0;
      for (let index = 0; index < candidate.length; index += 1) {
        const point = candidate[(index + shift) % candidate.length];
        const dx = point.norm_x - oldPoints[index].norm_x;
        const dy = point.norm_y - oldPoints[index].norm_y;
        distance += dx * dx + dy * dy;
      }
      if (distance < minimumDistance) {
        minimumDistance = distance;
        alignedPoints = candidate.map((_, index) =>
          candidate[(index + shift) % candidate.length]
        );
      }
    }
  }

  const contourPoints = alignedPoints.map((point, index) => ({
    ...point,
    norm_x: point.norm_x * 0.65 + oldPoints[index].norm_x * 0.35,
    norm_y: point.norm_y * 0.65 + oldPoints[index].norm_y * 0.35
  }));
  return { ...next, contour_points: contourPoints };
}

export default function ARExperience({ onAnalysisComplete }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const animationFrameRef = useRef(null);
  const intervalRef = useRef(null);
  const inferenceInProgressRef = useRef(false);
  const overlayRef = useRef(null);
  const overlayTimestampRef = useRef(0);
  const previousOverlayRef = useRef(null);
  const facingModeRef = useRef("environment");

  const [cameraActive, setCameraActive] = useState(false);
  const [facingMode, setFacingMode] = useState("environment");
  const [error, setError] = useState(null);
  const [cameraLoading, setCameraLoading] = useState(true);
  const [arResult, setArResult] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [overlayFresh, setOverlayFresh] = useState(false);
  const [videoSize, setVideoSize] = useState({ width: 4, height: 3 });
  const overlayFreshRef = useRef(false);

  facingModeRef.current = facingMode;

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    overlayRef.current = null;
    previousOverlayRef.current = null;
    overlayTimestampRef.current = 0;
    overlayFreshRef.current = false;
    setOverlayFresh(false);
    setCameraActive(false);
    setArResult(null);
  };

  const analyzeFrame = async () => {
    const video = videoRef.current;
    if (
      !video ||
      video.readyState < 2 ||
      !video.videoWidth ||
      inferenceInProgressRef.current
    ) return;

    inferenceInProgressRef.current = true;
    setIsProcessing(true);
    const scale = Math.min(1, 640 / video.videoWidth, 480 / video.videoHeight);
    const frameCanvas = document.createElement("canvas");
    frameCanvas.width = Math.max(1, Math.round(video.videoWidth * scale));
    frameCanvas.height = Math.max(1, Math.round(video.videoHeight * scale));
    const context = frameCanvas.getContext("2d");
    if (!context) {
      inferenceInProgressRef.current = false;
      setIsProcessing(false);
      setError("Could not prepare a camera frame for analysis.");
      return;
    }
    context.drawImage(video, 0, 0, frameCanvas.width, frameCanvas.height);
    const frameImage = frameCanvas.toDataURL("image/jpeg", 0.82);
    const capturedAt = Date.now();

    try {
      const result = await predictWithBase64(frameImage, {
        include_visualizations: false
      });
      setError(null);
      const nextOverlay = result.ar_cv_overlay || {};
      if (nextOverlay.detected) {
        const smoothedOverlay = smoothNearbyContour(
          previousOverlayRef.current,
          nextOverlay
        );
        result.ar_cv_overlay = smoothedOverlay;
        overlayRef.current = smoothedOverlay;
        previousOverlayRef.current = smoothedOverlay;
        overlayTimestampRef.current = Date.now();
        overlayFreshRef.current = true;
        setOverlayFresh(true);
      } else {
        overlayRef.current = null;
        previousOverlayRef.current = null;
        overlayTimestampRef.current = 0;
        overlayFreshRef.current = false;
        setOverlayFresh(false);
      }

      setArResult(result);
      if (onAnalysisComplete) {
        onAnalysisComplete({
          result,
          originalImage: frameImage,
          workflow: "live-ar",
          capturedAt
        });
      }
    } catch (requestError) {
      overlayRef.current = null;
      previousOverlayRef.current = null;
      overlayTimestampRef.current = 0;
      overlayFreshRef.current = false;
      setOverlayFresh(false);
      setArResult(null);
      setError(requestError.message || "Live frame analysis failed.");
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
          facingMode: { ideal: facingModeRef.current },
          width: { ideal: 640 },
          height: { ideal: 480 }
        },
        audio: false
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setVideoSize({
          width: videoRef.current.videoWidth || 4,
          height: videoRef.current.videoHeight || 3
        });
      }
      setCameraActive(true);
      setCameraLoading(false);
      intervalRef.current = setInterval(analyzeFrame, FRAME_INTERVAL_MS);
    } catch (cameraError) {
      let message = cameraError.message || "Camera could not be started.";
      if (cameraError.name === "NotAllowedError") {
        message = "Camera permission was denied. Allow camera access and restart.";
      }
      if (cameraError.name === "NotFoundError") {
        message = "No camera device was found.";
      }
      if (cameraError.name === "NotReadableError") {
        message = "The camera is already in use by another application.";
      }
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
    if (!cameraActive) {
      const context = canvasRef.current?.getContext("2d");
      context?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      return undefined;
    }

    const renderOverlay = () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (video && canvas && video.videoWidth && video.videoHeight) {
        if (canvas.width !== video.videoWidth) canvas.width = video.videoWidth;
        if (canvas.height !== video.videoHeight) canvas.height = video.videoHeight;
        const context = canvas.getContext("2d");
        if (context) {
          context.clearRect(0, 0, canvas.width, canvas.height);
          const overlay = overlayRef.current;
          const isFresh =
            Date.now() - overlayTimestampRef.current <= OVERLAY_MAX_AGE_MS;
          if (overlay?.detected && isFresh) {
            const points = overlay.contour_points || [];
            if (points.length > 2) {
              context.beginPath();
              points.forEach((point, index) => {
                const x = point.norm_x * canvas.width;
                const y = point.norm_y * canvas.height;
                if (index === 0) context.moveTo(x, y);
                else context.lineTo(x, y);
              });
              context.closePath();
              context.fillStyle = "rgba(0, 122, 255, 0.18)";
              context.fill();
              context.strokeStyle = "#1687ff";
              context.lineWidth = Math.max(2, canvas.width / 320);
              context.stroke();

              const box = overlay.normalized_bounding_box;
              if (box) {
                context.strokeStyle = "rgba(255,255,255,0.9)";
                context.lineWidth = Math.max(1, canvas.width / 640);
                context.strokeRect(
                  box.x * canvas.width,
                  box.y * canvas.height,
                  box.width * canvas.width,
                  box.height * canvas.height
                );
              }
              const center = overlay.center;
              if (center) {
                context.beginPath();
                context.arc(center.x, center.y, Math.max(3, canvas.width / 160), 0, Math.PI * 2);
                context.fillStyle = "#ffffff";
                context.fill();
              }
            }
          } else if (overlay && !isFresh) {
            overlayRef.current = null;
            previousOverlayRef.current = null;
            if (overlayFreshRef.current) {
              overlayFreshRef.current = false;
              setOverlayFresh(false);
            }
          }
        }
      }
      animationFrameRef.current = requestAnimationFrame(renderOverlay);
    };
    animationFrameRef.current = requestAnimationFrame(renderOverlay);
    return () => {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    };
  }, [cameraActive]);

  const restartWithFacingMode = () => {
    setFacingMode((current) => current === "environment" ? "user" : "environment");
  };

  const detected = overlayFresh && Boolean(arResult?.ar_cv_overlay?.detected);
  return (
    <div>
      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px", flexWrap: "wrap", gap: "10px" }}>
          <div>
            <h2 className="card-title" style={{ margin: 0 }}>Live Augmented Reality Assessment</h2>
            <p style={{ fontSize: "13px", color: "var(--text-muted)" }}>
              Live classifier and OpenCV lesion-candidate analysis. The boundary is approximate, not a medical segmentation.
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

        <div className="ar-viewport-box" style={{ aspectRatio: `${videoSize.width} / ${videoSize.height}` }}>
          {cameraLoading && (
            <div style={{ position: "absolute", inset: 0, zIndex: 5, display: "grid", placeItems: "center", background: "#0f172a", color: "#fff" }}>
              Requesting camera access…
            </div>
          )}
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            onLoadedMetadata={(event) => setVideoSize({
              width: event.currentTarget.videoWidth || 4,
              height: event.currentTarget.videoHeight || 3
            })}
            className={`ar-video ${facingMode === "user" ? "ar-video-mirrored" : ""}`}
          />
          <canvas
            ref={canvasRef}
            width={640}
            height={480}
            className={`ar-canvas-overlay ${facingMode === "user" ? "ar-canvas-mirrored" : ""}`}
          />

          {cameraActive && (
            <div className="ar-hud-panel">
              <div style={{ color: "#00f0ff", fontWeight: "bold", marginBottom: "4px" }}>
                {isProcessing ? "ANALYZING LIVE FRAME…" : "LIVE ANALYSIS"}
              </div>
              <div>LESION CANDIDATE: {detected ? "DETECTED" : "NOT DETECTED"}</div>
              {arResult && (
                <>
                  <div>MODEL CATEGORY: <strong>{arResult.predicted_category.toUpperCase()}</strong></div>
                  <div>CONFIDENCE: {arResult.confidence_percentage}%</div>
                </>
              )}
            </div>
          )}
        </div>

        {arResult && (
          <div style={{ marginTop: "16px", padding: "14px", borderRadius: 8, background: "#f8fafc", border: "1px solid var(--border)" }}>
            <strong>{arResult.predicted_category.toUpperCase()}</strong>
            <span style={{ marginLeft: 12 }}>Confidence: {arResult.confidence_percentage}%</span>
            <p style={{ marginTop: 6, fontSize: 13, color: "var(--text-muted)" }}>
              {detected
                ? "Blue outline is an approximate OpenCV color-contrast candidate, not a disease segmentation."
                : arResult.ar_cv_overlay?.segmentation_warning || "Lesion boundary not detected."}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
