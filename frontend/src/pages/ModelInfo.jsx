import React, { useState, useEffect } from "react";
import { getModelInfo, getEvaluationMetrics, getChartUrl } from "../services/api";

export default function ModelInfo() {
  const [modelData, setModelData] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        const [info, met] = await Promise.all([
          getModelInfo().catch(() => null),
          getEvaluationMetrics().catch(() => null),
        ]);
        setModelData(info);
        setMetrics(met);
      } catch (err) {
        console.error("Failed to load model info:", err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  return (
    <div>

      {/* Overview Card */}
      <div className="card">
        <h2 className="card-title">📊 ResNet18 Model Architecture & Performance</h2>
        <p style={{ fontSize: "14px", color: "var(--text-muted)", marginBottom: "20px" }}>
          Comprehensive architectural specifications, training telemetry, and independent test dataset evaluation metrics.
        </p>

        <div className="grid-3" style={{ marginBottom: "20px" }}>
          <div style={{ background: "var(--primary-50)", padding: "16px", borderRadius: "8px", border: "1px solid #bae6fd" }}>
            <span style={{ fontSize: "12px", color: "var(--primary)", fontWeight: 700, textTransform: "uppercase" }}>
              Architecture
            </span>
            <div style={{ fontSize: "20px", fontWeight: 800, color: "var(--primary-hover)", marginTop: "4px" }}>
              ResNet18 (Residual CNN)
            </div>
            <p style={{ fontSize: "12px", color: "var(--text-muted)", marginTop: "4px" }}>
              18 Convolutional layers with identity shortcut connections
            </p>
          </div>

          <div style={{ background: "#f0fdf4", padding: "16px", borderRadius: "8px", border: "1px solid #bbf7d0" }}>
            <span style={{ fontSize: "12px", color: "#15803d", fontWeight: 700, textTransform: "uppercase" }}>
              Dataset Split
            </span>
            <div style={{ fontSize: "20px", fontWeight: 800, color: "#166534", marginTop: "4px" }}>
              70% / 15% / 15%
            </div>
            <p style={{ fontSize: "12px", color: "var(--text-muted)", marginTop: "4px" }}>
              Balanced train (7,684) | Validation (1,585) | Test (1,587)
            </p>
          </div>

          <div style={{ background: "#faf5ff", padding: "16px", borderRadius: "8px", border: "1px solid #e9d5ff" }}>
            <span style={{ fontSize: "12px", color: "#7e22ce", fontWeight: 700, textTransform: "uppercase" }}>
              Disease Categories
            </span>
            <div style={{ fontSize: "20px", fontWeight: 800, color: "#581c87", marginTop: "4px" }}>
              2 Binary Classes
            </div>
            <p style={{ fontSize: "12px", color: "var(--text-muted)", marginTop: "4px" }}>
              Benign vs malignant dataset labels
            </p>
          </div>
        </div>
      </div>

      {/* Quantitative KPI Metrics */}
      {metrics && metrics.overall_accuracy && (
        <div className="card">
          <h3 className="card-title">🎯 Independent Test Set Evaluation Metrics</h3>
          <p style={{ fontSize: "13px", color: "var(--text-muted)", marginBottom: "16px" }}>
            Evaluated on 1,587 separate test images after duplicate filtering:
          </p>
          <div className="grid-3" style={{ marginBottom: "20px" }}>
            <div style={{ textAlign: "center", padding: "16px", background: "#f8fafc", borderRadius: "8px", border: "1px solid var(--border)" }}>
              <div style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>Overall Accuracy</div>
              <div style={{ fontSize: "28px", fontWeight: 800, color: "var(--primary)" }}>{(metrics.overall_accuracy * 100).toFixed(2)}%</div>
            </div>
            <div style={{ textAlign: "center", padding: "16px", background: "#f8fafc", borderRadius: "8px", border: "1px solid var(--border)" }}>
              <div style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>Malignant Sensitivity (Recall)</div>
              <div style={{ fontSize: "28px", fontWeight: 800, color: "#dc2626" }}>{(metrics.sensitivity * 100).toFixed(2)}%</div>
            </div>
            <div style={{ textAlign: "center", padding: "16px", background: "#f8fafc", borderRadius: "8px", border: "1px solid var(--border)" }}>
              <div style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>Malignant Specificity</div>
              <div style={{ fontSize: "28px", fontWeight: 800, color: "#16a34a" }}>{(metrics.specificity * 100).toFixed(2)}%</div>
            </div>
            <div style={{ textAlign: "center", padding: "16px", background: "#f8fafc", borderRadius: "8px", border: "1px solid var(--border)" }}>
              <div style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>Macro Precision</div>
              <div style={{ fontSize: "28px", fontWeight: 800, color: "var(--text-main)" }}>{(metrics.macro_precision * 100).toFixed(2)}%</div>
            </div>
            <div style={{ textAlign: "center", padding: "16px", background: "#f8fafc", borderRadius: "8px", border: "1px solid var(--border)" }}>
              <div style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>Macro F1-Score</div>
              <div style={{ fontSize: "28px", fontWeight: 800, color: "var(--text-main)" }}>{(metrics.macro_f1 * 100).toFixed(2)}%</div>
            </div>
            <div style={{ textAlign: "center", padding: "16px", background: "#f8fafc", borderRadius: "8px", border: "1px solid var(--border)" }}>
              <div style={{ fontSize: "11px", color: "var(--text-muted)", textTransform: "uppercase", fontWeight: 700 }}>ROC-AUC Score</div>
              <div style={{ fontSize: "28px", fontWeight: 800, color: "#2563eb" }}>{metrics.roc_auc?.toFixed(4)}</div>
            </div>
          </div>
        </div>
      )}

      {/* Evaluation Visualizations */}
      <div className="card">
        <h3 className="card-title">📈 Model Training Curves & Confusion Matrix Visualizations</h3>
        <p style={{ fontSize: "13px", color: "var(--text-muted)", marginBottom: "20px" }}>
          Recorded training curves and independent test-set evaluation generated by the model pipeline:
        </p>

        <div className="grid-2">
          {/* Accuracy Curve */}
          <div style={{ border: "1px solid var(--border)", borderRadius: "8px", padding: "12px", background: "#fff" }}>
            <h4 style={{ fontSize: "14px", fontWeight: 700, marginBottom: "8px", color: "var(--text-main)" }}>
              Training & Validation Accuracy (recorded epochs)
            </h4>
            <img
              src={getChartUrl("training_validation_accuracy.png")}
              alt="Training Accuracy"
              style={{ width: "100%", height: "auto", borderRadius: "6px" }}
              onError={(e) => { e.target.style.display = 'none'; }}
            />
          </div>

          {/* Loss Curve */}
          <div style={{ border: "1px solid var(--border)", borderRadius: "8px", padding: "12px", background: "#fff" }}>
            <h4 style={{ fontSize: "14px", fontWeight: 700, marginBottom: "8px", color: "var(--text-main)" }}>
              Training & Validation Loss (recorded epochs)
            </h4>
            <img
              src={getChartUrl("training_validation_loss.png")}
              alt="Training Loss"
              style={{ width: "100%", height: "auto", borderRadius: "6px" }}
              onError={(e) => { e.target.style.display = 'none'; }}
            />
          </div>

          {/* Confusion Matrix */}
          <div style={{ border: "1px solid var(--border)", borderRadius: "8px", padding: "12px", background: "#fff" }}>
            <h4 style={{ fontSize: "14px", fontWeight: 700, marginBottom: "8px", color: "var(--text-main)" }}>
              Confusion Matrix on Independent Test Set
            </h4>
            <img
              src={getChartUrl("confusion_matrix.png")}
              alt="Confusion Matrix"
              style={{ width: "100%", height: "auto", borderRadius: "6px" }}
              onError={(e) => { e.target.style.display = 'none'; }}
            />
          </div>

          {/* ROC Curve */}
          <div style={{ border: "1px solid var(--border)", borderRadius: "8px", padding: "12px", background: "#fff" }}>
            <h4 style={{ fontSize: "14px", fontWeight: 700, marginBottom: "8px", color: "var(--text-main)" }}>
              Receiver Operating Characteristic (ROC) Curve
            </h4>
            <img
              src={getChartUrl("roc_curve.png")}
              alt="ROC Curve"
              style={{ width: "100%", height: "auto", borderRadius: "6px" }}
              onError={(e) => { e.target.style.display = 'none'; }}
            />
          </div>
        </div>
      </div>

      {/* Dataset Balancing Visualizations */}
      <div className="card">
        <h3 className="card-title">⚖️ Dataset Balancing & Distribution</h3>
        <div className="grid-2" style={{ marginTop: "16px" }}>
          <div style={{ border: "1px solid var(--border)", borderRadius: "8px", padding: "12px", background: "#fff" }}>
            <h4 style={{ fontSize: "14px", fontWeight: 700, marginBottom: "8px" }}>
              Class Distribution: Before vs After Balancing
            </h4>
            <img
              src={getChartUrl("before_and_after_class_balancing.png")}
              alt="Balancing Distribution"
              style={{ width: "100%", height: "auto", borderRadius: "6px" }}
            />
          </div>

          <div style={{ border: "1px solid var(--border)", borderRadius: "8px", padding: "12px", background: "#fff" }}>
            <h4 style={{ fontSize: "14px", fontWeight: 700, marginBottom: "8px" }}>
              Original Dataset Category Proportion
            </h4>
            <img
              src={getChartUrl("dataset_distribution.png")}
              alt="Original Distribution"
              style={{ width: "100%", height: "auto", borderRadius: "6px" }}
            />
          </div>
        </div>
      </div>

      {/* Model Limitations Card */}
      <div className="card" style={{ background: "#fffbeb", border: "1px solid #fef3c7" }}>
        <h3 className="card-title" style={{ color: "#92400e" }}>
          ⚠️ Critical Clinical & Technical Limitations
        </h3>
        <ul style={{ paddingLeft: "20px", fontSize: "13px", color: "#78350f", lineHeight: 1.7 }}>
          <li>
            <strong>Training Scope:</strong> Initial 5-epoch training protocol on a binary classification dataset (melanoma vs benign). Does not cover non-melanoma carcinomas (e.g. basal cell or squamous cell carcinomas).
          </li>
          <li>
            <strong>Environmental Sensitivity:</strong> Glare, uneven illumination, camera motion blur, or digital compression artifacts can reduce classification confidence.
          </li>
          <li>
            <strong>Approximation of Overlays:</strong> OpenCV spatial skin segmentation provides an approximate targeting HUD and morphological visual aid, not histological margin delineation.
          </li>
          <li>
            <strong>Regulatory Status:</strong> This software is strictly an academic research prototype and is not intended to replace formal dermatoscopic evaluation or biopsy.
          </li>
        </ul>
      </div>
    </div>
  );
}
