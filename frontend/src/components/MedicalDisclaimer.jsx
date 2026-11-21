import React from "react";

export default function MedicalDisclaimer() {
  return (
    <div className="medical-banner">
      <div className="banner-icon">⚠️</div>
      <div className="banner-text">
        <h4>Important Clinical Notice & Research Prototype Disclaimer</h4>
        <p>
          This application is an academic and engineering research prototype developed for educational evaluation of computer vision
          and augmented reality. It is <strong>NOT a certified medical diagnostic device</strong>. Model predictions reflect statistical
          similarity to training datasets, not clinical diagnostic certainty. Always seek prompt evaluation from a board-certified dermatologist
          for any skin lesion, mole, or cutaneous irregularity. Never delay medical consultation based on algorithmic outputs.
        </p>
      </div>
    </div>
  );
}
