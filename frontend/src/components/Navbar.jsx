import React from "react";

export default function Navbar({ activeTab, setActiveTab }) {
  const tabs = [
    { id: "home", label: "Home", icon: "🏥" },
    { id: "assessment", label: "Skin Assessment", icon: "🔬" },
    { id: "ar", label: "AR Experience", icon: "👓" },
    { id: "results", label: "Clinical Results", icon: "📋" },
    { id: "model", label: "Model & Evaluation", icon: "📊" },
    { id: "about", label: "About Project", icon: "ℹ️" },
  ];

  return (
    <header className="navbar">
      <div className="nav-container">
        <div className="brand" onClick={() => setActiveTab("home")} style={{ cursor: "pointer" }}>
          <div className="brand-icon">AR</div>
          <div className="brand-text">
            <h1>DermaVision AR</h1>
            <p>Computer Vision & Augmented Reality Skin Screening</p>
          </div>
        </div>
        <nav>
          <ul className="nav-links">
            {tabs.map((tab) => (
              <li key={tab.id}>
                <button
                  className={`nav-btn ${activeTab === tab.id ? "active" : ""}`}
                  onClick={() => setActiveTab(tab.id)}
                >
                  <span>{tab.icon}</span>
                  <span>{tab.label}</span>
                </button>
              </li>
            ))}
          </ul>
        </nav>
      </div>
    </header>
  );
}
