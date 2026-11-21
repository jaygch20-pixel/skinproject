import React, { useState } from "react";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import Home from "./pages/Home";
import Assessment from "./pages/Assessment";
import ARExperience from "./pages/ARExperience";
import Results from "./pages/Results";
import ModelInfo from "./pages/ModelInfo";
import AboutProject from "./pages/AboutProject";

export default function App() {
  const [activeTab, setActiveTab] = useState("home");
  const [analysisData, setAnalysisData] = useState(null);

  const handleAnalysisComplete = (data) => {
    setAnalysisData(data);
  };

  return (
    <div className="app-container">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />
      
      <main className="main-content">
        {activeTab === "home" && <Home setActiveTab={setActiveTab} />}
        {activeTab === "assessment" && (
          <Assessment
            onAnalysisComplete={handleAnalysisComplete}
            setActiveTab={setActiveTab}
          />
        )}
        {activeTab === "ar" && (
          <ARExperience
            onAnalysisComplete={handleAnalysisComplete}
            setActiveTab={setActiveTab}
          />
        )}
        {activeTab === "results" && (
          <Results
            analysisData={analysisData}
            setActiveTab={setActiveTab}
          />
        )}
        {activeTab === "model" && <ModelInfo />}
        {activeTab === "about" && <AboutProject />}
      </main>

      <Footer />
    </div>
  );
}
