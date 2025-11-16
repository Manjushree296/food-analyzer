import { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState(JSON.parse(localStorage.getItem("foodHistory")) || []);
  const [showHistory, setShowHistory] = useState(false);

  const COMMON_ALLERGENS = ["Nuts", "Milk", "Eggs", "Soy", "Wheat", "Shellfish", "Fish", "Sesame"];

  const handleUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setResult(null);
      
      // Create image preview
      const reader = new FileReader();
      reader.onload = (event) => {
        setImagePreview(event.target.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = async () => {
    if (!image) return alert("Please select an image.");
    const formData = new FormData();
    formData.append("image", image);

    setLoading(true);
    try {
      const res = await axios.post("http://127.0.0.1:5000/api/analyze", formData);
      setResult(res.data);
      
      // Add to history
      const newEntry = {
        id: Date.now(),
        timestamp: new Date().toLocaleString(),
        status: res.data.status,
        recommendation: res.data.recommendation.recommendation,
        image: imagePreview,
        harmfulCount: (res.data.harmful_high?.length || 0) + (res.data.harmful_medium?.length || 0)
      };
      const updatedHistory = [newEntry, ...history].slice(0, 10);
      setHistory(updatedHistory);
      localStorage.setItem("foodHistory", JSON.stringify(updatedHistory));
    } catch (err) {
      alert(err.response?.data?.error || "Server error");
    }
    setLoading(false);
  };

  const detectAllergens = () => {
    if (!result) return [];
    const text = result.ocr_text.toLowerCase();
    return COMMON_ALLERGENS.filter(allergen => 
      text.includes(allergen.toLowerCase()) || 
      text.includes(allergen.toLowerCase() + "s")
    );
  };

  const downloadResults = () => {
    if (!result) return;
    const data = {
      timestamp: new Date().toLocaleString(),
      status: result.status,
      recommendation: result.recommendation,
      harmfulIngredients: {
        high: result.harmful_high,
        medium: result.harmful_medium,
        low: result.harmful_low
      },
      allergens: detectAllergens(),
      ocrText: result.ocr_text
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `food_analysis_${Date.now()}.json`;
    a.click();
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem("foodHistory");
  };

  const loadFromHistory = (entry) => {
    setImagePreview(entry.image);
    setShowHistory(false);
  };

  const allergens = detectAllergens();

  const renderIngredients = (ingredients, title, className) => {
    if (!ingredients || ingredients.length === 0) return null;
    
    return (
      <div className={className}>
        <h4>{title}</h4>
        <ul className="ingredient-list">
          {ingredients.map((item, idx) => (
            <li key={idx} className={`ingredient-item ${item.risk.toLowerCase()}`}>
              <strong>{item.name}</strong>
              <p className="advice">{item.advice}</p>
            </li>
          ))}
        </ul>
      </div>
    );
  };

  return (
    <div className="app-wrapper">
      <div className="header">
        <h1 className="app-title">🍱 Food Ingredient Analyzer</h1>
        <p className="app-subtitle">Smart nutrition & safety analysis powered by AI</p>
      </div>

      <div className="container">
        <div className="main-content">
          {/* Upload Section */}
          <div className="upload-section">
            <div className="upload-box">
              <input 
                type="file" 
                id="file-input"
                accept="image/*" 
                onChange={handleUpload}
                className="file-input"
              />
              <label htmlFor="file-input" className="file-label">
                <span className="upload-icon">📷</span>
                <span className="upload-text">Click to upload or drag image</span>
                <span className="upload-hint">PNG, JPG up to 10MB</span>
              </label>
            </div>

            {/* Image Preview */}
            {imagePreview && (
              <div className="image-preview-container">
                <img src={imagePreview} alt="preview" className="image-preview" />
                <button className="clear-btn" onClick={() => { setImage(null); setImagePreview(null); setResult(null); }}>
                  ✕ Clear
                </button>
              </div>
            )}

            <button 
              onClick={handleSubmit} 
              disabled={loading || !image}
              className="analyze-btn"
            >
              {loading ? (
                <>
                  <span className="spinner"></span> Analyzing...
                </>
              ) : (
                <>🔍 Analyze Now</>
              )}
            </button>
          </div>

          {/* Results Section */}
          {result && (
            <div className="results-section">
              {/* Recommendation Card */}
              <div className={`recommendation-card ${result.recommendation.severity}`}>
                <h2>{result.recommendation.recommendation}</h2>
                <p>{result.recommendation.details}</p>
              </div>

              {/* Status Badge */}
              <h3 className={`status ${result.status.toLowerCase()}`}>
                Overall Status: <strong>{result.status}</strong>
              </h3>

              {/* Allergen Alerts */}
              {allergens.length > 0 && (
                <div className="allergen-alert">
                  <h4>⚠️ Allergen Alert</h4>
                  <div className="allergen-badges">
                    {allergens.map((allergen, idx) => (
                      <span key={idx} className="allergen-badge">{allergen}</span>
                    ))}
                  </div>
                </div>
              )}

              {/* High Risk Ingredients */}
              {renderIngredients(
                result.harmful_high,
                "🔴 HIGH RISK - Avoid Completely",
                "harmful-section high-risk"
              )}

              {/* Medium Risk Ingredients */}
              {renderIngredients(
                result.harmful_medium,
                "🟠 MEDIUM RISK - Consume in Moderation",
                "harmful-section medium-risk"
              )}

              {/* Low Risk Ingredients */}
              {renderIngredients(
                result.harmful_low,
                "🟡 LOW RISK - Generally Safe",
                "harmful-section low-risk"
              )}

              {/* Safe Message */}
              {!result.harmful_high?.length &&
                !result.harmful_medium?.length &&
                !result.harmful_low?.length && (
                  <div className="safe-section">
                    <p className="safe">✅ No harmful ingredients detected!</p>
                  </div>
                )}

              {/* OCR Text */}
              <div className="ocr-section">
                <h3>📄 Detected Ingredients (OCR)</h3>
                <p className="ocr-text">{result.ocr_text}</p>
              </div>

              {/* Action Buttons */}
              <div className="action-buttons">
                <button onClick={downloadResults} className="btn btn-download">
                  📥 Download Report
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Sidebar - History */}
        <div className="sidebar">
          <div className="sidebar-header">
            <h3>📜 History</h3>
            <button 
              className="toggle-history"
              onClick={() => setShowHistory(!showHistory)}
            >
              {showHistory ? "▼" : "▶"}
            </button>
          </div>

          {showHistory && (
            <>
              <div className="history-list">
                {history.length === 0 ? (
                  <p className="no-history">No history yet</p>
                ) : (
                  history.map((entry) => (
                    <div 
                      key={entry.id} 
                      className={`history-item ${entry.status.toLowerCase()}`}
                      onClick={() => loadFromHistory(entry)}
                    >
                      <div className="history-preview">
                        <img src={entry.image} alt="history" />
                      </div>
                      <div className="history-info">
                        <p className="history-status">{entry.status}</p>
                        <p className="history-time">{entry.timestamp.split(",")[0]}</p>
                        <p className="history-harmful">{entry.harmfulCount} issues</p>
                      </div>
                    </div>
                  ))
                )}
              </div>

              {history.length > 0 && (
                <button 
                  onClick={clearHistory}
                  className="btn btn-danger"
                >
                  🗑️ Clear History
                </button>
              )}
            </>
          )}
        </div>
      </div>

      {/* Footer */}
      <div className="footer">
        <p>Food Analyzer © 2025 | Built with ❤️ for your health</p>
      </div>
    </div>
  );
}

export default App;
