import React, { useState } from "react";
import TreeCard from "../components/TreeCard";
import TreeDetailModal from "../components/TreeDetailModal";
import PlantImage from "../components/PlantImage";
import { plantData } from "../data/plants";
import { generatePDFReport } from "../api/pdfReport";
import "./ResultsPage.css";

export default function ResultsPage({ results, onNavigate }) {
  const [detailPlant, setDetailPlant] = useState(null);
  const [pdfLoading, setPdfLoading] = useState(false);
  const [pdfDone, setPdfDone]       = useState(false);

  if (!results) { onNavigate("predict"); return null; }

  const { ranked, selectedTree, city, climate, soilData } = results;
  const selectedResult = ranked.find((r) => r.plant === selectedTree);
  const info = plantData[selectedTree];

  const summaryFields = [
    { label: "City",           value: city },
    { label: "Soil Type",      value: soilData.soil_type },
    { label: "pH Value",       value: soilData.pH.toFixed(2) },
    { label: "Temperature",    value: `${climate.avg_temp}°C` },
    { label: "Rainfall",       value: `${climate.rainfall} mm/year` },
    { label: "Nitrogen (N)",   value: `${soilData.nitrogen} kg/ha` },
    { label: "Phosphorus (P)", value: `${soilData.phosphorus} kg/ha` },
    { label: "Potassium (K)",  value: `${soilData.potassium} kg/ha` },
    { label: "Organic Carbon", value: `${soilData.organic_carbon}%` },
    { label: "EC",             value: `${soilData.ec} dS/m` },
  ];

  const handleDownload = async () => {
    setPdfLoading(true);
    setPdfDone(false);
    try {
      await generatePDFReport(results);
      setPdfDone(true);
      setTimeout(() => setPdfDone(false), 3500);
    } catch (err) {
      console.error("PDF generation failed:", err);
      alert("PDF generation failed. Please try again.");
    } finally {
      setPdfLoading(false);
    }
  };

  return (
    <div className="results">
      <div className="results__container">

        {/* Header */}
        <div className="results__topbar animate-fadeInUp">
          <div>
            <h1 className="results__title">Tree Survival Prediction &amp; Recommendations</h1>
            <p className="results__subtitle">
              Based on your soil and environmental data for <strong>{city}</strong>
            </p>
          </div>
          <button
            className={`btn btn--pdf btn--md${pdfLoading ? " btn--loading" : ""}${pdfDone ? " btn--done" : ""}`}
            onClick={handleDownload}
            disabled={pdfLoading}
          >
            {pdfLoading ? (
              <><span className="spinner" />Generating PDF…</>
            ) : pdfDone ? (
              <>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                  <path d="M20 6L9 17l-5-5" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                Downloaded!
              </>
            ) : (
              <>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                  <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"
                    stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
                Download PDF Report
              </>
            )}
          </button>
        </div>

        {/* Selected tree hero with real image */}
        {selectedResult && (
          <div className="selected-hero animate-scaleIn delay-1">
            <div className="selected-hero__img-wrap">
              <PlantImage plant={selectedTree} className="selected-hero__img" alt={selectedTree} />
              <div className="selected-hero__img-overlay" />
            </div>
            <div className="selected-hero__content">
              <div className="selected-hero__left">
                <span className="selected-hero__label">Your Selected Tree</span>
                <h2 className="selected-hero__name">{selectedTree}</h2>
                <span className="selected-hero__sci">{info.scientificName}</span>
                <div className="selected-hero__benefits">
                  {info.benefits.map((b) => (
                    <span key={b} className="benefit-tag">{b}</span>
                  ))}
                </div>
              </div>
              <div className="selected-hero__right">
                <span className="selected-hero__pct-label">Predicted Survival Rate</span>
                <div className="selected-hero__pct">
                  {selectedResult.probability.toFixed(0)}%<span className="selected-hero__arrow"> ↗</span>
                </div>
                <span className="selected-hero__compat">
                  {selectedResult.probability >= 85 ? "✓ Excellent compatibility"
                    : selectedResult.probability >= 70 ? "✓ Good potential"
                    : "⚡ Moderate compatibility"}
                </span>
                <span className="selected-hero__rank">Ranked #{selectedResult.rank} of {ranked.length}</span>
              </div>
            </div>
          </div>
        )}

        {/* Soil Summary */}
        <div className="summary-card animate-fadeInUp delay-2">
          <h3 className="summary-card__heading">Your Soil &amp; Environmental Data Summary</h3>
          <div className="summary-card__grid">
            {summaryFields.map((f) => (
              <div key={f.label} className="summary-field">
                <span className="summary-field__label">{f.label}</span>
                <span className="summary-field__value">{f.value}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Rankings */}
        <div className="results__recs animate-fadeInUp delay-3">
          <h2 className="results__recs-title">All Tree Recommendations (Sorted by Survival Rate)</h2>
          <p className="results__recs-sub">Trees ranked in descending order of predicted survival rate for your conditions</p>
          <div className="results__grid">
            {ranked.map((item) => (
              <TreeCard
                key={item.plant}
                plant={item.plant}
                probability={item.probability}
                rank={item.rank}
                isSelected={item.plant === selectedTree}
                onViewDetails={setDetailPlant}
              />
            ))}
          </div>
        </div>

        {/* Methodology */}
        <div className="calc-card animate-fadeInUp delay-4">
          <h3 className="calc-card__heading">📊 How We Calculate Survival Rates</h3>
          <p className="calc-card__body">
            Our AI model (Random Forest Classifier, 300 estimators) analyzes soil composition
            (type, pH), nutrient content (NPK), organic carbon, electrical conductivity (EC), and
            city-specific climate data. Rates ≥85% = Excellent compatibility. 70–85% = Good
            potential with proper care. Trained on 3,200 botanically-accurate synthetic samples.
          </p>
        </div>

        <div className="results__cta">
          <button className="btn btn--outline btn--md" onClick={() => onNavigate("predict")}>
            ← Run Another Prediction
          </button>
        </div>
      </div>

      {detailPlant && (
        <TreeDetailModal plant={detailPlant} onClose={() => setDetailPlant(null)} />
      )}
    </div>
  );
}
