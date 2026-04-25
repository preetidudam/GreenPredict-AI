import React, { useState } from "react";
import { cities, soilTypes, plantNames } from "../data/plants";
import { simulatePrediction, getCityClimate } from "../api/predict";
import "./PredictPage.css";

const initialForm = {
  city: "",
  soil_type: "",
  pH: "",
  nitrogen: "",
  phosphorus: "",
  potassium: "",
  organic_carbon: "",
  ec: "",
  tree: "",
};

const tips = [
  "Select your city to auto-populate rainfall and temperature data",
  "Choose the specific tree you plan to plant to get its survival prediction",
  "Ensure all measurements are accurate for best predictions",
  "pH values typically range from 6.0 (acidic) to 8.0 (alkaline)",
  "Use soil testing kits for precise NPK and EC values",
];

export default function PredictPage({ onNavigate, onResults }) {
  const [form, setForm] = useState(initialForm);
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState({});

  const update = (field, val) => {
    setForm((f) => ({ ...f, [field]: val }));
    setErrors((e) => ({ ...e, [field]: "" }));
  };

  const validate = () => {
    const errs = {};
    if (!form.city)         errs.city = "Please select a city";
    if (!form.soil_type)    errs.soil_type = "Please select soil type";
    if (!form.pH)           errs.pH = "Required";
    if (!form.nitrogen)     errs.nitrogen = "Required";
    if (!form.phosphorus)   errs.phosphorus = "Required";
    if (!form.potassium)    errs.potassium = "Required";
    if (!form.organic_carbon) errs.organic_carbon = "Required";
    if (!form.ec)           errs.ec = "Required";
    if (!form.tree)         errs.tree = "Please select a tree";
    return errs;
  };

  const handleSubmit = async () => {
    const errs = validate();
    if (Object.keys(errs).length) { setErrors(errs); return; }

    setLoading(true);
    const climate = getCityClimate(form.city);
    const payload = {
      pH: parseFloat(form.pH),
      nitrogen: parseFloat(form.nitrogen),
      phosphorus: parseFloat(form.phosphorus),
      potassium: parseFloat(form.potassium),
      organic_carbon: parseFloat(form.organic_carbon),
      ec: parseFloat(form.ec),
      rainfall: climate.rainfall,
      temperature: climate.avg_temp,
      soil_type: form.soil_type,
    };

    // Slight delay to show loading state
    await new Promise((r) => setTimeout(r, 900));
    const result = simulatePrediction(payload);

    setLoading(false);
    onResults({
      ...result,
      selectedTree: form.tree,
      city: form.city,
      climate,
      soilData: payload,
    });
    onNavigate("results");
  };

  return (
    <div className="predict">
      <div className="predict__container">
        <button className="back-btn" onClick={() => onNavigate("home")}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
            <path d="M19 12H5M5 12l7 7M5 12l7-7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          Back to Home
        </button>

        <div className="predict__header animate-fadeInUp">
          <span className="predict__icon">🌿</span>
          <h1 className="predict__title">Enter Soil &amp; Environmental Data</h1>
          <p className="predict__subtitle">
            Provide detailed information about your soil and environmental conditions
          </p>
        </div>

        <div className="predict__form-card animate-scaleIn delay-1">
          <div className="form-grid">
            {/* City */}
            <div className={`form-group ${errors.city ? "form-group--error" : ""}`}>
              <label className="form-label">City/Location</label>
              <select className="form-select" value={form.city} onChange={(e) => update("city", e.target.value)}>
                <option value="">Select city</option>
                {cities.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
              </select>
              {errors.city && <span className="form-error">{errors.city}</span>}
            </div>

            {/* Potassium */}
            <div className={`form-group ${errors.potassium ? "form-group--error" : ""}`}>
              <label className="form-label">Potassium (K) – kg/ha</label>
              <input className="form-input" type="number" placeholder="e.g., 200"
                value={form.potassium} onChange={(e) => update("potassium", e.target.value)} />
              {errors.potassium && <span className="form-error">{errors.potassium}</span>}
            </div>

            {/* Soil Type */}
            <div className={`form-group ${errors.soil_type ? "form-group--error" : ""}`}>
              <label className="form-label">Soil Type</label>
              <select className="form-select" value={form.soil_type} onChange={(e) => update("soil_type", e.target.value)}>
                <option value="">Select soil type</option>
                {soilTypes.map((s) => <option key={s} value={s}>{s}</option>)}
              </select>
              {errors.soil_type && <span className="form-error">{errors.soil_type}</span>}
            </div>

            {/* Organic Carbon */}
            <div className={`form-group ${errors.organic_carbon ? "form-group--error" : ""}`}>
              <label className="form-label">Organic Carbon (%)</label>
              <input className="form-input" type="number" step="0.1" placeholder="e.g., 0.8"
                value={form.organic_carbon} onChange={(e) => update("organic_carbon", e.target.value)} />
              {errors.organic_carbon && <span className="form-error">{errors.organic_carbon}</span>}
            </div>

            {/* pH */}
            <div className={`form-group ${errors.pH ? "form-group--error" : ""}`}>
              <label className="form-label">pH Value (0–14)</label>
              <input className="form-input" type="number" step="0.1" placeholder="e.g., 6.5"
                value={form.pH} onChange={(e) => update("pH", e.target.value)} />
              {errors.pH && <span className="form-error">{errors.pH}</span>}
            </div>

            {/* EC */}
            <div className={`form-group ${errors.ec ? "form-group--error" : ""}`}>
              <label className="form-label">EC (Electrical Conductivity) – dS/m</label>
              <input className="form-input" type="number" step="0.1" placeholder="e.g., 2.5"
                value={form.ec} onChange={(e) => update("ec", e.target.value)} />
              {errors.ec && <span className="form-error">{errors.ec}</span>}
            </div>

            {/* Nitrogen */}
            <div className={`form-group ${errors.nitrogen ? "form-group--error" : ""}`}>
              <label className="form-label">Nitrogen (N) – kg/ha</label>
              <input className="form-input" type="number" placeholder="e.g., 400"
                value={form.nitrogen} onChange={(e) => update("nitrogen", e.target.value)} />
              {errors.nitrogen && <span className="form-error">{errors.nitrogen}</span>}
            </div>

            {/* Rainfall (auto) */}
            <div className="form-group">
              <label className="form-label">Rainfall (mm/year)</label>
              <input className="form-input form-input--auto" type="text" readOnly
                value={form.city ? `${getCityClimate(form.city)?.rainfall ?? ""} mm/year` : ""}
                placeholder="Auto-populated from city" />
            </div>

            {/* Phosphorus */}
            <div className={`form-group ${errors.phosphorus ? "form-group--error" : ""}`}>
              <label className="form-label">Phosphorus (P) – kg/ha</label>
              <input className="form-input" type="number" placeholder="e.g., 30"
                value={form.phosphorus} onChange={(e) => update("phosphorus", e.target.value)} />
              {errors.phosphorus && <span className="form-error">{errors.phosphorus}</span>}
            </div>

            {/* Temperature (auto) */}
            <div className="form-group">
              <label className="form-label">Temperature (°C)</label>
              <input className="form-input form-input--auto" type="text" readOnly
                value={form.city ? `${getCityClimate(form.city)?.avg_temp ?? ""}°C` : ""}
                placeholder="Auto-populated from city" />
            </div>
          </div>

          {/* Tree Selection — full width */}
          <div className={`form-group form-group--centered ${errors.tree ? "form-group--error" : ""}`}>
            <label className="form-label">Tree to Plant <span className="form-required">*</span></label>
            <select className="form-select form-select--tree" value={form.tree} onChange={(e) => update("tree", e.target.value)}>
              <option value="">Select tree to plant</option>
              {plantNames.map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
            {errors.tree && <span className="form-error">{errors.tree}</span>}
          </div>

          <button
            className="btn btn--primary btn--full btn--submit"
            onClick={handleSubmit}
            disabled={loading}
          >
            {loading ? (
              <>
                <span className="spinner" />
                Analyzing...
              </>
            ) : (
              "Get Tree Recommendations"
            )}
          </button>
        </div>

        {/* Tips */}
        <div className="tips-card animate-fadeInUp delay-3">
          <h3 className="tips-card__heading">
            <span>💡</span> Tips for Accurate Results
          </h3>
          <ul className="tips-card__list">
            {tips.map((t) => (
              <li key={t} className="tips-card__item">
                <span className="tips-card__bullet">•</span>
                {t}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
