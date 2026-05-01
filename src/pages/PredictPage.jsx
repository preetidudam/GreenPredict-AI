import React, { useState, useCallback } from "react";
import { cities, soilTypes, plantNames } from "../data/plants";
import { predictSurvival, getCityClimate } from "../api/predict";
import "./PredictPage.css";

// ---------------------------------------------------------------------------
// Validation config — must match flask_api.py VALID_RANGES exactly
// ---------------------------------------------------------------------------
const NUMERIC_RULES = {
  pH:             { lo: 0,   hi: 10,  label: "pH",             msg: "pH must be between 0 and 10" },
  nitrogen:       { lo: 0,   hi: 700, label: "Nitrogen",       msg: "Nitrogen must be between 0 and 700" },
  phosphorus:     { lo: 0,   hi: 60,  label: "Phosphorus",     msg: "Phosphorus must be between 0 and 60" },
  potassium:      { lo: 0,   hi: 400, label: "Potassium",      msg: "Potassium must be between 0 and 400" },
  organic_carbon: { lo: 0,   hi: 2,   label: "Organic Carbon", msg: "Organic Carbon must be between 0 and 2" },
  ec:             { lo: 0,   hi: 4,   label: "EC",             msg: "EC must be between 0 and 4" },
};

const NUMERIC_FIELDS = Object.keys(NUMERIC_RULES);

// ---------------------------------------------------------------------------
// Pure validation — returns per-field error strings
// ---------------------------------------------------------------------------
function validateField(name, value) {
  if (NUMERIC_FIELDS.includes(name)) {
    if (value === "" || value === null || value === undefined) return "This field is required";
    const num = parseFloat(value);
    if (isNaN(num)) return "Must be a valid number";
    const { lo, hi, msg } = NUMERIC_RULES[name];
    if (num < lo || num > hi) return msg;
    return "";
  }
  if (name === "city")      return value ? "" : "Please select a city";
  if (name === "soil_type") return value ? "" : "Please select a soil type";
  if (name === "tree")      return value ? "" : "Please select a tree";
  return "";
}

function validateAll(form) {
  const errs = {};
  [...NUMERIC_FIELDS, "city", "soil_type", "tree"].forEach((name) => {
    const msg = validateField(name, form[name]);
    if (msg) errs[name] = msg;
  });
  return errs;
}

// ---------------------------------------------------------------------------
// Initial state
// ---------------------------------------------------------------------------
const initialForm = {
  city: "", soil_type: "",
  pH: "", nitrogen: "", phosphorus: "",
  potassium: "", organic_carbon: "", ec: "",
  tree: "",
};

const tips = [
  "Select your city to auto-populate rainfall and temperature data",
  "Choose the specific tree you plan to plant to get its survival prediction",
  "Ensure all measurements are accurate for best predictions",
  "pH values must be between 0 and 10 (typical soil: 6.0–8.0)",
  "Use soil testing kits for precise NPK and EC values",
];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export default function PredictPage({ onNavigate, onResults }) {
  const [form,    setForm]    = useState(initialForm);
  const [touched, setTouched] = useState({});   // tracks which fields have been blurred
  const [errors,  setErrors]  = useState({});   // per-field error strings (shown only for touched)
  const [loading, setLoading] = useState(false);
  const [apiError, setApiError] = useState("");

  // ── onChange: update value, clear error for that field, NO validation yet ──
  const handleChange = useCallback((field, val) => {
    setForm((f) => ({ ...f, [field]: val }));
    // If already touched, re-validate on change so feedback clears as soon as valid
    setTouched((t) => {
      if (t[field]) {
        setErrors((e) => ({ ...e, [field]: validateField(field, val) }));
      }
      return t;
    });
    setApiError("");
  }, []);

  // ── onBlur: mark touched and validate that single field ──
  const handleBlur = useCallback((field) => {
    setTouched((t) => ({ ...t, [field]: true }));
    setErrors((e) => ({ ...e, [field]: validateField(field, form[field]) }));
  }, [form]);

  // ── Derived: is the whole form valid (used to disable the button) ──
  const allErrors = validateAll(form);
  const isFormValid = Object.keys(allErrors).length === 0;

  // ── Visible errors: only show for touched fields ──
  const visibleErrors = {};
  Object.keys(errors).forEach((k) => {
    if (touched[k]) visibleErrors[k] = errors[k];
  });

  // ── Submit ──
  const handleSubmit = async () => {
    // On submit, mark everything touched and show all errors
    const allFields = [...NUMERIC_FIELDS, "city", "soil_type", "tree"];
    const touchAll = {};
    allFields.forEach((f) => { touchAll[f] = true; });
    setTouched(touchAll);

    const errs = validateAll(form);
    setErrors(errs);
    setApiError("");

    if (Object.keys(errs).length) return;   // stop — show inline errors

    setLoading(true);
    const climate = getCityClimate(form.city);
    const payload = {
      pH:             parseFloat(form.pH),
      nitrogen:       parseFloat(form.nitrogen),
      phosphorus:     parseFloat(form.phosphorus),
      potassium:      parseFloat(form.potassium),
      organic_carbon: parseFloat(form.organic_carbon),
      ec:             parseFloat(form.ec),
      rainfall:       climate.rainfall,
      temperature:    climate.avg_temp,
      soil_type:      form.soil_type,
    };

    try {
      const result = await predictSurvival(payload);
      setLoading(false);
      onResults({ ...result, selectedTree: form.tree, city: form.city, climate, soilData: payload });
      onNavigate("results");
    } catch (err) {
      setLoading(false);
      setApiError(err.message || "Prediction failed. Please check your inputs and try again.");
    }
  };

  // ── Helper: visible error for a field ──
  const fieldError = (name) => (touched[name] ? errors[name] || "" : "");

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

        {/* API / network error banner */}
        {apiError && (
          <div className="api-error-banner" role="alert">
            <strong>&#9888; Error:</strong> {apiError}
          </div>
        )}

        <div className="predict__form-card animate-scaleIn delay-1">
          <div className="form-grid">

            {/* City */}
            <div className={`form-group ${fieldError("city") ? "form-group--error" : ""}`}>
              <label className="form-label">City / Location</label>
              <select
                className="form-select"
                value={form.city}
                onChange={(e) => handleChange("city", e.target.value)}
                onBlur={() => handleBlur("city")}
              >
                <option value="">Select city</option>
                {cities.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
              </select>
              {fieldError("city") && <span className="form-error">{fieldError("city")}</span>}
            </div>

            {/* Potassium */}
            <div className={`form-group ${fieldError("potassium") ? "form-group--error" : ""}`}>
              <label className="form-label">Potassium (K) – kg/ha <span className="form-hint">(0–400)</span></label>
              <input
                className="form-input"
                type="number"
                placeholder="e.g., 200"
                value={form.potassium}
                onChange={(e) => handleChange("potassium", e.target.value)}
                onBlur={() => handleBlur("potassium")}
              />
              {fieldError("potassium") && <span className="form-error">{fieldError("potassium")}</span>}
            </div>

            {/* Soil Type */}
            <div className={`form-group ${fieldError("soil_type") ? "form-group--error" : ""}`}>
              <label className="form-label">Soil Type</label>
              <select
                className="form-select"
                value={form.soil_type}
                onChange={(e) => handleChange("soil_type", e.target.value)}
                onBlur={() => handleBlur("soil_type")}
              >
                <option value="">Select soil type</option>
                {soilTypes.map((s) => <option key={s} value={s}>{s}</option>)}
              </select>
              {fieldError("soil_type") && <span className="form-error">{fieldError("soil_type")}</span>}
            </div>

            {/* Organic Carbon */}
            <div className={`form-group ${fieldError("organic_carbon") ? "form-group--error" : ""}`}>
              <label className="form-label">Organic Carbon (%) <span className="form-hint">(0–2)</span></label>
              <input
                className="form-input"
                type="number"
                step="0.01"
                placeholder="e.g., 0.8"
                value={form.organic_carbon}
                onChange={(e) => handleChange("organic_carbon", e.target.value)}
                onBlur={() => handleBlur("organic_carbon")}
              />
              {fieldError("organic_carbon") && <span className="form-error">{fieldError("organic_carbon")}</span>}
            </div>

            {/* pH */}
            <div className={`form-group ${fieldError("pH") ? "form-group--error" : ""}`}>
              <label className="form-label">pH Value <span className="form-hint">(0–10)</span></label>
              <input
                className="form-input"
                type="number"
                step="0.1"
                placeholder="e.g., 6.5"
                value={form.pH}
                onChange={(e) => handleChange("pH", e.target.value)}
                onBlur={() => handleBlur("pH")}
              />
              {fieldError("pH") && <span className="form-error">{fieldError("pH")}</span>}
            </div>

            {/* EC */}
            <div className={`form-group ${fieldError("ec") ? "form-group--error" : ""}`}>
              <label className="form-label">EC (Electrical Conductivity) – dS/m <span className="form-hint">(0–4)</span></label>
              <input
                className="form-input"
                type="number"
                step="0.1"
                placeholder="e.g., 2.5"
                value={form.ec}
                onChange={(e) => handleChange("ec", e.target.value)}
                onBlur={() => handleBlur("ec")}
              />
              {fieldError("ec") && <span className="form-error">{fieldError("ec")}</span>}
            </div>

            {/* Nitrogen */}
            <div className={`form-group ${fieldError("nitrogen") ? "form-group--error" : ""}`}>
              <label className="form-label">Nitrogen (N) – kg/ha <span className="form-hint">(0–700)</span></label>
              <input
                className="form-input"
                type="number"
                placeholder="e.g., 400"
                value={form.nitrogen}
                onChange={(e) => handleChange("nitrogen", e.target.value)}
                onBlur={() => handleBlur("nitrogen")}
              />
              {fieldError("nitrogen") && <span className="form-error">{fieldError("nitrogen")}</span>}
            </div>

            {/* Rainfall (auto) */}
            <div className="form-group">
              <label className="form-label">Rainfall (mm/year)</label>
              <input
                className="form-input form-input--auto"
                type="text"
                readOnly
                value={form.city ? `${getCityClimate(form.city)?.rainfall ?? ""} mm/year` : ""}
                placeholder="Auto-populated from city"
              />
            </div>

            {/* Phosphorus */}
            <div className={`form-group ${fieldError("phosphorus") ? "form-group--error" : ""}`}>
              <label className="form-label">Phosphorus (P) – kg/ha <span className="form-hint">(0–60)</span></label>
              <input
                className="form-input"
                type="number"
                placeholder="e.g., 30"
                value={form.phosphorus}
                onChange={(e) => handleChange("phosphorus", e.target.value)}
                onBlur={() => handleBlur("phosphorus")}
              />
              {fieldError("phosphorus") && <span className="form-error">{fieldError("phosphorus")}</span>}
            </div>

            {/* Temperature (auto) */}
            <div className="form-group">
              <label className="form-label">Temperature (°C)</label>
              <input
                className="form-input form-input--auto"
                type="text"
                readOnly
                value={form.city ? `${getCityClimate(form.city)?.avg_temp ?? ""}°C` : ""}
                placeholder="Auto-populated from city"
              />
            </div>
          </div>

          {/* Tree Selection — full width */}
          <div className={`form-group form-group--centered ${fieldError("tree") ? "form-group--error" : ""}`}>
            <label className="form-label">Tree to Plant <span className="form-required">*</span></label>
            <select
              className="form-select form-select--tree"
              value={form.tree}
              onChange={(e) => handleChange("tree", e.target.value)}
              onBlur={() => handleBlur("tree")}
            >
              <option value="">Select tree to plant</option>
              {plantNames.map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
            {fieldError("tree") && <span className="form-error">{fieldError("tree")}</span>}
          </div>

          {/* Submit — disabled when form is invalid OR loading */}
          <button
            className="btn btn--primary btn--full btn--submit"
            onClick={handleSubmit}
            disabled={loading || !isFormValid}
            title={!isFormValid ? "Fill all fields with valid values to continue" : ""}
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

          {/* Subtle "form incomplete" hint shown before submit */}
          {!isFormValid && !loading && (
            <p className="form-incomplete-hint">
              Fill in all fields with values in the allowed ranges to enable prediction.
            </p>
          )}
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
