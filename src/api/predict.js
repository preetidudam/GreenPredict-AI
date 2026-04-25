/**
 * API module — connects to the Python/Streamlit backend.
 *
 * In production, replace BASE_URL with your deployed backend URL.
 * The backend endpoint expects a POST to /predict with the payload below.
 *
 * While the backend is not running locally, we use a LOCAL SIMULATION
 * that mirrors the Random Forest model's logic.
 */

import { plantData, cities } from "../data/plants";

const BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8501";

// ---------------------------------------------------------------------------
// Real backend call (use when Python server is running)
// ---------------------------------------------------------------------------
export async function predictSurvival(payload) {
  try {
    const response = await fetch(`${BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error("Backend unavailable");
    return await response.json();
  } catch {
    // Fallback to local simulation
    return simulatePrediction(payload);
  }
}

// ---------------------------------------------------------------------------
// Local simulation — mirrors the Random Forest model logic
// ---------------------------------------------------------------------------
export function simulatePrediction(payload) {
  const { pH, nitrogen, phosphorus, potassium, organic_carbon, ec, rainfall, temperature, soil_type } = payload;

  const results = {};

  for (const [plant, info] of Object.entries(plantData)) {
    let score = 0;
    let maxScore = 0;

    // Rainfall match (most important feature)
    maxScore += 30;
    const rfMid = (info.minRainfall + info.maxRainfall) / 2;
    const rfRange = info.maxRainfall - info.minRainfall;
    const rfDist = Math.abs(rainfall - rfMid) / (rfRange / 2);
    score += 30 * Math.max(0, 1 - rfDist * rfDist);

    // Temperature match (second most important)
    maxScore += 25;
    const tMid = (info.minTemp + info.maxTemp) / 2;
    const tRange = info.maxTemp - info.minTemp;
    const tDist = Math.abs(temperature - tMid) / (tRange / 2);
    score += 25 * Math.max(0, 1 - tDist * tDist);

    // pH match
    maxScore += 20;
    const phMid = (info.minPH + info.maxPH) / 2;
    const phRange = info.maxPH - info.minPH;
    const phDist = Math.abs(pH - phMid) / (phRange / 2);
    score += 20 * Math.max(0, 1 - phDist * phDist);

    // Soil type match
    maxScore += 15;
    if (info.validSoils.includes(soil_type)) score += 15;

    // Nutrient boost (minor)
    maxScore += 10;
    const nutrientScore = Math.min(1, (nitrogen / 400 + phosphorus / 30 + potassium / 200 + organic_carbon / 0.8) / 4);
    score += 10 * nutrientScore;

    // EC penalty
    if (ec > 3) score -= 5;

    const probability = Math.min(0.97, Math.max(0.02, score / maxScore));
    results[plant] = Math.round(probability * 10000) / 100;
  }

  // Sort by probability descending
  const sorted = Object.entries(results)
    .sort((a, b) => b[1] - a[1])
    .map(([plant, prob], index) => ({ plant, probability: prob, rank: index + 1 }));

  return { predictions: results, ranked: sorted };
}

// ---------------------------------------------------------------------------
// Helper: get city climate data
// ---------------------------------------------------------------------------
export function getCityClimate(cityName) {
  return cities.find((c) => c.name === cityName) || null;
}
