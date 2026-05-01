/**
 * API module — connects React to the Flask ML backend.
 *
 * Backend must be running:
 *   python flask_api.py   ->  http://localhost:5000
 *
 * Endpoint: POST /predict
 * Payload : { pH, nitrogen, phosphorus, potassium, organic_carbon, ec,
 *             rainfall, temperature, soil_type }
 * Response: { predictions: { [plant]: probability }, ranked: [...] }
 */

import { cities } from "../data/plants";

// ---------------------------------------------------------------------------
// Config — override with REACT_APP_API_URL env var for production
// ---------------------------------------------------------------------------
const BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:5000";

// ---------------------------------------------------------------------------
// Real backend call — NO fallback, fails loudly if backend is unavailable
// ---------------------------------------------------------------------------
/**
 * Sends soil + climate data to the Flask backend and returns ML predictions.
 * Throws an Error if the backend is unreachable or returns an error response.
 *
 * @param {object} payload  - { pH, nitrogen, phosphorus, potassium,
 *                              organic_carbon, ec, rainfall, temperature,
 *                              soil_type }
 * @returns {Promise<{ predictions: object, ranked: Array }>}
 */
export async function predictSurvival(payload) {
  let response;
  try {
    response = await fetch(`${BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch {
    throw new Error(
      "Cannot reach the prediction server. Make sure the Flask API is running: python flask_api.py"
    );
  }

  const json = await response.json();

  if (!response.ok) {
    // Relay the backend error message verbatim to the UI
    throw new Error(
      json.error || "Invalid input values. Please enter values within allowed range."
    );
  }

  return json; // { predictions, ranked }
}

// ---------------------------------------------------------------------------
// Helper: get city climate data
// ---------------------------------------------------------------------------
export function getCityClimate(cityName) {
  return cities.find((c) => c.name === cityName) || null;
}
