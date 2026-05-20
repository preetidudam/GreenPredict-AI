"""
flask_api.py — GreenPredict-AI Flask Backend
=============================================
Exposes POST /predict for the React frontend.
Streamlit (app.py) is completely untouched and still works independently.

Run:
    python flask_api.py

Then React calls: http://localhost:5000/predict
"""

import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

import sys

# RAG chatbot module lives in the RAG/ subfolder — loaded lazily at first /chat call
_rag = None

def get_rag():
    global _rag
    if _rag is None:
        # Add RAG/ to path so `import rag_chat` resolves to RAG/rag_chat.py
        rag_dir = os.path.join(os.path.dirname(__file__), "RAG")
        if rag_dir not in sys.path:
            sys.path.insert(0, rag_dir)
        import rag_chat
        _rag = rag_chat
    return _rag


# App setup

app = Flask(__name__)
CORS(app)  # Allow React dev server (localhost:3000) to call this API


# Load model once at startup (same file Streamlit uses)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "random_forest.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print(f"[OK] Model loaded from {MODEL_PATH}")
    print(f"     Classes : {list(model.classes_)}")
    print(f"     Features: {list(model.feature_names_in_)}")
except Exception as exc:
    raise RuntimeError(f"[ERROR] Could not load model from {MODEL_PATH}: {exc}") from exc


# Validation ranges (must match frontend)

VALID_RANGES = {
    "pH":             (0.0,  10.0),
    "nitrogen":       (0.0, 700.0),
    "phosphorus":     (0.0,  60.0),
    "potassium":      (0.0, 400.0),
    "organic_carbon": (0.0,   2.0),
    "ec":             (0.0,   4.0),
    # rainfall & temperature are derived from city — no strict cap needed here
}

REQUIRED_FIELDS = [
    "pH", "nitrogen", "phosphorus", "potassium",
    "organic_carbon", "ec", "rainfall", "temperature", "soil_type",
]

VALID_SOIL_TYPES = {"Sandy", "Loamy", "Alluvial", "Lateritic", "Red loam"}


# Helper: validate incoming payload

# Human-readable field labels for error messages
FIELD_LABELS = {
    "pH":             "pH",
    "nitrogen":       "Nitrogen",
    "phosphorus":     "Phosphorus",
    "potassium":      "Potassium",
    "organic_carbon": "Organic Carbon",
    "ec":             "EC",
}


def validate_payload(data: dict):
    """Return (True, None) if valid, else (False, error_message)."""
    # 1. Check all required fields are present
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return False, f"Missing required fields: {', '.join(missing)}"

    # 2. Validate numeric ranges (soil parameters only)
    for field, (lo, hi) in VALID_RANGES.items():
        raw = data[field]
        try:
            val = float(raw)
        except (TypeError, ValueError):
            label = FIELD_LABELS.get(field, field)
            return False, f"{label} must be a valid number."
        if not (lo <= val <= hi):
            label = FIELD_LABELS.get(field, field)
            # Format integers without decimals (e.g. 700 not 700.0)
            lo_str = int(lo) if lo == int(lo) else lo
            hi_str = int(hi) if hi == int(hi) else hi
            return False, f"{label} must be between {lo_str} and {hi_str}"

    # 3. Validate soil_type
    if data["soil_type"] not in VALID_SOIL_TYPES:
        return False, (
            f"Invalid soil type '{data['soil_type']}'. "
            f"Allowed: {', '.join(sorted(VALID_SOIL_TYPES))}"
        )

    return True, None


# POST /predict

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True)

    if not data:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    # --- Validate (MUST pass before model is touched) ---
    ok, err_msg = validate_payload(data)
    if not ok:
        return jsonify({"error": err_msg}), 400

    # --- Build DataFrame (mirrors exactly what Streamlit does) ---
    input_df = pd.DataFrame([{
        "pH":             float(data["pH"]),
        "nitrogen":       float(data["nitrogen"]),
        "phosphorus":     float(data["phosphorus"]),
        "potassium":      float(data["potassium"]),
        "organic_carbon": float(data["organic_carbon"]),
        "ec":             float(data["ec"]),
        "rainfall":       float(data["rainfall"]),
        "temperature":    float(data["temperature"]),
        "soil_type":      data["soil_type"],
    }])

    # One-hot encode soil_type and align to training columns
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    # --- Predict ---
    probabilities = model.predict_proba(input_df)[0]
    classes       = model.classes_

    prob_dict = {cls: round(float(prob) * 100, 2) for cls, prob in zip(classes, probabilities)}

    # Ranked list (descending probability)
    ranked = [
        {"plant": plant, "probability": prob, "rank": idx + 1}
        for idx, (plant, prob) in enumerate(
            sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        )
    ]

    return jsonify({
        "predictions": prob_dict,
        "ranked":      ranked,
    }), 200


# POST /chat  — RAG chatbot
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True)
    if not data or not data.get("question", "").strip():
        return jsonify({"error": "Please provide a question."}), 400

    question = data["question"].strip()
    if len(question) > 500:
        return jsonify({"error": "Question is too long."}), 400

    try:
        rag = get_rag()
        answer = rag.answer_question(question)
        return jsonify({"answer": answer}), 200
    except Exception as exc:
        err_str = str(exc)
        print(f"[CHAT ERROR] {type(exc).__name__}: {exc}")
        if (
            "429" in err_str
            or "RESOURCE_EXHAUSTED" in err_str
            or type(exc).__name__ == "RateLimitError"
        ):
            return jsonify({
                "error": "The AI is busy right now. Please wait a moment and try again.",
                "rate_limited": True,
            }), 429
        return jsonify({"error": "Unable to process your question. Please try again."}), 500



# Health check

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "random_forest.pkl"}), 200



# Entry point
if __name__ == "__main__":
    print("\n[GreenPredict-AI] Flask API starting on http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
