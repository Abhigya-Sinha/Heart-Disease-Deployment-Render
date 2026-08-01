"""
app.py
-------
Task 3: API Development

A Flask REST API that:
- Loads the trained model (model.pkl)
- Accepts patient clinical details as JSON input
- Returns the prediction as JSON, e.g. {"prediction": "Heart Disease Detected"}

Also serves a small, simple web UI (templates/index.html) at "/" so
graders/users can try predictions from a browser without needing curl/Postman.

Run locally:
    python app.py
Then open http://127.0.0.1:5000 in a browser.
"""

import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the trained model bundle once at startup
MODEL_PATH = "model.pkl"
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
scaler = bundle["scaler"]
FEATURE_ORDER = bundle["feature_order"]
TRAIN_ACCURACY = bundle["accuracy"]


def build_feature_vector(payload: dict):
    """Extract features from the incoming JSON in the exact order the
    model was trained on. Raises KeyError if a field is missing."""
    values = [float(payload[feature]) for feature in FEATURE_ORDER]
    return np.array(values).reshape(1, -1)


@app.route("/")
def home():
    return render_template("index.html", features=FEATURE_ORDER, accuracy=round(TRAIN_ACCURACY * 100, 2))


@app.route("/health")
def health():
    """Simple health check endpoint, useful for Render / uptime checks."""
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Please send patient details as JSON."}), 400

    missing = [f for f in FEATURE_ORDER if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    try:
        features = build_feature_vector(data)
    except (ValueError, TypeError):
        return jsonify({"error": "All fields must be numeric."}), 400

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected"

    return jsonify({
        "prediction": result,
        "risk_probability": round(float(probability), 4)
    })


if __name__ == "__main__":
    # Render sets the PORT environment variable; default to 5000 for local runs
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
