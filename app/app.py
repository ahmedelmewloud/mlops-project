"""
Application Flask de prédiction du GPA.

Charge le modèle sérialisé (models/model.joblib, produit par
src/build_model.py) et l'expose via :
  - GET  /            page HTML avec un formulaire
  - POST /predict     endpoint JSON : {StudyTimeWeekly, Age, Absences} -> GPA
  - GET  /health      statut du service

Usage:
    python app/app.py                 # http://localhost:8000
    MODEL_PATH=models/model.joblib PORT=8000 python app/app.py
"""

import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.environ.get(
    "MODEL_PATH", os.path.join(PROJECT_ROOT, "models", "model.joblib")
)

app = Flask(__name__)

# Chargé une seule fois au démarrage.
_payload = joblib.load(MODEL_PATH)
MODEL = _payload["model"]
FEATURES = _payload["features"]
MODEL_NAME = _payload.get("model_name", "unknown")
METRICS = _payload.get("metrics", {})
print(f"[app] Modèle chargé : {MODEL_NAME} | features={FEATURES} | metrics={METRICS}")


@app.route("/")
def index():
    return render_template("index.html", features=FEATURES, model_name=MODEL_NAME, metrics=METRICS)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or request.form
    try:
        row = {f: float(data[f]) for f in FEATURES}
    except (KeyError, TypeError, ValueError):
        return (
            jsonify({"error": f"Champs numériques requis : {', '.join(FEATURES)}"}),
            400,
        )

    gpa = float(MODEL.predict(pd.DataFrame([row]))[0])
    # Le GPA est borné entre 0 et 4.
    gpa = max(0.0, min(4.0, gpa))
    return jsonify({"gpa": round(gpa, 3), "inputs": row, "model": MODEL_NAME})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME, "features": FEATURES})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
