"""
Application Flask de prédiction du GPA.

Charge le modèle enregistré comme "champion" dans le MLflow Model Registry
(produit par src/evaluate.py) et l'expose via :
  - GET  /            page HTML avec un formulaire
  - POST /predict     endpoint JSON : {StudyTimeWeekly, Age, Absences} -> GPA
  - GET  /health      statut du service

Usage:
    python app/app.py                 # http://localhost:8000
    MLFLOW_TRACKING_URI=http://<ip>:5000 PORT=8000 python app/app.py
"""

import os

import mlflow
import pandas as pd
from flask import Flask, jsonify, render_template, request
from mlflow.tracking import MlflowClient

FEATURES = ["StudyTimeWeekly", "Age", "Absences"]
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
REGISTERED_MODEL_NAME = "student_gpa_model"
MODEL_ALIAS = "champion"

app = Flask(__name__)

# Chargé une seule fois au démarrage.
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
_client = MlflowClient()
_version = _client.get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS)
MODEL = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}")
MODEL_NAME = _client.get_run(_version.run_id).data.params.get("model_type", "unknown")
METRICS = _client.get_run(_version.run_id).data.metrics
print(
    f"[app] Modele charge : {REGISTERED_MODEL_NAME}@{MODEL_ALIAS} "
    f"(v{_version.version}, type={MODEL_NAME}) | metrics={METRICS}"
)


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
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
