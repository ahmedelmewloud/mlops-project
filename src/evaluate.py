"""
Script d'évaluation du modèle.

Charge tous les modèles entraînés dans le dernier batch (run_ids écrits par
train.py dans metrics/last_run_ids.txt), les évalue sur le jeu de test,
logue les métriques finales de chacun dans MLflow, puis enregistre dans le
Model Registry SEULEMENT le meilleur des deux (comparaison sur le RMSE
test) : le meilleur du batch vs. le meilleur déjà enregistré.

Usage:
    python src/evaluate.py --run-id <run_id>
    # ou, si --run-id est omis, on lit metrics/last_run_ids.txt (écrit par train.py)
    python src/evaluate.py
"""

import argparse
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import r2_score

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:  # older scikit-learn versions
    from sklearn.metrics import mean_squared_error

    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred) ** 0.5
FEATURES = ["StudyTimeWeekly", "Age", "Absences"]
TARGET = "GPA"

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
REGISTERED_MODEL_NAME = "student_gpa_model"

def get_run_ids(args) -> list:
    if args.run_id:
        return [args.run_id]
    with open("metrics/last_run_ids.txt") as f:
        return [line.strip() for line in f if line.strip()]