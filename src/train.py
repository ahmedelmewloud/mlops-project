"""
Script d'entraînement du modèle.

Entraîne un ou plusieurs modèles (définis dans params.yaml, ex :
LinearRegression, SVR avec différents hyperparamètres) sur les données
prétraitées, et logue chaque run dans MLflow : hyperparamètres, métriques
(sur train), et le modèle produit. Chaque run_id est écrit dans
metrics/last_run_ids.txt pour que evaluate.py compare les runs entre eux.

Usage:
    python src/train.py
    python src/train.py --params params.yaml --train-data data/processed/train.csv
"""

import argparse
import os

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

FEATURES = ["StudyTimeWeekly", "Age", "Absences"]
TARGET = "GPA"

# IMPORTANT : remplacez par l'IP publique de votre instance EC2 MLflow
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "student_performance_gpa_prediction"

MODEL_BUILDERS = {
    "linear": lambda p: LinearRegression(),
    "ridge": lambda p: Ridge(alpha=p.get("alpha", 1.0)),
    "lasso": lambda p: Lasso(alpha=p.get("alpha", 1.0)),
    "svr": lambda p: SVR(
        kernel=p.get("kernel", "linear"), C=p.get("C", 1.0), epsilon=p.get("epsilon", 0.1)
    ),
    "decision_tree": lambda p: DecisionTreeRegressor(
        max_depth=p.get("max_depth"),
        min_samples_leaf=p.get("min_samples_leaf", 1),
        random_state=42,
    ),
    "random_forest": lambda p: RandomForestRegressor(
        n_estimators=p.get("n_estimators", 100),
        max_depth=p.get("max_depth"),
        random_state=42,
    ),
    "knn": lambda p: KNeighborsRegressor(n_neighbors=p.get("n_neighbors", 5)),
}


def build_model(config: dict):
    model_type = config["type"]
    if model_type not in MODEL_BUILDERS:
        raise ValueError(f"Modèle inconnu : {model_type}")
    return MODEL_BUILDERS[model_type](config)


def train_one(config: dict, X_train, y_train) -> str:
    with mlflow.start_run() as run:
        # 1. Log des hyperparamètres (tous les champs du config, sauf "type")
        mlflow.log_param("model_type", config["type"])
        for key, value in config.items():
            if key != "type":
                mlflow.log_param(key, value)

        # 2. Entraînement
        model = build_model(config)
        model.fit(X_train, y_train)

        # 3. Métrique rapide sur le train (juste indicative, l'éval réelle
        #    se fait dans evaluate.py sur le jeu de test)
        train_r2 = model.score(X_train, y_train)
        mlflow.log_metric("train_r2", train_r2)
        print(f"[train] {config} -> Train R2 : {train_r2:.4f}")

        # 4. Log du modèle produit
        # serialization_format="pickle" : évite le rejet par MLflow des types
        # internes (ex. KDTree de KNeighborsRegressor) non "trusted" par le
        # format skops par défaut. Sûr ici car on ne charge que nos propres
        # modèles, jamais des artefacts externes non fiables.
        mlflow.sklearn.log_model(
            model, "model", serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE
        )

        return run.info.run_id
