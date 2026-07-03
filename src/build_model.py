"""
Construction du modèle "servable" (joblib).

Entraîne un modèle de régression pour prédire le GPA à partir de
StudyTimeWeekly, Age et Absences, puis sérialise le meilleur modèle dans
models/model.joblib pour qu'il soit consommé directement par l'app Flask.

Ce script est autonome : il ne dépend ni de MLflow ni de DVC. Il lit le
dataset brut, applique la même sélection de features que preprocess, et
compare quelques modèles pour ne garder que le meilleur (R2 sur le test).

Usage:
    python src/build_model.py
    python src/build_model.py --input data/raw/Student_performance_data.csv \
        --output models/model.joblib
"""

import argparse
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

FEATURES = ["StudyTimeWeekly", "Age", "Absences"]
TARGET = "GPA"

# Modèles candidats : on garde celui qui a le meilleur R2 sur le jeu de test.
CANDIDATES = {
    "linear": LinearRegression(),
    "random_forest": RandomForestRegressor(
        n_estimators=200, max_depth=8, random_state=42
    ),
}


def load_data(input_path: str) -> pd.DataFrame:
    print(f"[build] Lecture des données depuis {input_path}")
    df = pd.read_csv(input_path)
    df = df[FEATURES + [TARGET]].dropna()
    print(f"[build] {len(df)} lignes utilisables après nettoyage")
    return df


def evaluate(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    return {
        "r2": r2_score(y_test, pred),
        "mae": mean_absolute_error(y_test, pred),
        "rmse": rmse,
    }


def main(input_path: str, output_path: str, test_size: float, random_state: int):
    df = load_data(input_path)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    best_name, best_model, best_metrics = None, None, None
    for name, model in CANDIDATES.items():
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        print(
            f"[build] {name:14s} -> R2={metrics['r2']:.4f}  "
            f"MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}"
        )
        if best_metrics is None or metrics["r2"] > best_metrics["r2"]:
            best_name, best_model, best_metrics = name, model, metrics

    print(f"[build] Meilleur modèle : {best_name} (R2={best_metrics['r2']:.4f})")

    # On réentraîne le meilleur modèle sur toutes les données pour le service.
    best_model.fit(X, y)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    payload = {
        "model": best_model,
        "features": FEATURES,
        "target": TARGET,
        "model_name": best_name,
        "metrics": best_metrics,
    }
    joblib.dump(payload, output_path)
    print(f"[build] Modèle sérialisé -> {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/Student_performance_data.csv")
    parser.add_argument("--output", default="models/model.joblib")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    main(args.input, args.output, args.test_size, args.random_state)
