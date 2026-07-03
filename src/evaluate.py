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
import json
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error, r2_score

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
MODEL_ALIAS = "champion"


def get_run_ids(args) -> list:
    if args.run_id:
        return [args.run_id]
    with open("metrics/last_run_ids.txt") as f:
        return [line.strip() for line in f if line.strip()]


def evaluate_run(client: MlflowClient, run_id: str, X_test, y_test) -> dict:
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    pred = model.predict(X_test)
    metrics = {
        "test_rmse": root_mean_squared_error(y_test, pred),
        "test_mae": mean_absolute_error(y_test, pred),
        "test_r2": r2_score(y_test, pred),
    }
    for key, value in metrics.items():
        client.log_metric(run_id, key, value)
    print(f"[evaluate] run {run_id} -> " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return metrics


def get_current_champion_rmse(client: MlflowClient) -> float | None:
    try:
        champion = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS)
    except mlflow.exceptions.MlflowException:
        return None
    run = client.get_run(champion.run_id)
    return run.data.metrics.get("test_rmse")


def evaluate(args):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    test_df = pd.read_csv(args.test_data)
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    run_ids = get_run_ids(args)
    results = {}
    best_run_id, best_rmse = None, None
    for run_id in run_ids:
        metrics = evaluate_run(client, run_id, X_test, y_test)
        results[run_id] = metrics
        if best_rmse is None or metrics["test_rmse"] < best_rmse:
            best_run_id, best_rmse = run_id, metrics["test_rmse"]

    champion_rmse = get_current_champion_rmse(client)
    registered = False
    if champion_rmse is None or best_rmse < champion_rmse:
        model_uri = f"runs:/{best_run_id}/model"
        model_version = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS, model_version.version)
        registered = True
        print(
            f"[evaluate] New champion: run {best_run_id} (test_rmse={best_rmse:.4f}) "
            f"registered as {REGISTERED_MODEL_NAME} v{model_version.version}"
        )
    else:
        print(
            f"[evaluate] Best of batch (test_rmse={best_rmse:.4f}) did not beat "
            f"current champion (test_rmse={champion_rmse:.4f}) -- not registered"
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(
            {
                "runs": results,
                "best_run_id": best_run_id,
                "best_test_rmse": best_rmse,
                "registered": registered,
            },
            f,
            indent=2,
        )
    print(f"[evaluate] Metrics written -> {args.output}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", default="data/processed/test.csv")
    parser.add_argument("--run-id", default=None, help="Evaluate a single run instead of the last batch")
    parser.add_argument("--output", default="metrics/eval.json")
    args = parser.parse_args()

    evaluate(args)