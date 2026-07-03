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

def get_best_registered_rmse(client: MlflowClient) -> float:
    """Renvoie le RMSE test du modèle actuellement en production/latest.
    Si aucun modèle n'est encore enregistré, renvoie +infini (pour que
    le premier modèle soit toujours enregistré)."""
    try:
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
        if not versions:
            return float("inf")
        best_rmse = float("inf")
        for v in versions:
            run = client.get_run(v.run_id)
            rmse = run.data.metrics.get("test_rmse")
            if rmse is not None and rmse < best_rmse:
                best_rmse = rmse
        return best_rmse
    except mlflow.exceptions.MlflowException:
        return float("inf")



def evaluate_run(run_id: str, X_test, y_test) -> tuple:
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    preds = model.predict(X_test)

    test_r2 = r2_score(y_test, preds)
    test_rmse = root_mean_squared_error(y_test, preds)

    # Log des métriques finales sur le même run
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("test_rmse", test_rmse)

    print(f"[evaluate] run {run_id} -> Test R2 : {test_r2:.4f} | Test RMSE : {test_rmse:.4f}")
    return test_r2, test_rmse

def evaluate(args):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    run_ids = get_run_ids(args)
    print(f"[evaluate] Évaluation de {len(run_ids)} run(s) : {run_ids}")

    test_df = pd.read_csv(args.test_data)
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    # Évalue chaque run du batch et garde le meilleur (RMSE le plus bas)
    results = {run_id: evaluate_run(run_id, X_test, y_test) for run_id in run_ids}
    best_run_id = min(results, key=lambda r: results[r][1])
    best_test_r2, best_test_rmse = results[best_run_id]
    print(f"[evaluate] Meilleur run du batch : {best_run_id} (RMSE={best_test_rmse:.4f})")

    # Comparaison avec le meilleur modèle déjà enregistré
    best_rmse_so_far = get_best_registered_rmse(client)
    print(f"[evaluate] Meilleur RMSE enregistré jusqu'ici : {best_rmse_so_far}")

    if best_test_rmse < best_rmse_so_far:
        print("[evaluate] Nouveau meilleur modèle -> enregistrement dans le Model Registry")
        result = mlflow.register_model(
            model_uri=f"runs:/{best_run_id}/model",
            name=REGISTERED_MODEL_NAME,
        )
        print(f"[evaluate] Modèle enregistré : version {result.version}")
    else:
        print("[evaluate] Aucun modèle du batch n'améliore le meilleur RMSE -> rien enregistré")

    # Sauvegarde locale des métriques (utile pour DVC metrics tracking)
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/eval.json", "w") as f:
        f.write(
            f'{{"best_run_id": "{best_run_id}", "test_r2": {best_test_r2}, '
            f'"test_rmse": {best_test_rmse}}}'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", default="data/processed/test.csv")
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run ID à évaluer (sinon lu depuis metrics/last_run_ids.txt, écrit par train.py)",
    )
    args = parser.parse_args()

    evaluate(args)