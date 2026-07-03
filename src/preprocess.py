"""
Script de prétraitement des données.

Lit le dataset brut (Student_performance_data.csv), sélectionne les
features utiles, nettoie les valeurs manquantes, puis split en
train/test. Les fichiers résultants sont sauvegardés dans data/processed/.

"""

import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

# Features utilisées pour prédire le GPA (basé sur le TP1)
FEATURES = ["StudyTimeWeekly", "Age", "Absences"]
TARGET = "GPA"


def preprocess(input_path: str, output_dir: str, test_size: float = 0.2, random_state: int = 42):
    print(f"[preprocess] Lecture des données brutes depuis {input_path}")
    df = pd.read_csv(input_path)

    # Ne garder que les colonnes utiles
    df = df[FEATURES + [TARGET]]

    # Nettoyage basique : suppression des valeurs manquantes
    n_before = len(df)
    df = df.dropna()
    n_after = len(df)
    print(f"[preprocess] Lignes supprimées (NaN) : {n_before - n_after}")

    # Split train / test
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[preprocess] Train set : {train_df.shape} -> {train_path}")
    print(f"[preprocess] Test set  : {test_df.shape} -> {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/Student_performance_data.csv")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    preprocess(args.input, args.output_dir, args.test_size, args.random_state)