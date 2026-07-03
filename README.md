# 🎓 Student GPA Prediction - MLOps Pipeline

## 📊 Présentation du problème

**Problème ML** : Régression — prédire le **GPA** (moyenne) d'un étudiant.
**Dataset** : `Student_performance_data.csv` — features utilisées : `StudyTimeWeekly`, `Age`, `Absences`.
**Objectif** : Aider à identifier les étudiants à risque en fonction de leur temps d'étude et de leur assiduité.

## 🏗️ Architecture technique

- **EC2 `mlflow-server-mlops`** (t3.small, eu-west-3) : héberge à la fois le serveur MLflow
  Tracking (port 5000, tracking + Model Registry) et l'application Flask de prédiction
  (port 8000). Les deux tournent en `systemd` (`mlflow.service`, `flaskapp.service`),
  redémarrage automatique en cas de crash. Une seule machine partagée pour tout le groupe,
  conformément à la consigne "ne pas multiplier les instances".
- **S3 (`mlops-projects-4`)** : remote DVC (`s3://mlops-projects-4/dvc-store`) pour les
  données prétraitées et les artefacts versionnés.
- **DVC** : pipeline reproductible (`dvc.yaml` : preprocess → train → evaluate), au moins
  deux versions du dataset brut taguées `data-v1` / `data-v2` (bascule via
  `git checkout <tag> && dvc checkout`).
- **GitHub** : code, historique de collaboration (branches personnelles + merges vers `main`).

## ⚙️ Installation et reproduction

```bash
# 1. Cloner le repo
git clone <url-du-repo>
cd mlops-project

# 2. Installer les dépendances (idéalement dans un venv/conda dédié)
pip install -r requirements.txt

# 3. Configurer les identifiants AWS (accès au bucket S3 du remote DVC)
aws configure   # ou export AWS_PROFILE=<profil-avec-acces-a-mlops-projects-4>

# 4. Récupérer les données versionnées
dvc pull

# 5. Configurer l'URL du serveur MLflow partagé
export MLFLOW_TRACKING_URI="http://51.44.106.147:5000"

# 6. Relancer tout le pipeline (preprocess -> train -> evaluate)
dvc repro

# 7. Lancer l'application Flask en local (optionnel, une instance tourne déjà sur l'EC2)
python app/app.py
```

### Basculer entre les deux versions du dataset

```bash
git checkout data-v1   # ou data-v2
dvc checkout
```

## 🔗 Liens d'accès
- **MLflow UI** : http://51.44.106.147:5000
- **Application Flask** : http://51.44.106.147:8000
- **Repository GitHub** : https://github.com/ahmedelmewloud/mlops-project

## 📁 Structure du projet
```
├── data/raw/            # Données brutes (Student_performance_data.csv, versionné DVC + Git)
├── data/processed/      # train.csv / test.csv (générés, cachés par DVC)
├── src/
│   ├── preprocess.py    # Nettoyage + split train/test
│   ├── train.py         # Entraînement de plusieurs modèles + log MLflow
│   ├── evaluate.py      # Évaluation sur le test set + registre du meilleur modèle (alias "champion")
│   └── build_model.py   # Script autonome (hors pipeline officiel) pour un modèle "servable" rapide
├── app/
│   ├── app.py            # Interface Flask, charge models:/student_gpa_model@champion depuis MLflow
│   └── templates/index.html
├── dvc.yaml              # Pipeline DVC (preprocess -> train -> evaluate)
├── params.yaml            # Hyperparamètres des modèles candidats
└── requirements.txt
```

## 👥 Équipe
| Membre | Contribution |
|---|---|
| 22013 | Infrastructure (EC2 MLflow + Flask, S3, DVC), `evaluate.py`, `app.py`, déploiement, merges vers `main` |
| 22024 | `preprocess.py`, `app.py` (v1), `build_model.py` |
| 22051 | `evaluate.py` (squelette initial) |
