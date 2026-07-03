# 🎓 Student GPA Prediction - MLOps Pipeline

## 📊 Présentation du problème

**Problème ML** : Régression — prédire le **GPA** (moyenne) d'un étudiant.
**Dataset** : `Student_performance_data.csv` — features utilisées : `StudyTimeWeekly`, `Age`, `Absences`.
**Objectif** : Aider à identifier les étudiants à risque en fonction de leur temps d'étude et de leur assiduité.

## 🏗️ Architecture technique

- **EC2 #1** : Serveur MLflow Tracking (port 5000) — tracking + Model Registry
- **EC2 #2** : Application Flask de prédiction (port 8000)
- **S3** : Stockage centralisé (données brutes, prétraitées, modèles, remote DVC)
- **DVC** : Versionning des données + pipeline reproductible (`dvc.yaml`)
- **GitHub** : Code, historique de collaboration

## ⚙️ Installation et reproduction

```bash
# 1. Cloner le repo
git clone <url-du-repo>
cd mlops-student-performance

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Récupérer les données versionnées (une fois DVC/S3 configurés)
dvc pull

# 4. Configurer l'URL du serveur MLflow (EC2)
export MLFLOW_TRACKING_URI="http://<IP_EC2_MLFLOW>:5000"

# 5. Relancer tout le pipeline (preprocess -> train -> evaluate)
dvc repro

# 6. Lancer l'application Flask
python app/app.py
```

## 🔗 Liens d'accès
- **MLflow UI** : http://<IP_EC2_MLFLOW>:5000
- **Application Flask** : http://<IP_EC2_FLASK>:8000
- **Repository GitHub** : (ce repo)

## 📁 Structure du projet
```
├── data/raw/           # Données brutes (versionnées DVC)
├── data/processed/     # train.csv / test.csv (générés)
├── src/
│   ├── preprocess.py   # Nettoyage + split
│   ├── train.py        # Entraînement + log MLflow
│   └── evaluate.py     # Évaluation + registre du meilleur modèle
├── app/
│   ├── app.py           # Interface Flask
│   └── templates/index.html
├── dvc.yaml             # Pipeline DVC
└── requirements.txt
```

## 👥 Équipe
| Membre | Contribution |
|---|---|
| ... | ... |
