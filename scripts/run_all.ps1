# Pipeline complet, reproductible de bout en bout (BNF02).
# Usage (PowerShell, depuis la racine du projet) :  .\scripts\run_all.ps1
$ErrorActionPreference = "Stop"
$py = "C:\Users\nejim\venvs\ransomware-dl\Scripts\python.exe"   # venv : torch CUDA, mlflow, sklearn...
$env:MLFLOW_DISABLE_AGENT_HINT = "1"

# 1. Données : téléchargement en flux + extraction des features (reprenable)
& $py scripts\build_dataset.py --workers 8

# 2. Serveur de suivi MLflow (écritures concurrentes sûres) -> http://127.0.0.1:5000
$root = (Get-Location).Path -replace '\\', '/'
Start-Process -WindowStyle Hidden "C:\Users\nejim\venvs\ransomware-dl\Scripts\mlflow.exe" -ArgumentList `
    "server", "--backend-store-uri", "sqlite:///$root/mlflow.db", "--default-artifact-root", "file:///$root/mlruns", `
    "--host", "127.0.0.1", "--port", "5000"
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

# 3. Qualité des données + version du dataset tracée dans MLflow
& $py scripts\data_report.py

# 4. Protocoles expérimentaux
$classic = "hirano_rf hirano_svm hirano_knn sum_logreg sum_rf sum_lightgbm sum_xgboost".Split(" ")
$deep = "mlp_window cnn1d lstm gru bilstm cnn_lstm mhsa_lstm transformer tcn multimodal gru_anomaly gru_storage gru_memory gru_hirano23".Split(" ")
$improved = "stats_gru ensemble_mm_lgbm multimodal_mixaug lgbm_mixaug ensemble_mixaug".Split(" ")
& $py scripts\run_experiments.py --protocol e1 --models @classic --seeds 0 1 2 3 4 --tag cpu
& $py scripts\run_experiments.py --protocol e1 --models @deep --seeds 0 1 2 3 4 --tag gpu
& $py scripts\run_experiments.py --protocol e1 --models @improved --seeds 0 1 2 3 4 --tag improved
foreach ($p in "e2", "e3", "e4") {
    & $py scripts\run_experiments.py --protocol $p --models @classic --seeds 0 --tag cpu
    & $py scripts\run_experiments.py --protocol $p --models @deep --seeds 0 --tag gpu
    & $py scripts\run_experiments.py --protocol $p --models @improved --seeds 0 --tag improved
}
# tentatives de régularisation zero-day (tracées ; résultats négatifs conservés)
& $py scripts\run_experiments.py --protocol e2 --models gru_featdrop gru_dro --seeds 0 --tag reg

# 5. Efficacité à conditions identiques, puis rapport comparatif (+ Model Registry)
& $py scripts\benchmark_latency.py
& $py scripts\make_report.py
