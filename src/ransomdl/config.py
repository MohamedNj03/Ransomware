"""Configuration centrale du projet (chemins, labels, constantes)."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Les CSV bruts (~133 Go décompressés) restent hors OneDrive ; seules les
# features agrégées (quelques centaines de Mo) vivent dans le projet.
RAW_REPO = Path(r"C:\Users\nejim\datasets\RanSMAP")
RAW_DATASET = RAW_REPO / "dataset"

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RUNS_DIR = PROCESSED_DIR / "runs"          # un .npz par exécution
INDEX_FILE = PROCESSED_DIR / "index.csv"   # métadonnées de toutes les exécutions
REPORTS_DIR = PROJECT_ROOT / "reports"
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
MLFLOW_ARTIFACTS = PROJECT_ROOT / "mlruns"
# Par défaut : base SQLite locale. Pour des entraînements parallèles, lancer
# `mlflow server` (voir README) et exporter MLFLOW_TRACKING_URI=http://127.0.0.1:5000.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB.as_posix()}")

# Pas de temps des séquences : 1 bin = 1 seconde d'activité.
BIN_SECONDS = 1.0

RANSOMWARE_FAMILIES = ["Conti", "Darkside", "LockBit", "REvil", "Ryuk", "WannaCry"]
BENIGN_APPS = ["AESCrypt", "Firefox", "Idle", "Office", "SDelete", "Zip"]

# Fractions de séquence imposées par le cahier des charges (RQ1/RQ2).
FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]

SEED = 42


def label_of(app: str) -> int:
    """1 = ransomware, 0 = bénin. Les mélanges contenant un ransomware sont positifs."""
    parts = app.split("_")
    base = parts[0]
    if base == "Conti" and len(parts) == 2 and parts[1].isdigit():
        return 1  # variantes Conti_01..07
    return int(any(p in RANSOMWARE_FAMILIES for p in parts))


def family_of(app: str) -> str:
    """Famille de ransomware (pour le protocole famille inconnue) ou 'benign'."""
    parts = app.split("_")
    for p in parts:
        if p in RANSOMWARE_FAMILIES:
            return p
    return "benign"
