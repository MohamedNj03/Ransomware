"""Inférence : traces brutes -> analyse structurée JSON (interface avec Hermes / JEV).

Sortie conforme à l'exemple de la section 6.2 du cahier des charges :
    {"ransomware_probability": 0.78, "anomaly_probability": 0.91, ...}
enrichie des indicateurs comportementaux utiles à l'état du cas (activité
stockage, entropie des écritures, réécritures en place, activité mémoire).
Chaque décision est ajoutée à un journal JSONL (BNF04 traçabilité, BF16).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import mlflow
import numpy as np
import torch

from .config import MLFLOW_TRACKING_URI
from .data import ALL_FEATURES, derive_features
from .features import extract_run


class _Loaded:
    def __init__(self, run_id: str):
        self.run_id = run_id
        local = Path(mlflow.artifacts.download_artifacts(run_id=run_id))
        pre = json.loads((local / "preprocessing.json").read_text())
        self.features = pre["features"]
        self.cols = [ALL_FEATURES.index(f) for f in self.features]
        self.mean, self.std = np.asarray(pre["scaler_mean"]), np.asarray(pre["scaler_std"])
        self.window = pre.get("window_s", 1)
        self.model = mlflow.pytorch.load_model(f"runs:/{run_id}/model", map_location="cpu").eval()
        self.name = mlflow.get_run(run_id).data.tags.get("model", "?")

    def sequence(self, stats: np.ndarray) -> np.ndarray:
        X = derive_features(stats, self.window)[:, self.cols]
        return np.clip((X - self.mean) / self.std, -10, 10).astype(np.float32)


class RansomwareDetector:
    """Détecteur supervisé (+ optionnellement le détecteur d'anomalies non supervisé)."""

    def __init__(self, classifier_run_id: str, threshold: float = 0.5):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self.clf = _Loaded(classifier_run_id)
        self.threshold = threshold

    @classmethod
    def from_registry(cls, name="ransomware-detector", alias="champion"):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mv = mlflow.tracking.MlflowClient().get_model_version_by_alias(name, alias)
        thr = mlflow.get_run(mv.run_id).data.metrics.get("operating_threshold", 0.5)
        return cls(mv.run_id, threshold=thr)

    @torch.no_grad()
    def analyze(self, stats: np.ndarray, observed_seconds: int | None = None, case_id: str = "") -> dict:
        T = stats.shape[0] if observed_seconds is None else max(1, min(observed_seconds, stats.shape[0]))
        stats = stats[:T]
        x = torch.from_numpy(self.clf.sequence(stats))[None]
        p = float(torch.sigmoid(self.clf.model(x, torch.tensor([T]))).item())
        raw = derive_features(stats, 1)
        f = {n: float(raw[:, i].mean()) for i, n in enumerate(ALL_FEATURES)}
        out = {
            "case_id": case_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": self.clf.name, "model_run_id": self.clf.run_id,
            "observed_seconds": int(T),
            "ransomware_probability": round(p, 4),
            "decision": "ransomware" if p >= self.threshold else "benign",
            "threshold": self.threshold,
            "indicators": {  # contexte lisible pour Hermes (état du cas)
                "storage_write_mb_s": round(float(np.expm1(f["aw_tput"])), 3),
                "storage_read_mb_s": round(float(np.expm1(f["ar_tput"])), 3),
                "write_entropy_mean": round(f["aw_ent_mean"], 3),
                "high_entropy_write_ratio": round(f["aw_hi_frac"], 3),
                "in_place_overwrite_ratio": round(f["aw_ovw_frac"], 3),
                "memory_write_entropy_mean": round(f["mw_ent_mean"], 3),
            },
        }
        weights = getattr(self.clf.model, "last_modality_weights", None)
        if weights is not None:
            w = weights[0].tolist()
            out["modality_weights"] = {"storage": round(w[0], 3), "memory": round(w[1], 3)}
        return out


def analyze_run_dir(run_dir: str | Path, detector: RansomwareDetector, observed_seconds=None,
                    log_file: str | Path | None = None) -> dict:
    """Analyse un dossier d'exécution au format RanSMAP (ata_*.csv, mem_*.csv)."""
    stats, _ = extract_run(Path(run_dir))
    res = detector.analyze(stats, observed_seconds, case_id=Path(run_dir).name)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(res, ensure_ascii=False) + "\n")
    return res
