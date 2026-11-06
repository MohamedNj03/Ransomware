"""Baselines classiques (scikit-learn / gradient boosting), étape 1 de la démarche
progressive du cahier des charges : baseline -> Deep Learning -> modèle amélioré.

Deux familles :
  * WindowModel  — reproduction de Hirano & Kobayashi (CSR 2022 / RanSMAP) :
    chaque fenêtre glissante de 10 s (23 features) est classée indépendamment,
    le score d'un préfixe est la moyenne des scores des fenêtres observées.
  * PrefixSummaryModel — le préfixe observé est résumé par des statistiques
    (moyenne, écart-type, min, max, moyenne des 10 dernières secondes) puis classé.
"""
from __future__ import annotations

import pickle
import time

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .train_dl import prefix_len

TRAIN_FRACS = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]


def make_estimator(kind: str, seed: int):
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced"))
    if kind == "rf":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, class_weight="balanced", random_state=seed)
    if kind == "rf_hirano":  # configuration exacte du papier CSR 2022
        return RandomForestClassifier(n_estimators=10, max_depth=10, n_jobs=-1, random_state=seed)
    if kind == "svm":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed))
    if kind == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    if kind == "lightgbm":
        return LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, random_state=seed, verbose=-1)
    if kind == "xgboost":
        return XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                             tree_method="hist", random_state=seed, n_jobs=-1, eval_metric="logloss")
    raise ValueError(kind)


def summarize(seq: np.ndarray, L: int) -> np.ndarray:
    x = seq[:max(1, L)]
    return np.concatenate([x.mean(0), x.std(0), x.min(0), x.max(0), x[-10:].mean(0)])


class PrefixSummaryModel:
    def __init__(self, kind: str, seed: int = 0):
        self.kind, self.seed = kind, seed
        self.est = make_estimator(kind, seed)

    def fit(self, seqs, y):
        rng = np.random.default_rng(self.seed)
        X, Y = [], []
        for s, lab in zip(seqs, y):
            fracs = TRAIN_FRACS + list(rng.uniform(0.05, 1.0, 2))
            for f in fracs:
                X.append(summarize(s, prefix_len(len(s), f)))
                Y.append(lab)
        t = time.time()
        self.est.fit(np.asarray(X), np.asarray(Y))
        self.train_time_s = time.time() - t
        return self

    def predict(self, seqs, fracs=None, lengths=None):
        if lengths is None:
            lengths = [prefix_len(len(s), fracs) for s in seqs]
        X = np.asarray([summarize(s, l) for s, l in zip(seqs, lengths)])
        return self.est.predict_proba(X)[:, 1]


class WindowModel:
    """Classifieur de fenêtres (séquences calculées avec window=10 par l'appelant)."""

    def __init__(self, kind: str, seed: int = 0, step: int = 2, max_windows: int | None = None):
        self.kind, self.seed, self.step, self.max_windows = kind, seed, step, max_windows
        self.est = make_estimator(kind, seed)

    def fit(self, seqs, y):
        X = np.concatenate([s[::self.step] for s in seqs])
        Y = np.concatenate([np.full(len(s[::self.step]), lab) for s, lab in zip(seqs, y)])
        if self.max_windows and len(X) > self.max_windows:
            idx = np.random.default_rng(self.seed).choice(len(X), self.max_windows, replace=False)
            X, Y = X[idx], Y[idx]
        t = time.time()
        self.est.fit(X, Y)
        self.train_time_s = time.time() - t
        return self

    def predict(self, seqs, fracs=None, lengths=None, cache=True):
        """Score du préfixe = moyenne des probabilités des fenêtres observées.

        Les fenêtres étant causales, les probabilités de toute la séquence sont
        calculées une seule fois puis réutilisées pour chaque préfixe (cache)."""
        if lengths is None:
            lengths = [prefix_len(len(s), fracs) for s in seqs]
        lengths = [max(1, min(l, len(s))) for l, s in zip(lengths, seqs)]
        if not hasattr(self, "_cache"):
            self._cache = {}
        todo = [s for s in seqs if not cache or id(s) not in self._cache]
        if todo:
            allp = self.est.predict_proba(np.concatenate(todo))[:, 1]
            i = 0
            for s in todo:
                self._cache[id(s)] = (s, np.cumsum(allp[i:i + len(s)]))  # garde une référence : id stable
                i += len(s)
        out = [self._cache[id(s)][1][l - 1] / l for s, l in zip(seqs, lengths)]
        if not cache:
            for s in todo:
                self._cache.pop(id(s), None)
        return np.asarray(out)


def measure_latency_classic(model, seqs, n=30) -> dict:
    ts = []
    for i in range(n):
        s = seqs[i % len(seqs)]
        kw = {"cache": False} if isinstance(model, WindowModel) else {}
        t = time.perf_counter()
        model.predict([s.copy()], lengths=[len(s)], **kw)
        ts.append((time.perf_counter() - t) * 1000)
    return {"latency_ms_cpu": float(np.median(ts)), "latency_p95_ms_cpu": float(np.percentile(ts, 95)),
            "model_size_kb": len(pickle.dumps(model.est)) / 1024}
