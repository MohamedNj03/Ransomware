"""Exécution d'une expérience (modèle x protocole x graine) avec traçabilité MLflow.

Chaque run MLflow contient :
  * tags      : protocole, modèle, famille de modèle, causalité, empreinte des
                données (digest de l'index) et du code (sha1 des sources) ;
  * params    : hyperparamètres, jeu de features, tailles des découpages ;
  * metrics   : métriques par fraction (ex. test_f1_p10 ... test_f1_p100),
                courbes pas-à-pas (step = % de séquence observée), détection
                en ligne (délai médian, fausses alertes), efficacité ;
  * artifacts : prédictions par exécution et par fraction (CSV), figures,
                rappel par famille, historique d'entraînement, modèle.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import baselines as B  # noqa: E402
from .anomaly import AnomalyDetector  # noqa: E402
from .config import FRACTIONS, MLFLOW_ARTIFACTS, MLFLOW_TRACKING_URI, PROJECT_ROOT  # noqa: E402
from .data import (FEATURE_SETS, MEMORY_FEATURES, STORAGE_FEATURES, Scaler, Split, index_digest,  # noqa: E402
                   load_sequences, synthetic_mixes)
from .metrics import binary_metrics, earliest_reliable_fraction, time_to_detection  # noqa: E402
from .models import CAUSAL_MODELS  # noqa: E402
from .train_dl import measure_latency, predict, train_model  # noqa: E402

# ---------------------------------------------------------------------------
# Catalogue des modèles comparés
# ---------------------------------------------------------------------------
MODEL_SPECS = {
    # -- Baselines « projets existants » : Hirano & Kobayashi (fenêtres 10 s, 23 features)
    "hirano_rf":   {"kind": "window", "est": "rf_hirano", "features": "hirano23", "window": 10},
    "hirano_svm":  {"kind": "window", "est": "svm", "features": "hirano23", "window": 10, "max_windows": 12000},
    "hirano_knn":  {"kind": "window", "est": "knn", "features": "hirano23", "window": 10, "max_windows": 30000},
    # -- Baselines classiques sur résumé du préfixe (features étendues)
    "sum_logreg":   {"kind": "summary", "est": "logreg", "features": "all"},
    "sum_rf":       {"kind": "summary", "est": "rf", "features": "all"},
    "sum_lightgbm": {"kind": "summary", "est": "lightgbm", "features": "all"},
    "sum_xgboost":  {"kind": "summary", "est": "xgboost", "features": "all"},
    # -- Deep Learning séquentiel
    "mlp_window":  {"kind": "dl", "model": "mlp_window", "features": "all"},
    "cnn1d":       {"kind": "dl", "model": "cnn1d", "features": "all"},
    "lstm":        {"kind": "dl", "model": "lstm", "features": "all"},
    "gru":         {"kind": "dl", "model": "gru", "features": "all"},
    "bilstm":      {"kind": "dl", "model": "bilstm", "features": "all"},
    "cnn_lstm":    {"kind": "dl", "model": "cnn_lstm", "features": "all"},
    "mhsa_lstm":   {"kind": "dl", "model": "mhsa_lstm", "features": "all"},
    "transformer": {"kind": "dl", "model": "transformer", "features": "all"},
    "tcn":         {"kind": "dl", "model": "tcn", "features": "all"},
    "multimodal":  {"kind": "dl", "model": "multimodal", "features": "all"},
    # -- Modèles améliorés (étape 3 de la démarche progressive)
    "stats_gru":   {"kind": "dl", "model": "stats_gru", "features": "all"},
    "ensemble_mm_lgbm": {"kind": "ensemble", "members": ["multimodal", "sum_lightgbm"], "features": "all"},
    # -- Augmentation par exécutions mixtes synthétiques (robustesse au mode « mix »)
    "multimodal_mixaug": {"kind": "dl", "model": "multimodal", "features": "all", "mix_aug": 1},
    "lgbm_mixaug":       {"kind": "summary", "est": "lightgbm", "features": "all", "mix_aug": 1},
    "ensemble_mixaug":   {"kind": "ensemble", "members": ["multimodal", "sum_lightgbm"], "features": "all", "mix_aug": 1},
    # -- Tentatives de régularisation pour le zero-day (E2), conservées pour traçabilité
    "gru_featdrop": {"kind": "dl", "model": "gru", "features": "all", "train_kwargs": {"feat_dropout": 0.3}},
    "gru_dro":      {"kind": "dl", "model": "gru", "features": "all", "train_kwargs": {"group_dro": True}},
    # -- Non supervisé (entraîné sur le bénin uniquement) -> anomaly_probability
    "gru_anomaly": {"kind": "anomaly", "features": "all"},
    # -- Ablations (apport de chaque modalité et des features ajoutées)
    "gru_storage":  {"kind": "dl", "model": "gru", "features": "storage"},
    "gru_memory":   {"kind": "dl", "model": "gru", "features": "memory"},
    "gru_hirano23": {"kind": "dl", "model": "gru", "features": "hirano23"},
}

ONLINE_GRID_S = 5  # pas (s) de la ré-évaluation en ligne


def measure_latency_anomaly(model, seqs, n=30) -> dict:
    import copy
    import torch
    model = copy.copy(model)  # copie CPU : même contexte de mesure que les autres modèles
    model.model = copy.deepcopy(model.model).cpu()
    model.device = torch.device("cpu")
    ts = []
    for i in range(n):
        s = seqs[i % len(seqs)]
        t = time.perf_counter()
        model.predict([s], lengths=[len(s)])
        ts.append((time.perf_counter() - t) * 1000)
    size_kb = sum(p.numel() * p.element_size() for p in model.model.parameters()) / 1024
    return {"latency_ms_cpu": float(np.median(ts)), "latency_p95_ms_cpu": float(np.percentile(ts, 95)),
            "model_size_kb": size_kb}


def code_digest() -> str:
    h = hashlib.sha1()
    for p in sorted((PROJECT_ROOT / "src").rglob("*.py")) + sorted((PROJECT_ROOT / "scripts").glob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def setup_mlflow(experiment: str) -> str:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        try:
            return mlflow.create_experiment(experiment, artifact_location=(MLFLOW_ARTIFACTS / experiment).as_uri())
        except mlflow.exceptions.MlflowException:  # créée entre-temps par un processus parallèle
            exp = mlflow.get_experiment_by_name(experiment)
    return exp.experiment_id


# ---------------------------------------------------------------------------
class _Prepared:
    """Séquences normalisées pour un découpage et un jeu de features donnés."""

    def __init__(self, split: Split, features: str, window: int, mix_aug: int = 0, seed: int = 0):
        cols = [FEATURE_SETS["all"].index(f) for f in FEATURE_SETS[features]]
        self.feature_names = FEATURE_SETS[features]

        def get(df):
            return [s[:, cols] for s in load_sequences(df["run_uid"].tolist(), window)]

        raw = {"train": get(split.train), "val": get(split.val), "test": get(split.test)}
        raw.update({k: get(v) for k, v in split.extra_tests.items()})
        self.scaler = Scaler.fit(raw["train"])  # ajusté sur les exécutions réelles uniquement
        self.y_train = split.train["label"].to_numpy()
        self.g_train = split.train["app"].to_numpy()
        self.n_synthetic = 0
        if mix_aug:
            syn, ys, gs = synthetic_mixes(split.train, seed, window, mix_aug)
            raw["train"] = raw["train"] + [s[:, cols] for s in syn]
            self.y_train = np.concatenate([self.y_train, ys])
            self.g_train = np.concatenate([self.g_train, gs])
            self.n_synthetic = len(syn)
        self.seqs = {k: self.scaler.transform(v) for k, v in raw.items()}
        self.frames = {"train": split.train, "val": split.val, "test": split.test, **split.extra_tests}


def _fit(spec_name, spec, prep: _Prepared, seed, dl_cfg, log_fn):
    ytr = prep.y_train  # inclut les éventuelles exécutions mixtes synthétiques
    if spec["kind"] == "ensemble":
        # moyenne des probabilités de membres hétérogènes (DL séquentiel + boosting)
        members = [_fit(m, MODEL_SPECS[m], prep, seed, dl_cfg, log_fn if MODEL_SPECS[m]["kind"] == "dl" else None)
                   for m in spec["members"]]
        scorers = [s for _, s, _ in members]
        scorer = lambda seqs, fracs=None, lengths=None: np.mean(  # noqa: E731
            [s(seqs, fracs=fracs, lengths=lengths) for s in scorers], axis=0)
        infos = [i for _, _, i in members]
        info = {k: float(np.nansum([i.get(k, np.nan) for i in infos]))
                for k in ("train_time_s", "n_params", "latency_ms_cpu", "latency_p95_ms_cpu", "model_size_kb")}
        info["device"] = "mixed"
        return [m for m, _, _ in members], scorer, info
    if spec["kind"] == "dl":
        kwargs = dict(spec.get("kwargs", {}))
        if spec["model"] == "multimodal":
            kwargs["storage_idx"] = [prep.feature_names.index(f) for f in STORAGE_FEATURES]
            kwargs["memory_idx"] = [prep.feature_names.index(f) for f in MEMORY_FEATURES]
        model, info = train_model(spec["model"], prep.seqs["train"], ytr, prep.seqs["val"],
                                  prep.frames["val"]["label"].to_numpy(), n_features=len(prep.feature_names),
                                  model_kwargs=kwargs, seed=seed, log_fn=log_fn,
                                  groups=prep.g_train, **spec.get("train_kwargs", {}), **dl_cfg)
        device = info["device"]
        scorer = lambda seqs, fracs=None, lengths=None: predict(model, seqs, fracs, lengths, device=device)  # noqa: E731
        info.update(measure_latency(model, prep.seqs["test"]))
        return model, scorer, info
    if spec["kind"] == "anomaly":
        model = AnomalyDetector(seed=seed).fit(prep.seqs["train"], ytr, prep.seqs["val"], prep.frames["val"]["label"].to_numpy())
        info = {"train_time_s": model.train_time_s, "epochs_run": model.epochs_run, "device": str(model.device),
                "n_params": float(sum(p.numel() for p in model.model.parameters()))}
        info.update(measure_latency_anomaly(model, prep.seqs["test"]))
        return model, model.predict, info
    cls = B.WindowModel if spec["kind"] == "window" else B.PrefixSummaryModel
    extra = {"max_windows": spec["max_windows"]} if "max_windows" in spec else {}
    model = cls(spec["est"], seed=seed, **extra).fit(prep.seqs["train"], ytr)
    info = {"train_time_s": model.train_time_s, "n_params": float("nan"), "device": "cpu"}
    info.update(B.measure_latency_classic(model, prep.seqs["test"]))
    return model, model.predict, info


def _evaluate(scorer, prep: _Prepared, set_name: str):
    """Scores pour chaque fraction + courbes en ligne ; renvoie (métriques, prédictions)."""
    seqs, df = prep.seqs[set_name], prep.frames[set_name]
    y = df["label"].to_numpy()
    preds, per_frac = [], {}
    for f in FRACTIONS:
        p = scorer(seqs, fracs=f)
        per_frac[f] = binary_metrics(y, p)
        preds.append(pd.DataFrame({"run_uid": df["run_uid"].values, "app": df["app"].values, "family": df["family"].values,
                                   "hw": df["hw"].values, "label": y, "set": set_name, "fraction": f, "score": p}))
    curves, grid = _online_curves(scorer, seqs)
    return per_frac, (y, curves, grid), pd.concat(preds, ignore_index=True)


def _online_curves(scorer, seqs):
    """Score recalculé toutes les ONLINE_GRID_S secondes (simulation du streaming)."""
    T = np.array([len(s) for s in seqs])
    grid = np.arange(ONLINE_GRID_S, T.max() + ONLINE_GRID_S, ONLINE_GRID_S)
    curves = np.full((len(seqs), len(grid)), np.nan)
    for j, g in enumerate(grid):
        alive = np.where(T >= g - ONLINE_GRID_S + 1)[0]
        if alive.size:
            curves[alive, j] = scorer([seqs[i] for i in alive], lengths=[min(g, T[i]) for i in alive])
    return curves, grid


def _operating_threshold(y_val, curves, grid, max_far=0.05) -> float:
    """Seuil opérationnel : le plus bas seuil dont le taux de fausses alertes EN LIGNE
    sur la validation reste <= max_far (une alerte à n'importe quel instant compte)."""
    for thr in np.round(np.arange(0.50, 1.00, 0.01), 2):
        if time_to_detection(y_val, curves, grid, thr).get("online_false_alarm_rate", 0.0) <= max_far:
            return float(thr)
    return 0.99


def _plot_early(per_frac_by_set: dict, path: Path, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for set_name, per_frac in per_frac_by_set.items():
        fr = [100 * f for f in per_frac]
        if set_name == "test":
            for m, c in [("f1", "C0"), ("recall", "C1"), ("precision", "C2"), ("fpr", "C3")]:
                axes[0].plot(fr, [per_frac[f][m] for f in per_frac], marker="o", label=m, color=c)
        else:
            axes[1].plot(fr, [per_frac[f]["recall"] for f in per_frac], marker="o", label=f"{set_name} recall")
    axes[0].set(title="Test : métriques vs fraction observée", xlabel="% de séquence", ylim=(-0.02, 1.02))
    axes[1].set(title="Tests externes : rappel", xlabel="% de séquence", ylim=(-0.02, 1.02))
    for a in axes:
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_confusions(per_frac: dict, path: Path):
    fig, axes = plt.subplots(1, len(per_frac), figsize=(2.4 * len(per_frac), 2.4))
    for a, (f, m) in zip(np.atleast_1d(axes), per_frac.items()):
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        a.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            a.text(j, i, str(v), ha="center", va="center", fontsize=9)
        a.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["bén.", "rans."], yticklabels=["bén.", "rans."],
              title=f"{int(f * 100)} %")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_experiment(spec_name: str, split: Split, seed: int, protocol: str, dl_cfg: dict | None = None,
                   log_model: bool = False, parent_run_id: str | None = None) -> dict:
    spec = MODEL_SPECS[spec_name]
    dl_cfg = dl_cfg or {}
    window = spec.get("window", 1)
    t_start = time.time()
    prep = _Prepared(split, spec["features"], window, mix_aug=spec.get("mix_aug", 0), seed=seed)
    all_frames = pd.concat(list(prep.frames.values()))
    tags = {
        "protocol": protocol, "model": spec_name, "model_kind": spec["kind"],
        "causal_streaming": str(spec.get("model") in CAUSAL_MODELS),
        "feature_set": spec["features"], "split": split.name,
        "data_digest": index_digest(all_frames), "code_digest": code_digest(),
    }
    with mlflow.start_run(run_name=f"{spec_name} | {split.name}", nested=parent_run_id is not None,
                          parent_run_id=parent_run_id, tags=tags) as run:
        params = {"seed": seed, "window_s": window, "n_features": len(prep.feature_names), **split.describe(),
                  "n_train_synthetic_mixes": prep.n_synthetic,
                  **{k: v for k, v in spec.items() if k not in ("kind",)}}
        if spec["kind"] == "dl":
            params.update({"epochs": 60, "batch_size": 32, "lr": 1e-3, "weight_decay": 1e-4, "patience": 12,
                           "prefix_aug": "U(0.02,1) p=0.8", **dl_cfg})
        mlflow.log_params(params)
        mlflow.log_input(mlflow.data.from_pandas(split.train[["run_uid", "app", "family", "hw", "label"]],
                                                 name="ransmap_train", targets="label"), context="training")
        mlflow.log_input(mlflow.data.from_pandas(split.test[["run_uid", "app", "family", "hw", "label"]],
                                                 name="ransmap_test", targets="label"), context="test")

        log_fn = (lambda m, step: mlflow.log_metrics(m, step=step)) if spec["kind"] == "dl" else None
        model, scorer, info = _fit(spec_name, spec, prep, seed, dl_cfg, log_fn)

        results, preds_all, per_frac_by_set = {}, [], {}
        val_curves, val_grid = _online_curves(scorer, prep.seqs["val"])
        op_thr = _operating_threshold(prep.frames["val"]["label"].to_numpy(), val_curves, val_grid)
        results["operating_threshold"] = op_thr
        for set_name in prep.seqs:
            if set_name in ("train", "val"):
                continue
            per_frac, (y_set, curves, grid), preds = _evaluate(scorer, prep, set_name)
            per_frac_by_set[set_name] = per_frac
            preds_all.append(preds)
            prefix = "test" if set_name == "test" else set_name
            keys = ["f1", "precision", "recall", "accuracy", "fpr", "auc", "pr_auc", "mcc", "brier", "ece", "balanced_acc"]
            for f, m in per_frac.items():
                pct = int(round(f * 100))
                for k in keys:
                    if not np.isnan(m[k]):
                        results[f"{prefix}_{k}_p{pct}"] = float(m[k])
                        mlflow.log_metric(f"{prefix}_{k}_curve", float(m[k]), step=pct)
                # mêmes métriques au seuil opérationnel calibré sur la validation
                p_f = preds[preds.fraction == f]["score"].to_numpy()
                mo = binary_metrics(y_set, p_f, threshold=op_thr)
                for k in ("f1", "precision", "recall", "fpr"):
                    if not np.isnan(mo[k]):
                        results[f"{prefix}_op_{k}_p{pct}"] = float(mo[k])
            for thr, tag in ((0.5, ""), (op_thr, "op_")):
                for k, v in time_to_detection(y_set, curves, grid, thr).items():
                    if not np.isnan(v):
                        results[f"{prefix}_{tag}{k}"] = float(v)
            if set_name == "test":
                erf = earliest_reliable_fraction(per_frac, "f1", 0.95)
                results["test_earliest_frac_f1_95"] = erf if not np.isnan(erf) else 1.5

        for k in ("train_time_s", "n_params", "latency_ms_cpu", "latency_p95_ms_cpu", "model_size_kb", "epochs_run"):
            if k in info and not (isinstance(info[k], float) and np.isnan(info[k])):
                results[k] = float(info[k])
        results["total_time_s"] = time.time() - t_start
        mlflow.log_metrics(results)

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            preds = pd.concat(preds_all, ignore_index=True)
            preds.to_csv(td / "predictions.csv", index=False)
            fam = (preds[preds["set"] == "test"].assign(pred=lambda d: d.score >= 0.5)
                   .groupby(["app", "fraction"])["pred"].mean().unstack())
            fam.to_csv(td / "detection_rate_by_app.csv")
            _plot_early(per_frac_by_set, td / "early_detection.png", f"{spec_name} — {split.name}")
            _plot_confusions(per_frac_by_set["test"], td / "confusion_matrices.png")
            json.dump({"features": prep.feature_names, "scaler_mean": prep.scaler.mean.tolist(),
                       "scaler_std": prep.scaler.std.tolist(), "window_s": window},
                      open(td / "preprocessing.json", "w"), indent=1)
            if "history" in info:
                pd.DataFrame(info["history"]).to_csv(td / "training_history.csv", index=False)
            mlflow.log_artifacts(str(td))
        if log_model and spec["kind"] == "ensemble":
            for mname, member in zip(spec["members"], model):
                if MODEL_SPECS[mname]["kind"] == "dl":
                    mlflow.pytorch.log_model(member.cpu(), name=f"model_{mname}", serialization_format="pickle")
                else:
                    import skops.io
                    trusted = skops.io.get_untrusted_types(data=skops.io.dumps(member.est))
                    mlflow.sklearn.log_model(member.est, name=f"model_{mname}", skops_trusted_types=trusted)
        elif log_model:
            if spec["kind"] == "dl":
                mlflow.pytorch.log_model(model.cpu(), name="model", serialization_format="pickle")
            elif spec["kind"] == "anomaly":
                mlflow.pytorch.log_model(model.model.cpu(), name="model", serialization_format="pickle")
            else:
                import skops.io  # MLflow 3 sérialise via skops : on déclare nos propres types comme sûrs
                trusted = skops.io.get_untrusted_types(data=skops.io.dumps(model.est))
                mlflow.sklearn.log_model(model.est, name="model", skops_trusted_types=trusted)
        results.update({"run_id": run.info.run_id, "model": spec_name, "split": split.name, "seed": seed,
                        "protocol": protocol, "kind": spec["kind"], "features": spec["features"]})
    return results
