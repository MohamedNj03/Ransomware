"""Métriques d'évaluation (section 9.3 du cahier des charges) + fiabilité."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             matthews_corrcoef, roc_auc_score)


def ece(y, p, n_bins=10) -> float:
    """Expected Calibration Error : écart moyen entre score et fréquence réelle."""
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def binary_metrics(y, p, threshold=0.5) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) and not np.isnan(rec) else 0.0
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    both = len(np.unique(y)) == 2
    return {
        "precision": prec, "recall": rec, "f1": f1,
        "accuracy": (tp + tn) / len(y), "fpr": fpr,
        "fnr": fn / (fn + tp) if fn + tp else float("nan"),
        "balanced_acc": np.nanmean([rec, 1 - fpr]) if both else float("nan"),
        "mcc": matthews_corrcoef(y, yhat) if both else float("nan"),
        "auc": roc_auc_score(y, p) if both else float("nan"),
        "pr_auc": average_precision_score(y, p) if both else float("nan"),
        "brier": brier_score_loss(y, p, pos_label=1) if both else float(np.mean((p - y) ** 2)),
        "ece": ece(y, p),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def time_to_detection(y, score_curves, grid_s, threshold=0.5) -> dict:
    """Évaluation en ligne : le score est recalculé toutes les `grid` secondes.

    - ransomware : délai (s) avant la première alerte (inf si jamais détecté) ;
    - bénin : fausse alerte si le score dépasse le seuil à un instant quelconque.
    score_curves : [N, G] scores à chaque instant de la grille (NaN après la fin de la séquence).
    """
    y = np.asarray(y).astype(int)
    S = np.asarray(score_curves, dtype=float)
    alarm = np.nan_to_num(S, nan=-1.0) >= threshold
    first = np.where(alarm.any(1), np.asarray(grid_s)[alarm.argmax(1)], np.inf)
    pos, neg = first[y == 1], first[y == 0]
    out = {}
    if pos.size:
        det = np.isfinite(pos)
        out.update({
            "online_detection_rate": det.mean(),
            "ttd_median_s": float(np.median(pos[det])) if det.any() else float("nan"),
            "ttd_mean_s": float(np.mean(pos[det])) if det.any() else float("nan"),
            "ttd_p90_s": float(np.percentile(pos[det], 90)) if det.any() else float("nan"),
        })
    if neg.size:
        out["online_false_alarm_rate"] = float(np.isfinite(neg).mean())
    return out


def earliest_reliable_fraction(per_fraction: dict, metric="f1", target=0.95) -> float:
    """RQ1 : plus petite fraction de séquence où `metric` >= target (NaN sinon)."""
    for frac in sorted(per_fraction):
        if per_fraction[frac].get(metric, 0) >= target:
            return frac
    return float("nan")
