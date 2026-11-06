"""Rapport qualité / nettoyage des données + traçabilité de la version du dataset.

Produit reports/DATA_QUALITY.md et enregistre dans MLflow (expérience
« RanSMAP_Data ») : index complet des exécutions avec SHA git des CSV sources,
exécutions exclues et raison, profil moyen des features par application.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from ransomdl.config import PROCESSED_DIR, REPORTS_DIR  # noqa: E402
from ransomdl.data import ALL_FEATURES, index_digest, load_index, load_sequences  # noqa: E402
from ransomdl.experiment import setup_mlflow  # noqa: E402


def main():
    full = load_index(keep_excluded=True)
    kept = full[full.exclude_reason == ""]
    excl = full[full.exclude_reason != ""]
    lines = ["# Qualité des données et nettoyage (RanSMAP)", "",
             f"Version des données (digest) : `{index_digest(kept)}` — commit RanSMAP `{full.ransmap_commit.iloc[0][:10]}`", "",
             "## Volumétrie", "",
             f"- Exécutions extraites : **{len(full)}** ; conservées : **{len(kept)}** ; exclues : **{len(excl)}**",
             f"- Événements bruts traités : **{int(full[[c for c in full.columns if c.startswith('n_') and c != 'n_bins']].sum().sum()):,}**",
             f"- Durée médiane d'une exécution : **{kept.duration_s.median():.0f} s** (séquence de {kept.n_bins.median():.0f} pas d'1 s)", "",
             pd.crosstab([kept.subset, kept.cpu + "/" + kept.ram], kept.label.map({0: "bénin", 1: "ransomware"})).to_markdown(), "",
             "## Exécutions par application (conservées)", "",
             kept.groupby(["subset", "app"]).size().unstack(0).fillna(0).astype(int).to_markdown(), "",
             "## Nettoyage appliqué", "",
             "1. **Capture incomplète** : durée < 50 % de la médiane du sous-ensemble -> exclue.",
             "2. **Modalité absente** : aucune trace disque ou aucune trace mémoire -> exclue.",
             "3. **Doublons** : même empreinte de features -> exclue.",
             "4. **Valeurs** : entropies bornées à [0, 1] (valeurs sentinelles -1 ignorées), entropie mémoire "
             "calculée uniquement sur les pages 4 Ko non-MMIO, NaN/inf remplacés par 0.",
             "5. **Échelle** : comptes, débits et variances passés en log1p, puis standardisation ajustée "
             "**uniquement sur l'entraînement**, écrêtage à ±10 écarts-types.",
             "6. **Découpage par exécution** (jamais par fenêtre) : une exécution n'apparaît que dans un seul "
             "des ensembles entraînement / validation / test.", ""]
    if len(excl):
        lines += ["### Exécutions exclues", "",
                  excl[["run_uid", "app", "duration_s", "n_bins", "exclude_reason"]].to_markdown(index=False), ""]
    orig = kept[kept.subset == "original"]
    seqs = load_sequences(orig.run_uid.tolist())
    prof = pd.DataFrame([s.mean(0) for s in seqs], columns=ALL_FEATURES)
    prof["app"] = orig.app.values
    key = ["aw_tput", "ar_tput", "aw_ent_mean", "aw_hi_frac", "aw_ovw_frac", "aw_seq_frac", "mw_t2", "mw_ent_mean", "mr_t4", "me_t2"]
    pt = prof.groupby("app")[key].mean()
    pt.insert(0, "classe", ["ransomware" if a in ("Conti", "Darkside", "LockBit", "REvil", "Ryuk", "WannaCry") else "bénin" for a in pt.index])
    pt = pt.sort_values("classe")
    lines += ["## Profil comportemental moyen (sous-ensemble original)", "",
              "`aw_ovw_frac` = part des écritures qui réécrivent un secteur lu plus tôt (chiffrement en place) ; "
              "`aw_hi_frac` = part d'écritures à entropie > 0,9 ; débits en log1p(Mo/s).", "",
              pt.round(3).to_markdown(), ""]
    # contrôle comportemental : exécutions étiquetées ransomware sans activité de chiffrement visible
    rans = kept[kept.label == 1]
    rs = load_sequences(rans.run_uid.tolist())
    hi = pd.Series([s[:, ALL_FEATURES.index("aw_hi_frac")].mean() for s in rs], index=rans.index)
    inactive = rans[hi < 0.05].assign(aw_hi_frac=hi[hi < 0.05])
    lines += ["## Contrôle comportemental (signalé, non exclu)", "",
              "Exécutions étiquetées ransomware dont moins de 5 % des écritures sont à forte entropie "
              "(aucun chiffrement visible). Elles sont **conservées** (pas d'exclusion a posteriori) mais "
              "expliquent un plafond de rappel : aucun détecteur comportemental ne peut les repérer.", ""]
    if len(inactive):
        lines += [inactive.groupby(["subset", "app"]).agg(n=("run_uid", "size"), aw_hi_frac_moyen=("aw_hi_frac", "mean"))
                  .round(3).to_markdown(), ""]
    out = REPORTS_DIR / "DATA_QUALITY.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    full.to_csv(PROCESSED_DIR / "index_with_quality.csv", index=False)
    print("\n".join(lines[:30]))

    exp = setup_mlflow(os.environ.get("RANSOMDL_EXP_PREFIX", "") + "RanSMAP_Data")
    mlflow.set_experiment(experiment_id=exp)
    with mlflow.start_run(run_name=f"dataset_{index_digest(kept)}"):
        mlflow.log_params({"n_runs_total": len(full), "n_runs_kept": len(kept), "n_runs_excluded": len(excl),
                           "data_digest": index_digest(kept), "ransmap_commit": full.ransmap_commit.iloc[0]})
        mlflow.log_input(mlflow.data.from_pandas(kept[["run_uid", "subset", "hw", "app", "family", "label"]],
                                                 name="ransmap_index", targets="label"), context="dataset")
        mlflow.log_artifact(str(out))
        mlflow.log_artifact(str(PROCESSED_DIR / "index_with_quality.csv"))


if __name__ == "__main__":
    main()
