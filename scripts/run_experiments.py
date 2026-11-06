"""Lance les protocoles expérimentaux du cahier des charges (section 9.2).

  e1 : détection précoce (RQ1, RQ2) — découpage par exécution, plusieurs graines ;
  e2 : famille inconnue (RQ3)       — leave-one-family-out sur les 6 familles ;
  e3 : matériel inconnu (robustesse) — entraînement sur un CPU, test sur les autres ;
  e4 : application bénigne inconnue  — faux positifs sur un logiciel légitime jamais vu.

Pour chaque modèle, un run MLflow parent agrège (moyenne, écart-type) les runs
enfants (une graine ou un pli chacun). Reprenable : les couples (modèle, split)
déjà présents dans reports/results/<protocole>.csv sont sautés.

Exemples :
  python scripts/run_experiments.py --protocol e1 --models all --seeds 0 1 2 3 4
  python scripts/run_experiments.py --protocol e2 --models gru tcn sum_lightgbm
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from ransomdl.config import BENIGN_APPS, RANSOMWARE_FAMILIES, REPORTS_DIR  # noqa: E402
from ransomdl.data import benign_app_split, family_split, hardware_split, load_index, run_split  # noqa: E402
from ransomdl.experiment import MODEL_SPECS, run_experiment, setup_mlflow  # noqa: E402

EXPERIMENTS = {
    "e1": "RanSMAP_E1_early_detection",
    "e2": "RanSMAP_E2_unknown_family",
    "e3": "RanSMAP_E3_unknown_hardware",
    "e4": "RanSMAP_E4_unknown_benign_app",
}


def make_splits(protocol, idx, seeds):
    if protocol == "e1":
        return [(run_split(idx, s), s) for s in seeds]
    if protocol == "e2":
        return [(family_split(idx, fam, s), s) for fam in RANSOMWARE_FAMILIES for s in seeds]
    if protocol == "e3":
        return [(hardware_split(idx, s, cpu), s) for cpu in ("i3-gen12", "celeron-gen6") for s in seeds]
    if protocol == "e4":
        return [(benign_app_split(idx, app, s), s) for app in BENIGN_APPS for s in seeds]
    raise ValueError(protocol)


def find_parent(exp_id, model):
    runs = mlflow.search_runs([exp_id], filter_string=f"tags.model = '{model}' and tags.role = 'parent'",
                              output_format="list")
    return runs[0].info.run_id if runs else None


def aggregate_parent(parent_id, rows: pd.DataFrame):
    num = rows.select_dtypes(include=[np.number]).drop(columns=["seed"], errors="ignore")
    agg = {}
    for c in num.columns:
        v = num[c].dropna()
        if len(v):
            agg[f"{c}_mean"] = float(v.mean())
            agg[f"{c}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    agg["n_children"] = len(rows)
    client = mlflow.tracking.MlflowClient()
    for k, v in agg.items():
        client.log_metric(parent_id, k, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=list(EXPERIMENTS), required=True)
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--log-model-seed", type=int, default=0, help="graine pour laquelle le modèle est sauvegardé")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--tag", default="", help="suffixe du CSV de résultats (exécutions parallèles)")
    args = ap.parse_args()

    models = list(MODEL_SPECS) if args.models == ["all"] else args.models
    idx = load_index()
    exp_name = os.environ.get("RANSOMDL_EXP_PREFIX", "") + EXPERIMENTS[args.protocol]
    exp_id = setup_mlflow(exp_name)
    mlflow.set_experiment(experiment_id=exp_id)
    out_csv = REPORTS_DIR / "results" / f"{args.protocol}{'_' + args.tag if args.tag else ''}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    others = [pd.read_csv(p) for p in out_csv.parent.glob(f"{args.protocol}*.csv") if p != out_csv]
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(columns=["model", "split"])
    seen = pd.concat([done[["model", "split"]]] + [o[["model", "split"]] for o in others], ignore_index=True)
    splits = make_splits(args.protocol, idx, args.seeds)
    print(f"{len(idx)} runs | {len(splits)} splits | {len(models)} models -> {exp_name}", flush=True)

    for model in models:
        parent = find_parent(exp_id, model)
        if parent is None:
            with mlflow.start_run(run_name=f"{model} [{args.protocol}]",
                                  tags={"role": "parent", "model": model, "protocol": args.protocol,
                                        "model_kind": MODEL_SPECS[model]["kind"],
                                        "feature_set": MODEL_SPECS[model]["features"]}) as r:
                mlflow.log_params({k: v for k, v in MODEL_SPECS[model].items()})
                parent = r.info.run_id
        for split, seed in splits:
            if ((seen["model"] == model) & (seen["split"] == split.name)).any():
                continue
            print(f"--> {model:14s} {split.name}", flush=True)
            try:
                res = run_experiment(model, split, seed, args.protocol, dl_cfg={"epochs": args.epochs},
                                     log_model=(seed == args.log_model_seed and args.protocol == "e1"),
                                     parent_run_id=parent)
            except Exception:
                traceback.print_exc()
                continue
            done = pd.concat([done, pd.DataFrame([res])], ignore_index=True)
            done.to_csv(out_csv, index=False)
            key = [k for k in ("test_f1_p10", "test_f1_p25", "test_f1_p100", "test_fpr_p100", "test_recall_p25") if k in res]
            print("    " + "  ".join(f"{k}={res[k]:.3f}" for k in key), flush=True)
        rows = pd.concat([done] + others, ignore_index=True)
        rows = rows[rows["model"] == model]
        if len(rows):
            aggregate_parent(parent, rows)


if __name__ == "__main__":
    main()
