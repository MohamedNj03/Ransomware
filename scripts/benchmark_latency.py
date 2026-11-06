"""Benchmark d'efficacité à conditions identiques (à lancer quand rien d'autre ne tourne).

Recharge depuis MLflow les modèles E1 sauvegardés (graine 0) et mesure, sur CPU
mono-thread (contexte d'un agent de détection sur poste) :
  * latence d'inférence d'une séquence complète (médiane, p95) ;
  * débit (séquences / s) ;
  * taille sérialisée du modèle.
Les mesures sont ajoutées au run MLflow du modèle et à reports/tables/latency_benchmark.csv.
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from ransomdl import baselines as B  # noqa: E402
from ransomdl.config import MLFLOW_TRACKING_URI, REPORTS_DIR  # noqa: E402
from ransomdl.data import load_index, run_split  # noqa: E402
from ransomdl.experiment import MODEL_SPECS, _Prepared  # noqa: E402
from ransomdl.train_dl import pad_batch  # noqa: E402

N_REPEAT = 60


def bench(fn, seqs):
    for s in seqs[:5]:
        fn(s)
    ts = []
    for i in range(N_REPEAT):
        s = seqs[i % len(seqs)]
        t = time.perf_counter()
        fn(s)
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts)), float(np.percentile(ts, 95))


def main():
    torch.set_num_threads(1)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    res = pd.concat([pd.read_csv(f) for f in (REPORTS_DIR / "results").glob("e1*.csv")], ignore_index=True)
    res = res[res.seed == 0]
    idx = load_index()
    split = run_split(idx, 0)
    rows = []
    for _, r in res.iterrows():
        spec = MODEL_SPECS[r.model]
        if spec["kind"] == "ensemble":
            continue  # calculé plus bas : somme des membres
        try:
            prep = _Prepared(split, spec["features"], spec.get("window", 1))
            seqs = [s.copy() for s in prep.seqs["test"]]
            if spec["kind"] in ("dl", "anomaly"):
                model = mlflow.pytorch.load_model(f"runs:/{r.run_id}/model", map_location="cpu").eval()
                size_kb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024
                if spec["kind"] == "dl":
                    def fn(s, m=model):
                        with torch.no_grad():
                            x, l = pad_batch([s], [len(s)], "cpu")
                            return m(x, l)
                else:
                    def fn(s, m=model):
                        with torch.no_grad():
                            x = torch.from_numpy(s)[None]
                            return ((m(x)[:, :-1] - x[:, 1:]) ** 2).mean()
            else:
                est = mlflow.sklearn.load_model(f"runs:/{r.run_id}/model")
                for k in ("n_jobs", "nthread"):
                    try:
                        est.set_params(**{k: 1})
                    except Exception:
                        pass
                wrapper = (B.WindowModel if spec["kind"] == "window" else B.PrefixSummaryModel)(spec["est"])
                wrapper.est = est
                size_kb = len(pickle.dumps(est)) / 1024
                kw = {"cache": False} if spec["kind"] == "window" else {}
                fn = lambda s, w=wrapper, kw=kw: w.predict([s], lengths=[len(s)], **kw)  # noqa: E731
            med, p95 = bench(fn, seqs)
        except Exception as e:
            print(f"!! {r.model}: {e}")
            continue
        row = {"model": r.model, "run_id": r.run_id, "bench_latency_ms": med, "bench_latency_p95_ms": p95,
               "bench_throughput_seq_s": 1000.0 / med, "bench_model_size_kb": size_kb}
        rows.append(row)
        client = mlflow.tracking.MlflowClient()
        for k in ("bench_latency_ms", "bench_latency_p95_ms", "bench_throughput_seq_s", "bench_model_size_kb"):
            client.log_metric(r.run_id, k, row[k])
        print(f"{r.model:14s} {med:8.2f} ms  p95 {p95:8.2f} ms  {size_kb:9.1f} Ko")
    by = {row["model"]: row for row in rows}
    for _, r in res.iterrows():
        spec = MODEL_SPECS[r.model]
        if spec["kind"] == "ensemble" and all(m in by for m in spec["members"]):
            med = sum(by[m]["bench_latency_ms"] for m in spec["members"])
            row = {"model": r.model, "run_id": r.run_id, "bench_latency_ms": med,
                   "bench_latency_p95_ms": sum(by[m]["bench_latency_p95_ms"] for m in spec["members"]),
                   "bench_throughput_seq_s": 1000.0 / med,
                   "bench_model_size_kb": sum(by[m]["bench_model_size_kb"] for m in spec["members"])}
            rows.append(row)
            for k in ("bench_latency_ms", "bench_latency_p95_ms", "bench_throughput_seq_s", "bench_model_size_kb"):
                mlflow.tracking.MlflowClient().log_metric(r.run_id, k, row[k])
            print(f"{r.model:14s} {med:8.2f} ms  (somme des membres)")
    out = REPORTS_DIR / "tables"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "latency_benchmark.csv", index=False)


if __name__ == "__main__":
    main()
