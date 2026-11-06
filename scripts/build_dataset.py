"""Construit le dataset de features à partir du dépôt RanSMAP (BF01-BF03).

Le dépôt complet fait ~133 Go décompressé : on le traite en flux, une
configuration matérielle à la fois.
  1. `git sparse-checkout set <config>` récupère uniquement les CSV de la config ;
  2. chaque exécution est résumée en statistiques par seconde (features.py)
     et sauvegardée dans data/processed/runs/<run_uid>.npz ;
  3. les CSV et les packs git récupérés sont supprimés avant la config suivante.
Le script est reprenable : les configs déjà traitées sont sautées.

Usage : python scripts/build_dataset.py [--subsets variants mix original extra] [--workers 8]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import stat
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from ransomdl.config import PROCESSED_DIR, RAW_DATASET, RAW_REPO, RUNS_DIR, INDEX_FILE, family_of, label_of  # noqa: E402
from ransomdl.features import STAT_NAMES, extract_run  # noqa: E402

PARTS_DIR = PROCESSED_DIR / "index_parts"


def git(*args: str, check: bool = True) -> str:
    out = subprocess.run(["git", "-C", str(RAW_REPO), *args], capture_output=True, text=True)
    if check and out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {out.stderr}")
    return out.stdout


def list_configs(subsets: list[str]) -> list[str]:
    cfgs = []
    for s in subsets:
        for cpu in git("ls-tree", "-d", "--name-only", "HEAD", f"dataset/{s}/").split():
            cfgs += git("ls-tree", "-d", "--name-only", "HEAD", f"{cpu}/").split()
    return cfgs


def blob_shas(cfg: str) -> dict[str, str]:
    """SHA git de chaque fichier source : traçabilité exacte des données brutes."""
    shas = {}
    for line in git("ls-tree", "-r", "HEAD", f"{cfg}/").splitlines():
        meta, path = line.split("\t")
        shas[path] = meta.split()[2]
    return shas


def pack_files() -> set[Path]:
    return set((RAW_REPO / ".git" / "objects" / "pack").glob("pack-*"))


def run_uid(rel: str) -> str:
    # dataset/<subset>/<cpu>/<ram>/<app>/<run>  ->  subset__cpu__ram__run
    p = rel.split("/")
    return "__".join([p[1], p[2], p[3], p[5]])


def process_one(run_dir: str) -> dict:
    run_dir = Path(run_dir)
    rel = run_dir.relative_to(RAW_REPO).as_posix()
    p = rel.split("/")
    subset, cpu, ram, app, run = p[1], p[2], p[3], p[4], p[5]
    uid = run_uid(rel)
    out = RUNS_DIR / f"{uid}.npz"
    t = time.time()
    stats, meta = extract_run(run_dir)
    np.savez_compressed(out, stats=stats.astype(np.float64))
    return {
        "run_uid": uid, "subset": subset, "cpu": cpu, "ram": ram, "app": app, "run": run,
        "label": label_of(app), "family": family_of(app), "rel_path": rel,
        "npz_sha1": hashlib.sha1(out.read_bytes()).hexdigest(), "extract_s": round(time.time() - t, 2),
        **meta,
    }


def checkout(cfg: str, retries: int = 5):
    for attempt in range(1, retries + 1):
        try:
            git("sparse-checkout", "set", cfg)
            missing = [d for d in (RAW_REPO / cfg).glob("*/*") if d.is_dir() and not any(d.iterdir())]
            if not missing:
                return
        except RuntimeError as e:
            print(f"  checkout attempt {attempt} failed: {str(e)[:200]}", flush=True)
        time.sleep(10 * attempt)
    raise RuntimeError(f"checkout of {cfg} failed after {retries} attempts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="+", default=["original", "variants", "mix", "extra"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    git("config", "transfer.unpackLimit", "1")  # objets récupérés toujours en pack (supprimables)
    base_packs = pack_files()
    head = git("rev-parse", "HEAD").strip()

    for cfg in list_configs(args.subsets):
        part = PARTS_DIR / (cfg.replace("/", "__") + ".csv")
        if part.exists():
            print(f"[skip] {cfg}", flush=True)
            continue
        t = time.time()
        print(f"[fetch] {cfg}", flush=True)
        checkout(cfg)
        shas = blob_shas(cfg)
        runs = sorted(d for d in (RAW_REPO / cfg).glob("*/*") if d.is_dir())
        print(f"  {len(runs)} runs fetched in {time.time() - t:.0f}s", flush=True)
        rows = []
        with ProcessPoolExecutor(args.workers) as ex:
            futs = {ex.submit(process_one, str(r)): r for r in runs}
            for f in as_completed(futs):
                try:
                    rows.append(f.result())
                except Exception as e:  # BNF05 : on trace l'exécution défectueuse sans tout arrêter
                    print(f"  !! {futs[f]}: {e}", flush=True)
        for row in rows:
            row["src_blob_shas"] = ";".join(f"{Path(k).name}={v}" for k, v in shas.items()
                                            if k.startswith(row["rel_path"] + "/"))
            row["ransmap_commit"] = head
        pd.DataFrame(rows).sort_values("run_uid").to_csv(part, index=False)
        # nettoyage : CSV du worktree + packs git récupérés pour cette config
        git("sparse-checkout", "set", "_none_")
        for pf in pack_files() - base_packs:
            os.chmod(pf, stat.S_IWRITE)  # git marque ses packs en lecture seule sous Windows
            pf.unlink(missing_ok=True)
        print(f"[done] {cfg}: {len(rows)}/{len(runs)} runs in {time.time() - t:.0f}s", flush=True)

    parts = sorted(PARTS_DIR.glob("*.csv"))
    index = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    index.to_csv(INDEX_FILE, index=False)
    (PROCESSED_DIR / "stat_names.txt").write_text("\n".join(STAT_NAMES))
    print(f"index: {len(index)} runs -> {INDEX_FILE}")


if __name__ == "__main__":
    main()
