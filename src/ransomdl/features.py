"""Extraction des features temporelles à partir des CSV bruts RanSMAP.

Chaque exécution (dossier contenant ata_read/ata_write/mem_*.csv) est résumée en
une matrice [T, S] de *statistiques suffisantes* par bin d'une seconde : comptes,
sommes et sommes des carrés. On peut ensuite recalculer exactement moyennes,
variances et ratios sur n'importe quelle fenêtre de k bins (BF02/BF03) sans
relire les 133 Go de CSV.

Features inspirées de Hirano & Kobayashi (CSR 2022, RanSMAP 2024) :
  * stockage : débit lecture/écriture, variance des LBA, entropie des écritures ;
  * mémoire : nombre de pages 4 Ko / 2 Mo / MMIO lues, écrites, lues-écrites,
    exécutées, variance des adresses physiques, entropie des pages écrites.
Ajouts propres au projet (signatures connues du chiffrement en place) :
  * fraction d'écritures à forte entropie (> 0,9) ;
  * fraction d'écritures sur un LBA déjà lu auparavant (réécriture en place) ;
  * fraction d'accès séquentiels.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pc

from .config import BIN_SECONDS

COLS = ["sec", "ns", "addr", "size", "ent", "typ"]
_TYPES = {"sec": pa.int64(), "ns": pa.int64(), "addr": pa.int64(),
          "size": pa.int64(), "ent": pa.float64(), "typ": pa.int64()}

SECTOR = 512
GIB = float(1 << 30)
HIGH_ENTROPY = 0.9
PAGE_TYPES = (1, 2, 4)  # codes du champ "type" des CSV mémoire

STAT_NAMES = (
    # --- stockage : lectures
    ["ar_n", "ar_bytes", "ar_s1", "ar_s2", "ar_seq"]
    # --- stockage : écritures
    + ["aw_n", "aw_bytes", "aw_s1", "aw_s2", "aw_seq",
       "aw_ent_sum", "aw_ent_s2", "aw_ent_hi", "aw_ovw"]
    # --- mémoire : lecture / exécution (pas d'entropie)
    + [f"mr_t{t}" for t in PAGE_TYPES] + ["mr_s1", "mr_s2"]
    + [f"me_t{t}" for t in PAGE_TYPES] + ["me_s1", "me_s2"]
    # --- mémoire : écriture / lecture-écriture (avec entropie)
    + [f"mw_t{t}" for t in PAGE_TYPES] + ["mw_s1", "mw_s2", "mw_ent_n", "mw_ent_sum", "mw_ent_s2", "mw_ent_hi"]
    + [f"mx_t{t}" for t in PAGE_TYPES] + ["mx_s1", "mx_s2", "mx_ent_n", "mx_ent_sum", "mx_ent_s2", "mx_ent_hi"]
)
STAT_INDEX = {n: i for i, n in enumerate(STAT_NAMES)}
FILES = ["ata_read", "ata_write", "mem_read", "mem_write", "mem_readwrite", "mem_exec"]


def _read_table(run_dir: Path, name: str) -> dict[str, np.ndarray]:
    """Lit <name>.csv (ou <name>.zip) et renvoie des colonnes numpy."""
    csv_path = run_dir / f"{name}.csv"
    zip_path = run_dir / f"{name}.zip"
    ro = pc.ReadOptions(column_names=COLS)
    co = pc.ConvertOptions(column_types=_TYPES)
    if csv_path.exists():
        src = str(csv_path)
    elif zip_path.exists():
        with zipfile.ZipFile(zip_path) as z:
            member = next(i for i in z.infolist() if i.filename.endswith(".csv"))
            src = io.BytesIO(z.read(member))
    else:
        return {c: np.empty(0, dtype=np.int64 if c != "ent" else np.float64) for c in COLS}
    if isinstance(src, str) and Path(src).stat().st_size == 0:
        return {c: np.empty(0, dtype=np.int64 if c != "ent" else np.float64) for c in COLS}
    tb = pc.read_csv(src, read_options=ro, convert_options=co)
    return {c: tb.column(c).to_numpy() for c in COLS}


def _bincount(idx: np.ndarray, T: int, w: np.ndarray | None = None) -> np.ndarray:
    return np.bincount(idx, weights=w, minlength=T)[:T].astype(np.float64)


def _sequential(addr: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Accès dont le LBA suit immédiatement l'accès précédent."""
    seq = np.zeros(addr.shape[0], dtype=bool)
    if addr.shape[0] > 1:
        seq[1:] = addr[1:] == addr[:-1] + size[:-1] // SECTOR
    return seq


def extract_run(run_dir: Path) -> tuple[np.ndarray, dict]:
    """Renvoie (stats [T, S] float64, métadonnées) pour une exécution."""
    tabs = {name: _read_table(run_dir, name) for name in FILES}
    times = {name: t["sec"].astype(np.float64) + t["ns"].astype(np.float64) * 1e-9
             for name, t in tabs.items()}
    non_empty = [t for t in times.values() if t.size]
    t0 = min(t.min() for t in non_empty)
    t1 = max(t.max() for t in non_empty)
    T = int(np.floor((t1 - t0) / BIN_SECONDS)) + 1
    S = np.zeros((T, len(STAT_NAMES)), dtype=np.float64)
    bins = {name: np.floor((t - t0) / BIN_SECONDS).astype(np.int64) for name, t in times.items()}

    def put(stat: str, values: np.ndarray):
        S[:, STAT_INDEX[stat]] = values

    # ---------------- stockage : lectures
    r, b = tabs["ata_read"], bins["ata_read"]
    lba = r["addr"].astype(np.float64) * SECTOR / GIB
    put("ar_n", _bincount(b, T))
    put("ar_bytes", _bincount(b, T, r["size"].astype(np.float64)))
    put("ar_s1", _bincount(b, T, lba))
    put("ar_s2", _bincount(b, T, lba * lba))
    put("ar_seq", _bincount(b, T, _sequential(r["addr"], r["size"]).astype(np.float64)))

    # ---------------- stockage : écritures
    w, bw = tabs["ata_write"], bins["ata_write"]
    lbw = w["addr"].astype(np.float64) * SECTOR / GIB
    ent = np.clip(w["ent"], 0.0, 1.0)
    put("aw_n", _bincount(bw, T))
    put("aw_bytes", _bincount(bw, T, w["size"].astype(np.float64)))
    put("aw_s1", _bincount(bw, T, lbw))
    put("aw_s2", _bincount(bw, T, lbw * lbw))
    put("aw_seq", _bincount(bw, T, _sequential(w["addr"], w["size"]).astype(np.float64)))
    put("aw_ent_sum", _bincount(bw, T, ent))
    put("aw_ent_s2", _bincount(bw, T, ent * ent))
    put("aw_ent_hi", _bincount(bw, T, (ent > HIGH_ENTROPY).astype(np.float64)))
    # Réécriture en place : l'écriture touche un LBA lu plus tôt dans l'exécution
    # (strictement causal : seule l'information passée est utilisée).
    if w["addr"].size and r["addr"].size:
        rt = times["ata_read"]
        uniq, first_pos = np.unique(r["addr"], return_index=True)  # lectures triées par temps
        first_read_t = rt[first_pos]
        pos = np.searchsorted(uniq, w["addr"])
        pos_c = np.clip(pos, 0, uniq.size - 1)
        hit = (uniq[pos_c] == w["addr"]) & (first_read_t[pos_c] < times["ata_write"])
        put("aw_ovw", _bincount(bw, T, hit.astype(np.float64)))

    # ---------------- mémoire
    def mem_block(prefix: str, name: str, with_entropy: bool):
        m, bm = tabs[name], bins[name]
        gpa = m["addr"].astype(np.float64) / GIB
        for t in PAGE_TYPES:
            put(f"{prefix}_t{t}", _bincount(bm, T, (m["typ"] == t).astype(np.float64)))
        put(f"{prefix}_s1", _bincount(bm, T, gpa))
        put(f"{prefix}_s2", _bincount(bm, T, gpa * gpa))
        if with_entropy:
            has = m["size"] > 0  # seules les pages 4 Ko non-MMIO portent une entropie
            e = np.where(has, np.clip(m["ent"], 0.0, 1.0), 0.0)
            put(f"{prefix}_ent_n", _bincount(bm, T, has.astype(np.float64)))
            put(f"{prefix}_ent_sum", _bincount(bm, T, e))
            put(f"{prefix}_ent_s2", _bincount(bm, T, e * e))
            put(f"{prefix}_ent_hi", _bincount(bm, T, (has & (e > HIGH_ENTROPY)).astype(np.float64)))

    mem_block("mr", "mem_read", False)
    mem_block("me", "mem_exec", False)
    mem_block("mw", "mem_write", True)
    mem_block("mx", "mem_readwrite", True)

    meta = {
        "t0": t0,
        "duration_s": t1 - t0,
        "n_bins": T,
        **{f"n_{name}": int(tabs[name]["sec"].size) for name in FILES},
        "zipped": ",".join(n for n in FILES if (run_dir / f"{n}.zip").exists()),
    }
    return S, meta
