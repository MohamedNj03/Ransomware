"""Séquences temporelles, features dérivées et protocoles de découpage (BF02, BF03, BF06, BF07)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from .config import BENIGN_APPS, INDEX_FILE, RANSOMWARE_FAMILIES, RUNS_DIR
from .features import STAT_INDEX

# ---------------------------------------------------------------------------
# Features dérivées par pas de temps
# ---------------------------------------------------------------------------
STORAGE_FEATURES = [
    "ar_ops", "ar_tput", "ar_lba_var", "ar_seq_frac",
    "aw_ops", "aw_tput", "aw_lba_var", "aw_seq_frac",
    "aw_ent_mean", "aw_ent_std", "aw_hi_frac", "aw_ovw_frac",
]
MEMORY_FEATURES = (
    [f"mr_t{t}" for t in (1, 2, 4)] + ["mr_var"]
    + [f"me_t{t}" for t in (1, 2, 4)] + ["me_var"]
    + [f"mw_t{t}" for t in (1, 2, 4)] + ["mw_var", "mw_ent_mean", "mw_hi_frac"]
    + [f"mx_t{t}" for t in (1, 2, 4)] + ["mx_var", "mx_ent_mean", "mx_hi_frac"]
)
ALL_FEATURES = STORAGE_FEATURES + MEMORY_FEATURES

# Jeu de 23 features de Hirano & Kobayashi (5 stockage + 18 mémoire), pour
# reproduire la référence RanSMAP. Le code de type de page 2/1/4 correspond
# respectivement aux pages 4 Ko / 2 Mo / MMIO.
HIRANO23 = (
    ["ar_tput", "aw_tput", "ar_lba_var", "aw_lba_var", "aw_ent_mean"]
    + [f"mw_t{t}" for t in (2, 1, 4)] + ["mw_var", "mw_ent_mean"]
    + [f"mx_t{t}" for t in (2, 1, 4)] + ["mx_var", "mx_ent_mean"]
    + [f"mr_t{t}" for t in (2, 1, 4)] + ["mr_var"]
    + [f"me_t{t}" for t in (2, 1, 4)] + ["me_var"]
)

FEATURE_SETS = {
    "all": ALL_FEATURES,
    "storage": STORAGE_FEATURES,
    "memory": MEMORY_FEATURES,
    "hirano23": HIRANO23,
}


def _rolling(S: np.ndarray, window: int) -> np.ndarray:
    """Somme causale sur les `window` derniers bins (statistiques additives)."""
    if window <= 1:
        return S
    c = np.cumsum(np.vstack([np.zeros((1, S.shape[1])), S]), axis=0)
    idx = np.arange(1, S.shape[0] + 1)
    return c[idx] - c[np.maximum(idx - window, 0)]


def _var(n, s1, s2):
    with np.errstate(invalid="ignore", divide="ignore"):
        v = (s2 - s1 * s1 / np.maximum(n, 1)) / np.maximum(n - 1, 1)
    return np.where(n > 1, np.clip(v, 0, None), 0.0)


def _ratio(a, b):
    return np.where(b > 0, a / np.maximum(b, 1e-12), 0.0)


def derive_features(S: np.ndarray, window: int = 1) -> np.ndarray:
    """Statistiques suffisantes [T, n_stats] -> features [T, len(ALL_FEATURES)]."""
    S = _rolling(S, window)
    g = lambda k: S[:, STAT_INDEX[k]]  # noqa: E731
    out = {}
    secs = float(window)
    for p in ("ar", "aw"):
        n = g(f"{p}_n")
        out[f"{p}_ops"] = np.log1p(n / secs)
        out[f"{p}_tput"] = np.log1p(g(f"{p}_bytes") / secs / 1e6)          # Mo/s
        out[f"{p}_lba_var"] = np.log1p(_var(n, g(f"{p}_s1"), g(f"{p}_s2")))  # Gio^2
        out[f"{p}_seq_frac"] = _ratio(g(f"{p}_seq"), n)
    n = g("aw_n")
    mean = _ratio(g("aw_ent_sum"), n)
    out["aw_ent_mean"] = mean
    out["aw_ent_std"] = np.sqrt(np.clip(_ratio(g("aw_ent_s2"), n) - mean ** 2, 0, None))
    out["aw_hi_frac"] = _ratio(g("aw_ent_hi"), n)
    out["aw_ovw_frac"] = _ratio(g("aw_ovw"), n)
    for p in ("mr", "me", "mw", "mx"):
        n_all = sum(g(f"{p}_t{t}") for t in (1, 2, 4))
        for t in (1, 2, 4):
            out[f"{p}_t{t}"] = np.log1p(g(f"{p}_t{t}") / secs)
        out[f"{p}_var"] = np.log1p(_var(n_all, g(f"{p}_s1"), g(f"{p}_s2")))
        if p in ("mw", "mx"):
            ne = g(f"{p}_ent_n")
            out[f"{p}_ent_mean"] = _ratio(g(f"{p}_ent_sum"), ne)
            out[f"{p}_hi_frac"] = _ratio(g(f"{p}_ent_hi"), ne)
    X = np.stack([out[f] for f in ALL_FEATURES], axis=1).astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------
def quality_flags(idx: pd.DataFrame) -> pd.Series:
    """Nettoyage au niveau exécution : raison d'exclusion (chaîne vide = conservée).

    * capture incomplète : durée < 50 % de la médiane de son sous-ensemble ;
    * modalité absente : aucun accès disque, ou aucun accès mémoire enregistré ;
    * doublon : même contenu de features qu'une exécution déjà vue.
    """
    reason = pd.Series("", index=idx.index)
    med = idx.groupby("subset")["duration_s"].transform("median")
    reason[idx["duration_s"] < 0.5 * med] = "capture_incomplete"
    no_disk = (idx["n_ata_read"] + idx["n_ata_write"]) == 0
    no_mem = idx[["n_mem_read", "n_mem_write", "n_mem_readwrite", "n_mem_exec"]].sum(1) == 0
    reason[(reason == "") & no_disk] = "no_storage_trace"
    reason[(reason == "") & no_mem] = "no_memory_trace"
    reason[(reason == "") & idx["npz_sha1"].duplicated()] = "duplicate"
    return reason


def load_index(keep_excluded: bool = False) -> pd.DataFrame:
    parts = sorted((INDEX_FILE.parent / "index_parts").glob("*.csv"))
    idx = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True) if parts else pd.read_csv(INDEX_FILE)
    idx["hw"] = idx["cpu"] + "/" + idx["ram"]
    idx["exclude_reason"] = quality_flags(idx)
    if not keep_excluded:
        idx = idx[idx["exclude_reason"] == ""]
    return idx.reset_index(drop=True)


def index_digest(idx: pd.DataFrame) -> str:
    """Empreinte de la version de données (traçabilité MLflow)."""
    h = hashlib.sha1()
    for uid, sha in zip(idx["run_uid"], idx["npz_sha1"]):
        h.update(f"{uid}:{sha};".encode())
    return h.hexdigest()[:16]


@lru_cache(maxsize=None)
def _load_stats(run_uid: str) -> np.ndarray:
    with np.load(RUNS_DIR / f"{run_uid}.npz") as z:
        return z["stats"]


def load_sequences(run_uids, window: int = 1) -> list[np.ndarray]:
    return [derive_features(_load_stats(u), window) for u in run_uids]


def synthetic_mixes(train: pd.DataFrame, seed: int, window: int = 1, n_per_run: int = 1):
    """Augmentation « exécutions simultanées » rendue possible par les statistiques additives :
    deux processus actifs en même temps sur une machine produisent (approximativement) la
    somme de leurs comptes / sommes / sommes des carrés, seconde par seconde.

    * ransomware + bénin (même matériel)       -> étiquette 1 (le ransomware reste à détecter)
    * bénin + autre bénin (même matériel)      -> étiquette 0 (plus d'activité ≠ ransomware)
    Uniquement à partir des exécutions d'ENTRAÎNEMENT ; les vraies exécutions mixtes de
    RanSMAP (sous-ensemble `mix`) ne servent qu'au test.
    """
    rng = np.random.default_rng(seed + 1000)
    seqs, labels, groups = [], [], []
    for _, r in train.iterrows():
        pool = train[(train.hw == r.hw) & (train.label == 0) & (train.app != r.app)]
        if pool.empty:
            continue
        for b in pool.sample(n_per_run, random_state=int(rng.integers(1 << 31)), replace=len(pool) < n_per_run).itertuples():
            S1, S2 = _load_stats(r.run_uid), _load_stats(b.run_uid)
            T = min(len(S1), len(S2))
            seqs.append(derive_features(S1[:T] + S2[:T], window))
            labels.append(int(r.label))
            groups.append(f"mix_{r.app}_{b.app}")
    return seqs, np.asarray(labels), np.asarray(groups)


# ---------------------------------------------------------------------------
# Normalisation (ajustée sur l'entraînement uniquement)
# ---------------------------------------------------------------------------
@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, seqs: list[np.ndarray]) -> "Scaler":
        X = np.concatenate(seqs, axis=0)
        std = X.std(0)
        return cls(X.mean(0), np.where(std < 1e-6, 1.0, std))

    def transform(self, seqs: list[np.ndarray]) -> list[np.ndarray]:
        return [np.clip((s - self.mean) / self.std, -10, 10).astype(np.float32) for s in seqs]


# ---------------------------------------------------------------------------
# Protocoles de découpage — toujours PAR EXÉCUTION (jamais par fenêtre)
# ---------------------------------------------------------------------------
@dataclass
class Split:
    name: str
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    extra_tests: dict  # nom -> DataFrame (tests externes : variantes, mix, matériel inconnu...)

    def describe(self) -> dict:
        d = {"split": self.name, "n_train": len(self.train), "n_val": len(self.val), "n_test": len(self.test)}
        for k, v in self.extra_tests.items():
            d[f"n_test_{k}"] = len(v)
        return d


def _stratified_run_split(df: pd.DataFrame, seed: int, val_frac=0.2, test_frac=0.2):
    """Découpe chaque groupe (application x configuration matérielle) par exécution."""
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for _, g in df.groupby(["app", "hw"], sort=True):
        ids = g.index.to_numpy().copy()
        rng.shuffle(ids)
        n = len(ids)
        n_te = max(1, int(round(n * test_frac))) if test_frac > 0 else 0
        n_va = max(1, int(round(n * val_frac))) if n - n_te > 1 else 0
        te += list(ids[:n_te]); va += list(ids[n_te:n_te + n_va]); tr += list(ids[n_te + n_va:])
    return df.loc[tr], df.loc[va], df.loc[te]


def run_split(idx: pd.DataFrame, seed: int) -> Split:
    """E1 — détection précoce : 60/20/20 par exécution sur `original`.
    Tests externes : variantes Conti, exécutions simultanées (mix), matériel inédit (extra)."""
    orig = idx[idx.subset == "original"]
    tr, va, te = _stratified_run_split(orig, seed)
    extra = {k: idx[idx.subset == k] for k in ("variants", "mix", "extra") if (idx.subset == k).any()}
    return Split(f"run_split_seed{seed}", tr, va, te, extra)


def family_split(idx: pd.DataFrame, family: str, seed: int) -> Split:
    """E2 — famille inconnue (zero-day) : toutes les exécutions de `family` (et
    toutes ses variantes) sont exclues de l'entraînement et de la validation.
    Test = famille retirée + bénins de test (pour mesurer les faux positifs)."""
    orig = idx[idx.subset == "original"]
    known = orig[orig.family != family]
    tr, va, te_known = _stratified_run_split(known, seed)
    held = idx[(idx.family == family) & idx.subset.isin(["original", "variants"])]
    benign_test = te_known[te_known.label == 0]
    test = pd.concat([held, benign_test])
    extra = {"known_families_test": te_known}
    return Split(f"lofo_{family}_seed{seed}", tr, va, test, extra)


def benign_app_split(idx: pd.DataFrame, app: str, seed: int) -> Split:
    """E4 — application bénigne inconnue : `app` est absente de l'entraînement ;
    on mesure les faux positifs sur elle (test = app retirée + ransomwares de test)."""
    orig = idx[idx.subset == "original"]
    known = orig[orig.app != app]
    tr, va, te_known = _stratified_run_split(known, seed)
    held = orig[orig.app == app]
    test = pd.concat([held, te_known[te_known.label == 1]])
    return Split(f"lobo_{app}_seed{seed}", tr, va, test, {})


def hardware_split(idx: pd.DataFrame, seed: int, train_cpu="i3-gen12") -> Split:
    """E3 — matériel inconnu : entraînement sur un CPU, test sur l'autre CPU + `extra`."""
    orig = idx[idx.subset == "original"]
    src = orig[orig.cpu == train_cpu]
    tr, va, _ = _stratified_run_split(src, seed, val_frac=0.2, test_frac=0.0)
    test = orig[orig.cpu != train_cpu]
    extra = {"extra_hw": idx[idx.subset == "extra"]} if (idx.subset == "extra").any() else {}
    return Split(f"hw_train_{train_cpu}_seed{seed}", tr, va, test, extra)


ALL_FAMILIES = RANSOMWARE_FAMILIES
ALL_BENIGN = BENIGN_APPS
