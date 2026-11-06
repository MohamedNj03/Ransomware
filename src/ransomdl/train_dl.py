"""Entraînement / inférence des modèles séquentiels PyTorch (BF04, BF05).

Détection précoce : à chaque époque, chaque séquence d'entraînement est
tronquée à une fraction aléatoire f ~ U(0.02, 1) de sa longueur. Le modèle
apprend ainsi à décider à partir de n'importe quel préfixe, ce qui correspond
au protocole d'évaluation 10 % / 25 % / 50 % / 75 % / 100 %.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn

from .models import build_model, count_params


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prefix_len(T: int, frac: float) -> int:
    return max(1, int(math.ceil(T * frac)))


def pad_batch(seqs: list[np.ndarray], lens: list[int], device) -> tuple[torch.Tensor, torch.Tensor]:
    L = max(lens)
    F = seqs[0].shape[1]
    x = np.zeros((len(seqs), L, F), dtype=np.float32)
    for i, (s, l) in enumerate(zip(seqs, lens)):
        x[i, :l] = s[:l]
    return torch.from_numpy(x).to(device), torch.as_tensor(lens, dtype=torch.long, device=device)


@torch.no_grad()
def predict(model, seqs, fracs=None, lengths=None, device="cpu", batch_size=128) -> np.ndarray:
    """Probabilités ransomware sur des préfixes (fraction ou longueur absolue)."""
    model.eval()
    if lengths is None:
        lengths = [prefix_len(len(s), fracs) for s in seqs]
    lengths = [max(1, min(l, len(s))) for l, s in zip(lengths, seqs)]
    out = []
    for i in range(0, len(seqs), batch_size):
        x, l = pad_batch(seqs[i:i + batch_size], lengths[i:i + batch_size], device)
        out.append(torch.sigmoid(model(x, l)).float().cpu().numpy())
    return np.concatenate(out)


def train_model(name: str, train_seqs, y_train, val_seqs, y_val, *, n_features: int, model_kwargs=None,
                epochs=60, batch_size=32, lr=1e-3, weight_decay=1e-4, patience=12, min_frac=0.02,
                device="cuda", seed=0, log_fn=None, groups=None, feat_dropout=0.0, group_dro=False,
                dro_eta=0.05) -> tuple[nn.Module, dict]:
    """feat_dropout : masque aléatoirement des canaux de features entiers (≈ sous-échantillonnage
    de colonnes du boosting) ; group_dro : minimise la perte pondérée vers les groupes
    (applications) les plus difficiles (Sagawa et al. 2020) — deux régularisations visant
    la généralisation à des familles inconnues."""
    set_seed(seed)
    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = build_model(name, n_features, **(model_kwargs or {})).to(device)
    y_train = np.asarray(y_train, dtype=np.float32)
    pos = y_train.mean()
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / max(pos, 1e-6), device=device),
                                reduction="none")
    if group_dro:
        g_codes, g_idx = np.unique(np.asarray(groups), return_inverse=True)
        g_idx = torch.as_tensor(g_idx, device=device)
        q = torch.ones(len(g_codes), device=device) / len(g_codes)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng = np.random.default_rng(seed)
    val_fracs = [0.1, 0.25, 0.5, 1.0]
    best, best_state, bad, history = math.inf, None, 0, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(train_seqs))
        tot = 0.0
        for i in range(0, len(perm), batch_size):
            b = perm[i:i + batch_size]
            seqs = [train_seqs[j] for j in b]
            lens = [prefix_len(len(s), rng.uniform(min_frac, 1.0)) if rng.random() < 0.8 else len(s) for s in seqs]
            x, l = pad_batch(seqs, lens, device)
            if feat_dropout > 0:  # canaux masqués = valeur moyenne (entrées standardisées)
                x = x * (torch.rand(x.size(0), 1, x.size(2), device=device) >= feat_dropout).float()
            yb = torch.from_numpy(y_train[b]).to(device)
            per = crit(model(x, l), yb)
            if group_dro:
                gb = g_idx[torch.as_tensor(b, device=device)]
                n_g = torch.zeros_like(q).index_add_(0, gb, torch.ones_like(per))
                l_g = torch.zeros_like(q).index_add_(0, gb, per) / n_g.clamp(min=1)
                with torch.no_grad():
                    q = q * torch.exp(dro_eta * l_g.detach() * (n_g > 0))
                    q = q / q.sum()
                loss = (q * l_g).sum()
            else:
                loss = per.mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        # validation : perte moyenne sur plusieurs fractions (sélection orientée précocité)
        vloss = []
        for f in val_fracs:
            p = np.clip(predict(model, val_seqs, f, device=device), 1e-6, 1 - 1e-6)
            yv = np.asarray(y_val)
            vloss.append(float(-np.mean(yv * np.log(p) + (1 - yv) * np.log(1 - p))))
        v = float(np.mean(vloss))
        history.append({"epoch": ep, "train_loss": tot / len(perm), "val_loss": v})
        if log_fn:
            log_fn({"train_loss": tot / len(perm), "val_loss": v, "lr": sched.get_last_lr()[0]}, ep)
        if v < best - 1e-4:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    info = {"train_time_s": time.time() - t0, "epochs_run": ep + 1, "best_val_loss": best,
            "n_params": count_params(model), "device": str(device), "history": history}
    return model, info


@torch.no_grad()
def measure_latency(model, seqs, n=50, device="cpu") -> dict:
    """Latence d'inférence d'une séquence complète, batch=1, sur CPU (contexte EDR)."""
    m = copy.deepcopy(model).to(device).eval()
    sample = [seqs[i % len(seqs)] for i in range(n)]
    x, l = pad_batch(sample[:1], [len(sample[0])], device)
    for _ in range(5):
        m(x, l)
    ts = []
    for s in sample:
        x, l = pad_batch([s], [len(s)], device)
        t = time.perf_counter()
        m(x, l)
        ts.append((time.perf_counter() - t) * 1000)
    size_kb = sum(p.numel() * p.element_size() for p in m.parameters()) / 1024
    return {"latency_ms_cpu": float(np.median(ts)), "latency_p95_ms_cpu": float(np.percentile(ts, 95)),
            "model_size_kb": size_kb}
