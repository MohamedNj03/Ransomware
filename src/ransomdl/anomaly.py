"""Détecteur d'anomalies non supervisé -> `anomaly_probability` (section 6.2).

Un GRU apprend à prédire la seconde suivante à partir du passé, uniquement sur
des exécutions BÉNIGNES. Une activité inhabituelle (chiffrement massif, rafales
d'écritures à forte entropie...) est mal prédite : l'erreur de prédiction
moyenne sur le préfixe sert de score d'anomalie. Le score est converti en
probabilité via la fonction de répartition empirique des scores bénins de
validation (calibration non paramétrique), à fraction de séquence égale.

Aucune famille de ransomware n'est vue à l'entraînement : ce modèle est par
construction un candidat pour la détection zero-day (RQ3).
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn

from .train_dl import pad_batch, prefix_len, set_seed


class NextStepGRU(nn.Module):
    def __init__(self, n_features, hidden=64, n_layers=2, dropout=0.1):
        super().__init__()
        self.rnn = nn.GRU(n_features, hidden, n_layers, batch_first=True, dropout=dropout)
        self.out = nn.Linear(hidden, n_features)

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.out(h)  # prédiction de x[t+1] à partir de x[:t+1]


@torch.no_grad()
def _errors(model, seqs, device, batch_size=128) -> list[np.ndarray]:
    """Erreur quadratique de prédiction par pas de temps, pour chaque séquence."""
    model.eval()
    out = []
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i + batch_size]
        lens = [len(s) for s in chunk]
        x, _ = pad_batch(chunk, lens, device)
        err = ((model(x)[:, :-1] - x[:, 1:]) ** 2).mean(-1).cpu().numpy()
        out += [err[j, :max(1, l - 1)] for j, l in enumerate(lens)]
    return out


class AnomalyDetector:
    def __init__(self, seed=0, epochs=40, lr=1e-3, device="cuda"):
        self.seed, self.epochs, self.lr = seed, epochs, lr
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def fit(self, train_seqs, y_train, val_seqs, y_val):
        set_seed(self.seed)
        ben_tr = [s for s, y in zip(train_seqs, y_train) if y == 0]
        ben_va = [s for s, y in zip(val_seqs, y_val) if y == 0]
        self.model = NextStepGRU(train_seqs[0].shape[1]).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        rng = np.random.default_rng(self.seed)
        best, best_state, bad = math.inf, None, 0
        t0 = time.time()
        for ep in range(self.epochs):
            self.model.train()
            for i in range(0, len(ben_tr), 32):
                b = [ben_tr[j] for j in rng.permutation(len(ben_tr))[i:i + 32]]
                x, l = pad_batch(b, [len(s) for s in b], self.device)
                mask = (torch.arange(x.size(1) - 1, device=self.device)[None] < (l - 1)[:, None]).float()
                loss = ((((self.model(x)[:, :-1] - x[:, 1:]) ** 2).mean(-1)) * mask).sum() / mask.sum().clamp(min=1)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
            v = float(np.mean(np.concatenate(_errors(self.model, ben_va, self.device))))
            if v < best - 1e-4:
                best, best_state, bad = v, copy.deepcopy(self.model.state_dict()), 0
            else:
                bad += 1
                if bad >= 8:
                    break
        self.model.load_state_dict(best_state)
        self.train_time_s = time.time() - t0
        self.epochs_run = ep + 1
        # calibration : scores bénins de validation par durée observée (causal :
        # ne dépend pas de la durée totale, inconnue en ligne)
        val_err = _errors(self.model, ben_va, self.device)
        self.calib = {L: np.sort([self._score(e, L) for e in val_err]) for L in self.CALIB_LENGTHS}
        return self

    CALIB_LENGTHS = (3, 5, 10, 20, 40, 80, 160, 320)
    TARGET_FPR = 0.05

    @staticmethod
    def _score(err: np.ndarray, L: int) -> float:
        return float(err[:max(1, L - 1)].mean())

    def _calibrate(self, ecdf: float) -> float:
        """Rang bénin -> score monotone dont 0,5 = percentile (1 - TARGET_FPR)."""
        q = 1 - self.TARGET_FPR
        return 0.5 * ecdf / q if ecdf <= q else 0.5 + 0.5 * (ecdf - q) / (1 - q)

    def predict(self, seqs, fracs=None, lengths=None):
        if lengths is None:
            lengths = [prefix_len(len(s), fracs) for s in seqs]
        errs = _errors(self.model, seqs, self.device)
        out = []
        for e, s, L in zip(errs, seqs, lengths):
            L = max(1, min(L, len(s)))
            ref = self.calib[min(self.CALIB_LENGTHS, key=lambda k: abs(np.log(k) - np.log(L)))]
            ecdf = np.searchsorted(ref, self._score(e, L), side="right") / len(ref)
            out.append(self._calibrate(ecdf))
        return np.asarray(out)
