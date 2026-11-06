"""Modèles séquentiels PyTorch.

Entrée commune : x [B, T, F] (séquence normalisée, rembourrée de zéros) et
lengths [B] (longueur réelle). Sortie : logit [B] (score de risque ransomware).

Modèles issus de la littérature citée dans le cahier des charges :
  * LSTM, GRU               — Davidian et al. 2024, RansoGuard (Cen et al. 2024)
  * CNN 1D, CNN-LSTM        — Davidian et al. 2024, Hirano & Kobayashi (RanSMAP)
  * MHSA-LSTM               — Alqahtani et al. 2026 (LSTM + multi-head self-attention)
  * Transformer encoder     — Khan et al. 2025 (séquences de comportements)
Propositions supplémentaires :
  * TCN causal              — convolutions dilatées causales : streaming, rapide
  * Multimodal (fusion)     — un encodeur par modalité (stockage / mémoire) +
                              fusion par attention : exploite la complémentarité
                              des deux sources (question centrale de RanSMAP)
  * MLP sur fenêtres        — référence « sans ordre temporel » (sac de fenêtres)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


def length_mask(lengths: torch.Tensor, T: int) -> torch.Tensor:
    """[B, T] booléen : True sur les pas valides."""
    return torch.arange(T, device=lengths.device)[None, :] < lengths[:, None]


def masked_mean(h, mask):
    m = mask.unsqueeze(-1).float()
    return (h * m).sum(1) / m.sum(1).clamp(min=1.0)


def masked_max(h, mask):
    return h.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values


def last_valid(h, lengths):
    idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, h.size(-1))
    return h.gather(1, idx).squeeze(1)


class Head(nn.Module):
    def __init__(self, d_in, hidden=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, z):
        return self.net(z).squeeze(-1)


# ---------------------------------------------------------------------------
class WindowMLP(nn.Module):
    """MLP appliqué à chaque seconde puis pooling moyenne+max (ignore l'ordre)."""

    def __init__(self, n_features, hidden=64, dropout=0.2, **_):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.head = Head(2 * hidden, hidden, dropout)

    def forward(self, x, lengths):
        mask = length_mask(lengths, x.size(1))
        h = self.enc(x)
        return self.head(torch.cat([masked_mean(h, mask), masked_max(h, mask)], -1))


class CNN1D(nn.Module):
    def __init__(self, n_features, hidden=64, kernel=5, n_layers=3, dropout=0.2, **_):
        super().__init__()
        layers, d = [], n_features
        for _ in range(n_layers):
            layers += [nn.Conv1d(d, hidden, kernel, padding=kernel // 2), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.conv = nn.Sequential(*layers)
        self.head = Head(2 * hidden, hidden, dropout)

    def forward(self, x, lengths):
        mask = length_mask(lengths, x.size(1))
        h = self.conv((x * mask.unsqueeze(-1)).transpose(1, 2)).transpose(1, 2)
        return self.head(torch.cat([masked_mean(h, mask), masked_max(h, mask)], -1))


class RNN(nn.Module):
    """LSTM ou GRU ; état caché au dernier pas valide (causal : utilisable en streaming)."""

    def __init__(self, n_features, hidden=64, n_layers=2, dropout=0.2, cell="lstm", bidirectional=False, **_):
        super().__init__()
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(n_features, hidden, n_layers, batch_first=True,
                           dropout=dropout if n_layers > 1 else 0.0, bidirectional=bidirectional)
        d = hidden * (2 if bidirectional else 1)
        self.bidirectional = bidirectional
        self.head = Head(d, hidden, dropout)

    def encode(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
        out, h = self.rnn(packed)
        h = h[0] if isinstance(h, tuple) else h
        if self.bidirectional:
            return torch.cat([h[-2], h[-1]], -1)
        return h[-1]

    def forward(self, x, lengths):
        return self.head(self.encode(x, lengths))


class CNNLSTM(nn.Module):
    """Extraction de motifs locaux (conv causale) puis dynamique longue (LSTM)."""

    def __init__(self, n_features, hidden=64, kernel=5, dropout=0.2, **_):
        super().__init__()
        self.pad = kernel - 1
        self.conv = nn.Sequential(nn.Conv1d(n_features, hidden, kernel), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout))
        self.rnn = RNN(hidden, hidden, n_layers=1, dropout=dropout, cell="lstm")

    def forward(self, x, lengths):
        h = self.conv(F.pad(x.transpose(1, 2), (self.pad, 0))).transpose(1, 2)
        return self.rnn(h, lengths)


class MHSALSTM(nn.Module):
    """LSTM + multi-head self-attention + pooling attentionnel (Alqahtani et al. 2026)."""

    def __init__(self, n_features, hidden=64, heads=4, dropout=0.2, **_):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, 1, batch_first=True)
        self.mhsa = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.att = nn.Linear(hidden, 1)
        self.head = Head(2 * hidden, hidden, dropout)

    def forward(self, x, lengths):
        mask = length_mask(lengths, x.size(1))
        h, _ = self.lstm(x)
        a, _ = self.mhsa(h, h, h, key_padding_mask=~mask)
        h2 = self.norm(h + a)
        w = self.att(h2).squeeze(-1).masked_fill(~mask, -1e4).softmax(-1)
        pooled = (h2 * w.unsqueeze(-1)).sum(1)
        return self.head(torch.cat([pooled, last_valid(h, lengths)], -1))


class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(self, n_features, hidden=64, heads=4, n_layers=2, dropout=0.2, **_):
        super().__init__()
        self.proj = nn.Linear(n_features, hidden)
        self.pos = PositionalEncoding(hidden)
        layer = nn.TransformerEncoderLayer(hidden, heads, 2 * hidden, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = Head(2 * hidden, hidden, dropout)

    def forward(self, x, lengths):
        mask = length_mask(lengths, x.size(1))
        h = self.enc(self.pos(self.proj(x)), src_key_padding_mask=~mask)
        return self.head(torch.cat([masked_mean(h, mask), masked_max(h, mask)], -1))


class TCN(nn.Module):
    """Temporal Convolutional Network causal (Bai et al. 2018) : champ réceptif
    exponentiel avec les dilatations, aucune fuite du futur, inférence parallèle."""

    def __init__(self, n_features, hidden=64, kernel=3, n_levels=5, dropout=0.2, **_):
        super().__init__()
        self.blocks = nn.ModuleList()
        d = n_features
        for i in range(n_levels):
            dil = 2 ** i
            self.blocks.append(nn.ModuleDict({
                "c1": nn.Conv1d(d, hidden, kernel, dilation=dil),
                "c2": nn.Conv1d(hidden, hidden, kernel, dilation=dil),
                "res": nn.Conv1d(d, hidden, 1) if d != hidden else nn.Identity(),
            }))
            d = hidden
        self.kernel = kernel
        self.drop = nn.Dropout(dropout)
        self.head = Head(2 * hidden, hidden, dropout)

    def forward(self, x, lengths):
        mask = length_mask(lengths, x.size(1))
        h = x.transpose(1, 2)
        for i, b in enumerate(self.blocks):
            pad = (self.kernel - 1) * 2 ** i
            y = self.drop(F.relu(b["c1"](F.pad(h, (pad, 0)))))
            y = self.drop(F.relu(b["c2"](F.pad(y, (pad, 0)))))
            h = F.relu(y + b["res"](h))
        h = h.transpose(1, 2)
        return self.head(torch.cat([last_valid(h, lengths), masked_max(h, mask)], -1))


class MultimodalFusion(nn.Module):
    """Deux encodeurs GRU (stockage, mémoire) + fusion par attention inter-modalités.

    Les poids d'attention par modalité sont interprétables : ils indiquent
    quelle source (disque ou mémoire) a porté la décision."""

    def __init__(self, n_features, storage_idx, memory_idx, hidden=64, dropout=0.2, **_):
        super().__init__()
        self.register_buffer("s_idx", torch.as_tensor(storage_idx, dtype=torch.long))
        self.register_buffer("m_idx", torch.as_tensor(memory_idx, dtype=torch.long))
        self.enc_s = RNN(len(storage_idx), hidden, n_layers=1, cell="gru")
        self.enc_m = RNN(len(memory_idx), hidden, n_layers=1, cell="gru")
        self.att = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.head = Head(3 * hidden, hidden, dropout)
        self.last_modality_weights = None

    def forward(self, x, lengths):
        zs = self.enc_s.encode(x.index_select(-1, self.s_idx), lengths)
        zm = self.enc_m.encode(x.index_select(-1, self.m_idx), lengths)
        z = torch.stack([zs, zm], 1)                     # [B, 2, H]
        w = self.att(z).squeeze(-1).softmax(-1)          # [B, 2]
        self.last_modality_weights = w.detach()
        fused = (z * w.unsqueeze(-1)).sum(1)
        return self.head(torch.cat([fused, zs * zm, (zs - zm).abs()], -1))


class StatsGRU(nn.Module):
    """Modèle amélioré « wide & deep » : GRU (dynamique temporelle) + statistiques
    robustes du préfixe calculées dans le réseau (moyenne, écart-type, min, max,
    moyenne des 10 dernières secondes). Les statistiques agrégées sont ce qui
    rend le boosting robuste aux familles inconnues ; le GRU apporte l'ordre
    temporel. Reste causal : tout est calculé sur le préfixe observé."""

    def __init__(self, n_features, hidden=64, dropout=0.3, **_):
        super().__init__()
        self.rnn = RNN(n_features, hidden, n_layers=1, cell="gru")
        self.stats = nn.Sequential(nn.LayerNorm(5 * n_features), nn.Linear(5 * n_features, hidden), nn.ReLU(),
                                   nn.Dropout(dropout))
        self.head = Head(2 * hidden, hidden, dropout)

    @staticmethod
    def prefix_stats(x, lengths):
        mask = length_mask(lengths, x.size(1))
        m = mask.unsqueeze(-1).float()
        n = m.sum(1).clamp(min=1.0)
        mean = (x * m).sum(1) / n
        std = ((((x - mean[:, None]) ** 2) * m).sum(1) / n).clamp(min=1e-8).sqrt()
        mx = masked_max(x, mask)
        mn = -masked_max(-x, mask)
        idx = torch.arange(x.size(1), device=x.device)[None]
        last = (mask & (idx >= (lengths[:, None] - 10))).unsqueeze(-1).float()
        last_mean = (x * last).sum(1) / last.sum(1).clamp(min=1.0)
        return torch.cat([mean, std, mn, mx, last_mean], -1)

    def forward(self, x, lengths):
        z = torch.cat([self.rnn.encode(x, lengths), self.stats(self.prefix_stats(x, lengths))], -1)
        return self.head(z)


# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "mlp_window": (WindowMLP, {}),
    "cnn1d": (CNN1D, {}),
    "lstm": (RNN, {"cell": "lstm"}),
    "gru": (RNN, {"cell": "gru"}),
    "bilstm": (RNN, {"cell": "lstm", "bidirectional": True}),
    "cnn_lstm": (CNNLSTM, {}),
    "mhsa_lstm": (MHSALSTM, {}),
    "transformer": (TransformerClassifier, {}),
    "tcn": (TCN, {}),
    "multimodal": (MultimodalFusion, {}),
    "stats_gru": (StatsGRU, {}),
}

# Modèles dont la sortie au temps t ne dépend que du passé : déployables en streaming.
CAUSAL_MODELS = {"lstm", "gru", "cnn_lstm", "tcn", "multimodal", "stats_gru"}


def build_model(name: str, n_features: int, **kwargs) -> nn.Module:
    cls, defaults = MODEL_REGISTRY[name]
    return cls(n_features=n_features, **{**defaults, **kwargs})


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
