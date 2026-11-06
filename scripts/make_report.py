"""Agrège les résultats des protocoles e1/e2/e3 et produit le rapport comparatif.

Sorties (reports/) :
  * REPORT.md                     — tableaux comparatifs + lecture des résultats
  * tables/*.csv                  — tableaux complets (moyenne, écart-type)
  * figures/*.png                 — courbes de détection précoce, efficacité, zero-day
Les mêmes fichiers sont enregistrés dans l'expérience MLflow « RanSMAP_Summary »,
et le meilleur modèle E1 est inscrit au Model Registry (alias « champion »).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from ransomdl.config import FRACTIONS, RANSOMWARE_FAMILIES, REPORTS_DIR  # noqa: E402
from ransomdl.experiment import MODEL_SPECS, setup_mlflow  # noqa: E402

RES = REPORTS_DIR / "results"
FIG = REPORTS_DIR / "figures"
TAB = REPORTS_DIR / "tables"
PCTS = [int(f * 100) for f in FRACTIONS]

# Palette de référence (skill dataviz) : rôles fixes, jamais recyclés.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
KIND_COLOR = {"classique": "#2a78d6", "deep learning": "#eb6834", "non supervisé": "#1baf7a", "ensemble": "#eda100"}
CONTEXT = "#c8c7c2"
INK, INK2 = "#0b0b0b", "#52514e"
KIND_LABEL = {"window": "classique", "summary": "classique", "dl": "deep learning", "anomaly": "non supervisé",
              "ensemble": "ensemble"}

NAMES = {
    "hirano_rf": "RF fenêtres (Hirano, RanSMAP)", "hirano_svm": "SVM fenêtres (Hirano)", "hirano_knn": "kNN fenêtres (Hirano)",
    "sum_logreg": "Régression logistique", "sum_rf": "Random Forest", "sum_lightgbm": "LightGBM", "sum_xgboost": "XGBoost",
    "mlp_window": "MLP (sans ordre)", "cnn1d": "CNN 1D", "lstm": "LSTM", "gru": "GRU", "bilstm": "BiLSTM",
    "cnn_lstm": "CNN-LSTM", "mhsa_lstm": "MHSA-LSTM", "transformer": "Transformer", "tcn": "TCN causal",
    "multimodal": "Fusion multimodale", "gru_anomaly": "GRU anomalies (bénin seul)",
    "stats_gru": "Stats-GRU hybride", "ensemble_mm_lgbm": "Ensemble multimodal + LightGBM",
    "multimodal_mixaug": "Fusion multimodale + mix synthétique", "lgbm_mixaug": "LightGBM + mix synthétique",
    "ensemble_mixaug": "Ensemble + mix synthétique",
    "gru_featdrop": "GRU + dropout de features", "gru_dro": "GRU + Group DRO",
    "gru_storage": "GRU — stockage seul", "gru_memory": "GRU — mémoire seule", "gru_hirano23": "GRU — 23 features Hirano",
}
ABLATIONS = ["gru_storage", "gru_memory", "gru_hirano23", "gru_featdrop", "gru_dro"]


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#d9d8d4")
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color="#ecebe8", linewidth=0.8)
    ax.set_axisbelow(True)


def agg(df: pd.DataFrame, by="model") -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number]).drop(columns=["seed"], errors="ignore")
    g = num.groupby(df[by])
    out = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_std")).join(g.size().rename("n"))
    out["kind"] = df.groupby(by)["kind"].first()
    out["label"] = [NAMES.get(m, m) for m in out.index]
    return out


def fmt(m, s, d=3):
    if pd.isna(m):
        return "—"
    return f"{m:.{d}f} ± {s:.{d}f}" if not pd.isna(s) else f"{m:.{d}f}"


def col(a, name, d=3):
    return [fmt(a.loc[i].get(f"{name}_mean"), a.loc[i].get(f"{name}_std"), d) for i in a.index]


# ---------------------------------------------------------------------------
def e1_section(df: pd.DataFrame, lines: list) -> tuple[pd.DataFrame, str]:
    a = agg(df)
    a["early_score_mean"] = a[[f"test_f1_p{p}_mean" for p in PCTS]].mean(1)
    bench = TAB / "latency_benchmark.csv"
    if bench.exists():  # mesures à conditions identiques (scripts/benchmark_latency.py)
        b = pd.read_csv(bench).set_index("model")
        a = a.join(b[["bench_latency_ms", "bench_latency_p95_ms", "bench_model_size_kb"]])
        a["lat_ms"] = a["bench_latency_ms"].fillna(a["latency_ms_cpu_mean"])
    else:
        a["lat_ms"] = a["latency_ms_cpu_mean"]
    main = a[~a.index.isin(ABLATIONS)].sort_values("early_score_mean", ascending=False)
    t = pd.DataFrame({
        "Modèle": main["label"], "Type": main["kind"].map(KIND_LABEL),
        **{f"F1 @{p}%": col(main, f"test_f1_p{p}") for p in PCTS},
        "FPR @25%": col(main, "test_fpr_p25"), "AUC @25%": col(main, "test_auc_p25"),
        "Score précoce": [f"{v:.3f}" for v in main["early_score_mean"]],
    })
    t.to_csv(TAB / "e1_early_detection.csv", index=False)
    lines += ["## E1 — Détection précoce (RQ1, RQ2)", "",
              f"Découpage par exécution (60/20/20, stratifié application × matériel), {int(main['n'].max())} graines. "
              "Moyenne ± écart-type. *Score précoce* = moyenne des F1 sur les 5 fractions.", "",
              t.to_markdown(index=False), "",
              "![Courbes de détection précoce](figures/e1_early_detection_curves.png)", "",
              "![F1 à 10 %](figures/e1_f1_at_10.png)", ""]

    # fiabilité / opérationnel / efficacité
    r = pd.DataFrame({
        "Modèle": main["label"],
        "Seuil op.": col(main, "operating_threshold", 2),
        "Rappel op. @25%": col(main, "test_op_recall_p25"), "FPR op. @25%": col(main, "test_op_fpr_p25"),
        "Délai médian (s)": col(main, "test_op_ttd_median_s", 1),
        "Détection en ligne": col(main, "test_op_online_detection_rate"),
        "Fausses alertes en ligne": col(main, "test_op_online_false_alarm_rate"),
        "Brier @100%": col(main, "test_brier_p100"), "ECE @100%": col(main, "test_ece_p100"),
    })
    r.to_csv(TAB / "e1_reliability.csv", index=False)
    lines += ["### Fiabilité et fonctionnement en ligne", "",
              "Le score est recalculé toutes les 5 s pendant l'exécution. Seuil opérationnel = plus bas seuil "
              "gardant ≤ 5 % de fausses alertes en ligne sur la validation. Délai = temps avant la première "
              "alerte sur les ransomwares détectés ; Brier/ECE = qualité de calibration du score de risque.", "",
              r.to_markdown(index=False), ""]

    ext = pd.DataFrame({
        "Modèle": main["label"],
        "Variantes Conti — rappel @25%": col(main, "variants_recall_p25"),
        "Variantes Conti — rappel @100%": col(main, "variants_recall_p100"),
        "Mix (rans.+bénin) — rappel @25%": col(main, "mix_recall_p25"),
        "Mix — rappel @100%": col(main, "mix_recall_p100"),
        "Matériel inédit — F1 @25%": col(main, "extra_f1_p25"),
        "Matériel inédit — F1 @100%": col(main, "extra_f1_p100"),
        "Matériel inédit — FPR @100%": col(main, "extra_fpr_p100"),
    })
    ext.to_csv(TAB / "e1_external_tests.csv", index=False)
    lines += ["### Tests externes (jamais vus à l'entraînement)", "",
              "*Variantes* : 7 variantes de Conti ; *Mix* : ransomware exécuté en même temps qu'une application "
              "bénigne ; *Matériel inédit* : CPU i5/i7 gen12 et DDR5 (sous-ensemble `extra`).", "",
              ext.to_markdown(index=False), ""]

    eff = pd.DataFrame({
        "Modèle": main["label"], "Paramètres": [f"{v:,.0f}" if not pd.isna(v) else "—" for v in main.get("n_params_mean", np.nan)],
        "Taille (Ko)": col(main, "model_size_kb", 0), "Entraînement (s)": col(main, "train_time_s", 1),
        "Latence CPU 1 thread (ms)": [f"{v:.2f}" if not pd.isna(v) else "—" for v in main["lat_ms"]],
        "Latence p95 (ms)": [f"{v:.2f}" if not pd.isna(v) else "—" for v in main.get("bench_latency_p95_ms", pd.Series(np.nan, index=main.index))],
    })
    eff.to_csv(TAB / "e1_efficiency.csv", index=False)
    lines += ["### Efficacité", "", "Latence = inférence d'une séquence complète (≈ 200 s d'activité), batch 1, CPU mono-thread, mesurée "
              "à conditions identiques pour tous les modèles (`scripts/benchmark_latency.py`). Entraînement sur GPU (RTX 3050) "
              "pour le deep learning, CPU pour les modèles classiques.", "",
              eff.to_markdown(index=False), ""]

    abl = a[a.index.isin(ABLATIONS + ["gru", "hirano_rf"])].sort_values("early_score_mean", ascending=False)
    ta = pd.DataFrame({"Variante": abl["label"], **{f"F1 @{p}%": col(abl, f"test_f1_p{p}") for p in PCTS},
                       "Matériel inédit F1 @100%": col(abl, "extra_f1_p100")})
    ta.to_csv(TAB / "e1_ablation.csv", index=False)
    lines += ["### Ablation : modalités et features", "",
              "Même architecture (GRU), entrées différentes : stockage seul, mémoire seule, les 23 features "
              "du papier RanSMAP, et notre jeu étendu (32 features).", "", ta.to_markdown(index=False), ""]

    # ---- figures
    fig_early_curves(main, a)
    fig_bar(main[main.kind != "anomaly"], "test_f1_p10", "F1 à 10 % de la séquence observée (≈ 20 s)", "e1_f1_at_10.png")
    return a, main.index.tolist()


def fig_early_curves(main, a):
    sup = main[main.kind != "anomaly"]  # le non supervisé (F1 ≈ 0,2) écraserait l'échelle : voir tableaux
    best_all = sup.index[:1].tolist()
    best_dl = sup[sup.kind == "dl"].index[:1].tolist()
    best_cl = sup[sup.kind.isin(["summary", "window"])].index[:1].tolist()
    hi = list(dict.fromkeys(best_all + best_dl + best_cl + (["hirano_rf"] if "hirano_rf" in a.index else [])))[:4]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for ax, metric, title in ((axes[0], "f1", "F1"), (axes[1], "fpr", "Taux de faux positifs")):
        for m in sup.index:
            if m not in hi:
                ax.plot(PCTS, [main.loc[m, f"test_{metric}_p{p}_mean"] for p in PCTS], color=CONTEXT, lw=1, zorder=1)
        for c, m in zip(SERIES, hi):
            mu = np.array([a.loc[m, f"test_{metric}_p{p}_mean"] for p in PCTS])
            sd = np.nan_to_num(np.array([a.loc[m, f"test_{metric}_p{p}_std"] for p in PCTS]))
            ax.fill_between(PCTS, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1), color=c, alpha=0.12, lw=0, zorder=2)
            ax.plot(PCTS, mu, color=c, lw=2, marker="o", ms=5, mec="white", mew=1.5, label=NAMES.get(m, m), zorder=3)
        style(ax)
        ax.set_xticks(PCTS)
        ax.set_xticklabels([f"{p} %" for p in PCTS])
        ax.set_xlabel("Fraction de la séquence observée", color=INK2, fontsize=8)
        ax.set_title(title, loc="left", color=INK, fontsize=10)
    lo = min(sup[f"test_f1_p{p}_mean"].min() for p in PCTS)
    axes[0].set_ylim(max(0.0, lo - 0.03), 1.005)
    axes[0].legend(fontsize=7.5, frameon=False, loc="lower right")
    axes[1].plot([], [], color=CONTEXT, lw=1, label="autres modèles")
    axes[1].legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "e1_early_detection_curves.png", dpi=150)
    plt.close(fig)


def fig_bar(main, metric, title, fname):
    d = main.sort_values(f"{metric}_mean")
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(d) + 1.0))
    colors = [KIND_COLOR[KIND_LABEL[k]] for k in d.kind]
    ax.barh(d["label"], d[f"{metric}_mean"], xerr=d[f"{metric}_std"].fillna(0), color=colors, height=0.62,
            error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2})
    for y, (v, e) in enumerate(zip(d[f"{metric}_mean"], d[f"{metric}_std"].fillna(0))):
        ax.text(v + e + 0.004, y, f"{v:.3f}", va="center", fontsize=7, color=INK2)
    lo = max(0.0, d[f"{metric}_mean"].min() - 0.1)
    ax.set_xlim(lo, 1.04)
    style(ax)
    ax.grid(axis="x", color="#ecebe8")
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=k) for k, c in KIND_COLOR.items() if (d.kind.map(KIND_LABEL) == k).any()]
    ax.legend(handles=handles, fontsize=7.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=150)
    plt.close(fig)


def fig_efficiency(g):
    """Score global (fiabilité, données jamais vues) vs latence CPU ; étiquettes sélectives."""
    d = g[g["Type"] != "non supervisé"].dropna(subset=["Latence CPU (ms)"])
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for k, c in KIND_COLOR.items():
        dk = d[d["Type"] == k]
        if not dk.empty:
            ax.scatter(dk["Latence CPU (ms)"], dk["Score global"], s=60, color=c, edgecolor="white", lw=1.5,
                       label=k, zorder=3)
    label = list(d.index[:4]) + [m for m in ("hirano_rf", "hirano_svm", "sum_logreg", "cnn1d", "gru") if m in d.index]
    for m in dict.fromkeys(label):
        r = d.loc[m]
        ax.annotate(NAMES.get(m, m), (r["Latence CPU (ms)"], r["Score global"]), fontsize=7, color=INK2,
                    xytext=(5, -3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlim(d["Latence CPU (ms)"].min() / 1.6, d["Latence CPU (ms)"].max() * 8)
    style(ax)
    ax.set_xlabel("Latence d'inférence par séquence complète (ms, CPU 1 thread, échelle log)", fontsize=8, color=INK2)
    ax.set_ylabel("Score global de fiabilité", fontsize=8, color=INK2)
    ax.set_title("Efficacité vs fiabilité", loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "efficiency_vs_reliability.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def e2_section(df: pd.DataFrame, lines: list):
    df = df.copy()
    df["family_out"] = df["split"].str.extract(r"lofo_([A-Za-z]+)_")[0]
    rows = []
    for (m, fam), g in df.groupby(["model", "family_out"]):
        rows.append({"model": m, "family": fam, **{f"recall_p{p}": g[f"test_recall_p{p}"].mean() for p in PCTS},
                     "fpr_p25": g["test_fpr_p25"].mean(), "fpr_p100": g["test_fpr_p100"].mean(),
                     "op_recall_p25": g.get("test_op_recall_p25", pd.Series(np.nan)).mean()})
    r = pd.DataFrame(rows)
    piv25 = r.pivot(index="model", columns="family", values="recall_p25")
    piv100 = r.pivot(index="model", columns="family", values="recall_p100")
    summ = r.groupby("model").agg(**{f"rappel_moyen_p{p}": (f"recall_p{p}", "mean") for p in PCTS},
                                  rappel_min_p100=("recall_p100", "min"), fpr_moyen_p25=("fpr_p25", "mean"),
                                  fpr_moyen_p100=("fpr_p100", "mean"))
    summ = summ.sort_values("rappel_moyen_p25", ascending=False)
    summ.to_csv(TAB / "e2_unknown_family_summary.csv")
    piv25.to_csv(TAB / "e2_recall_p25_by_family.csv")
    piv100.to_csv(TAB / "e2_recall_p100_by_family.csv")
    t = summ.copy()
    t.index = [NAMES.get(m, m) for m in t.index]
    lines += ["## E2 — Famille inconnue / zero-day (RQ3)", "",
              "Leave-one-family-out : la famille testée (et pour Conti, ses 7 variantes) est totalement absente "
              "de l'entraînement et de la validation. Rappel = part des exécutions de la famille inconnue détectées ; "
              "FPR mesuré sur les exécutions bénignes de test.", "",
              t.round(3).to_markdown(), "",
              "![Rappel par famille inconnue](figures/e2_unknown_family_heatmap.png)", "",
              "Rappel @25 % par famille retirée :", "",
              piv25.rename(index=NAMES).round(3).to_markdown(), ""]
    # heatmap
    order = summ.index.tolist()
    fams = [f for f in RANSOMWARE_FAMILIES if f in piv25.columns]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.34 * len(order) + 1.4), sharey=True)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#f0efec", "#86b6ef", "#2a78d6", "#104281"])
    for ax, piv, title in ((axes[0], piv25, "Rappel à 25 %"), (axes[1], piv100, "Rappel à 100 %")):
        M = piv.loc[order, fams].to_numpy()
        ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for (i, j), v in np.ndenumerate(M):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.6 else INK)
        ax.set_xticks(range(len(fams)))
        ax.set_xticklabels(fams, fontsize=8)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([NAMES.get(m, m) for m in order], fontsize=7.5)
        ax.set_title(title + " — famille jamais vue", loc="left", fontsize=10, color=INK)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "e2_unknown_family_heatmap.png", dpi=150)
    plt.close(fig)
    return summ


def e3_section(df: pd.DataFrame, lines: list):
    df = df.copy()
    df["train_cpu"] = df["split"].str.extract(r"hw_train_(.+)_seed")[0]
    rows = []
    for (m, cpu), g in df.groupby(["model", "train_cpu"]):
        rows.append({"model": m, "train_cpu": cpu, "f1_p25": g["test_f1_p25"].mean(), "f1_p100": g["test_f1_p100"].mean(),
                     "fpr_p100": g["test_fpr_p100"].mean(),
                     "extra_f1_p100": g["extra_hw_f1_p100"].mean() if "extra_hw_f1_p100" in g else np.nan})
    r = pd.DataFrame(rows)
    piv = r.pivot(index="model", columns="train_cpu", values=["f1_p25", "f1_p100", "extra_f1_p100"])
    piv.columns = [f"{a} (train {b})" for a, b in piv.columns]
    piv["moyenne_f1_p100"] = r.groupby("model")["f1_p100"].mean()
    piv = piv.sort_values("moyenne_f1_p100", ascending=False)
    piv.to_csv(TAB / "e3_unknown_hardware.csv")
    lines += ["## E3 — Matériel inconnu (robustesse)", "",
              "Entraînement sur un seul CPU (toutes ses configurations RAM), test sur l'autre CPU et sur le "
              "matériel `extra` (i5/i7 gen12, DDR5). Mesure la dépendance des motifs d'accès au matériel.", "",
              piv.rename(index=NAMES).round(3).to_markdown(), ""]
    return r.groupby("model")[["f1_p25", "f1_p100"]].mean()


def e4_section(df: pd.DataFrame, lines: list):
    df = df.copy()
    df["app_out"] = df["split"].str.extract(r"lobo_([A-Za-z]+)_")[0]
    piv = df.pivot_table(index="model", columns="app_out", values="test_fpr_p100", aggfunc="mean")
    piv25 = df.pivot_table(index="model", columns="app_out", values="test_fpr_p25", aggfunc="mean")
    summ = pd.DataFrame({"FPR moyen @25%": piv25.mean(1), "FPR moyen @100%": piv.mean(1),
                         "FPR max @100%": piv.max(1),
                         "Rappel rans. @25%": df.groupby("model")["test_recall_p25"].mean()}).sort_values("FPR moyen @100%")
    summ.to_csv(TAB / "e4_unknown_benign_summary.csv")
    piv.to_csv(TAB / "e4_fpr_p100_by_app.csv")
    lines += ["## E4 — Application bénigne inconnue (faux positifs)", "",
              "Une application légitime est retirée de l'entraînement ; on mesure la part de ses exécutions "
              "signalées à tort (FPR). AESCrypt (chiffrement), SDelete (effacement sécurisé) et Zip (compression) "
              "imitent le comportement d'un ransomware : ce sont les cas difficiles.", "",
              summ.rename(index=NAMES).round(3).to_markdown(), "",
              "FPR @100 % par application retirée :", "", piv.rename(index=NAMES).round(3).to_markdown(), ""]
    return summ


AXES = ["Précocité (E1)", "Mix @25 % (E1)", "Famille inconnue @25 % (E2)", "Matériel inconnu F1 @25 % (E3)",
        "1 − FPR bénin inconnu (E4)"]


def global_section(a1, s2, s3, s4, lines: list) -> pd.DataFrame:
    """Classement multi-critères : moyenne des 5 axes de fiabilité (tous protocoles)."""
    g = pd.DataFrame({
        AXES[0]: a1["early_score_mean"], AXES[1]: a1["mix_recall_p25_mean"],
        AXES[2]: s2["rappel_moyen_p25"], AXES[3]: s3["f1_p25"], AXES[4]: 1 - s4["FPR moyen @100%"],
    }).dropna()
    g["Score global"] = g[AXES].mean(1)
    g["Latence CPU (ms)"] = a1["lat_ms"].reindex(g.index)
    g["Type"] = [KIND_LABEL[MODEL_SPECS[m]["kind"]] for m in g.index]
    g = g.sort_values("Score global", ascending=False)
    g.to_csv(TAB / "global_ranking.csv")
    t = g.copy()
    t.index = [NAMES.get(m, m) for m in t.index]
    lines += ["## Classement global multi-critères", "",
              "Aucun modèle ne domine sur tous les axes : le classement moyenne cinq critères de fiabilité, "
              "chacun mesuré sur des données **jamais vues** : précocité (score précoce E1), robustesse à "
              "l'exécution simultanée d'un logiciel légitime (mix), détection d'une famille inconnue (E2), "
              "généralisation à un matériel inconnu (E3) et absence de fausse alerte sur un logiciel légitime "
              "inconnu (E4). Le détail de chaque protocole suit.", "",
              t[["Type"] + AXES + ["Score global", "Latence CPU (ms)"]].round(3).to_markdown(), "",
              "![Classement global](figures/global_ranking.png)", "",
              "![Efficacité vs fiabilité](figures/efficiency_vs_reliability.png)", ""]
    fig_efficiency(g)
    # figure : heatmap modèles x axes
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", ["#f0efec", "#86b6ef", "#2a78d6", "#104281"])
    cols = AXES + ["Score global"]
    M = g[cols].to_numpy()
    fig, ax = plt.subplots(figsize=(9.5, 0.34 * len(g) + 1.6))
    ax.imshow(M, cmap=cmap, vmin=0.4, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(M):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.8 else INK,
                fontweight="bold" if j == len(cols) - 1 else "normal")
    ax.set_xticks(range(len(cols)))
    short = ["Précocité|(E1)", "Mix|@25 % (E1)", "Famille|inconnue|@25 % (E2)", "Matériel|inconnu|@25 % (E3)",
             "Bénin|inconnu|1−FPR (E4)", "Score|global"]
    ax.set_xticklabels([t.replace("|", chr(10)) for t in short], fontsize=7.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels([NAMES.get(m, m) for m in g.index], fontsize=7.5)
    ax.axvline(len(cols) - 1.5, color="white", lw=3)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Fiabilité sur données jamais vues (1 = parfait)", loc="left", fontsize=10, color=INK, pad=28)
    fig.tight_layout()
    fig.savefig(FIG / "global_ranking.png", dpi=150)
    plt.close(fig)
    return g


def register_champion(df_e1: pd.DataFrame, ranking: list):
    """Inscrit le meilleur modèle Deep Learning du classement global (chargé par ransomdl.inference)."""
    champion = next(m for m in ranking if MODEL_SPECS[m]["kind"] == "dl")
    rows = df_e1[(df_e1.model == champion) & (df_e1.seed == 0)]
    if rows.empty:
        return None
    run_id = rows.iloc[0]["run_id"]
    client = mlflow.tracking.MlflowClient()
    try:  # idempotent : ne réinscrit pas un modèle déjà présent dans le registre
        for v in client.search_model_versions("name='ransomware-detector'"):
            if v.run_id == run_id:
                client.set_registered_model_alias("ransomware-detector", "champion", v.version)
                return f"ransomware-detector v{v.version} (alias champion, modèle {champion}, run {run_id})"
    except Exception:
        pass
    try:
        mv = mlflow.register_model(f"runs:/{run_id}/model", "ransomware-detector")
        client.set_registered_model_alias("ransomware-detector", "champion", mv.version)
        client.set_model_version_tag("ransomware-detector", mv.version, "model", champion)
        return f"ransomware-detector v{mv.version} (alias champion, modèle {champion}, run {run_id})"
    except Exception as e:  # modèle non sauvegardé pour ce run
        return f"enregistrement impossible : {e}"


def main():
    for d in (FIG, TAB):
        d.mkdir(parents=True, exist_ok=True)
    lines = ["# Détection précoce de ransomware — comparaison des modèles (RanSMAP)", "",
             "Rapport généré automatiquement par `scripts/make_report.py` à partir de `reports/results/*.csv`. "
             "Chaque ligne renvoie à des runs MLflow (voir `run_id` dans les CSV) : `mlflow ui --backend-store-uri "
             "sqlite:///mlflow.db`.", ""]
    champion, df1, ranking = None, None, []
    synth = REPORTS_DIR / "SYNTHESE.md"  # lecture humaine des résultats (rédigée à partir des tableaux)
    if synth.exists():
        lines += [synth.read_text(encoding="utf-8"), ""]
    load = lambda p: pd.concat([pd.read_csv(f) for f in sorted(RES.glob(f"{p}*.csv"))], ignore_index=True)  # noqa: E731
    l1, l2, l3, l4 = [], [], [], []
    a1 = s2 = s3 = s4 = None
    if list(RES.glob("e1*.csv")):
        df1 = load("e1")
        a1, ranking = e1_section(df1, l1)
    if list(RES.glob("e2*.csv")):
        s2 = e2_section(load("e2"), l2)
    if list(RES.glob("e3*.csv")):
        s3 = e3_section(load("e3"), l3)
    if list(RES.glob("e4*.csv")):
        s4 = e4_section(load("e4"), l4)
    if all(x is not None for x in (a1, s2, s3, s4)):
        g = global_section(a1, s2, s3, s4, lines)
        ranking = g.index.tolist()
    champion = ranking[0] if ranking else None
    lines += l1 + l2 + l3 + l4
    (REPORTS_DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:60]))

    prefix = os.environ.get("RANSOMDL_EXP_PREFIX", "")
    exp = setup_mlflow(prefix + "RanSMAP_Summary")
    mlflow.set_experiment(experiment_id=exp)
    client = mlflow.tracking.MlflowClient()
    for r in client.search_runs([exp], filter_string="attributes.run_name = 'comparaison_modeles'"):
        client.delete_run(r.info.run_id)  # un seul run de synthèse, toujours à jour
    with mlflow.start_run(run_name="comparaison_modeles"):
        mlflow.log_artifacts(str(TAB), "tables")
        mlflow.log_artifacts(str(FIG), "figures")
        mlflow.log_artifact(str(REPORTS_DIR / "REPORT.md"))
        if champion and not prefix:
            mlflow.set_tag("champion_global", champion)
            reg = register_champion(df1, ranking)
            if reg:
                mlflow.set_tag("registry", reg)
                print(reg)


if __name__ == "__main__":
    main()
