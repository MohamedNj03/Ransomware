# Détection précoce de ransomware par Deep Learning — volet Deep Learning

Projet semestriel TEK-UP (cahier des charges : `cahier_de_charge_ransomware.pdf`).
Ce dépôt couvre le **cœur obligatoire** (objectifs 1 à 6, BF01–BF07) : traces
comportementales RanSMAP → prétraitement → séquences temporelles → modèles →
score de risque → détection précoce (10/25/50/75/100 %) et famille inconnue,
avec une **traçabilité complète dans MLflow**. La sortie structurée du détecteur
(`ransomware_probability`, `anomaly_probability`, indicateurs) est prête pour
l'extension Hermes + JEV.

Résultats détaillés : **[REPORT.md](REPORT.md)** ·
qualité des données : **[DATA_QUALITY.md](DATA_QUALITY.md)**.

### Résultats clés (506 runs MLflow, données jamais vues à l'entraînement)

| | F1 @10 % (≈ 20 s) | Mix @25 % | Famille inconnue @25 % | Matériel inconnu F1 @25 % | FPR logiciel inconnu | Latence CPU |
|---|---|---|---|---|---|---|
| **Ensemble + mix synthétique** (meilleur global) | 0,972 | 0,960 | 0,783 | 0,988 | 4 % | 7,7 ms |
| **Fusion multimodale + mix synthétique** (champion DL) | 0,954 | 0,974 | 0,797 | 0,960 | 10 % | 7,0 ms |
| LightGBM (meilleure baseline classique) | 0,962 | 0,538 | 0,823 | 0,988 | 4 % | 0,5 ms |
| GRU (DL de référence) | 0,941 | 0,470 | 0,663 | 0,951 | 2 % | 7,0 ms |
| RF fenêtres — méthode du papier RanSMAP | 0,887 | 0,870 | 0,561 | 0,850 | 37 % | 0,5 ms |

* **RQ1** : détection fiable dès 10 % de la séquence (≈ 20 s) ; délai médian avant alerte en ligne 10–17 s.
* **RQ2** : au-delà de 25 %, tous les modèles ≥ 0,99 de F1 ; le coût de la précocité est en faux positifs (3–6 % à 10 %, < 1 % à 25 %).
* **RQ3** : 4 familles sur 6 détectées à ≥ 95 % sans jamais les avoir vues ; Ryuk (discret) reste difficile.
* Apports : features de **réécriture en place / forte entropie** (gain majeur vs les 23 features du papier),
  **fusion multimodale** stockage/mémoire, **augmentation par exécutions mixtes synthétiques** (mix 0,39 → 0,97).

---

## 1. Données

* **RanSMAP** (Hirano & Kobayashi, 2024), commit tracé dans l'index. 1 970 exécutions :
  6 ransomwares (Conti, Darkside, LockBit, REvil, Ryuk, WannaCry), 6 bénins
  (AESCrypt, Firefox, Idle, Office, SDelete, Zip), 7 variantes de Conti, 100
  exécutions simultanées ransomware + bénin, 17 configurations matérielles.
* Le dépôt fait **~133 Go décompressé** : `scripts/build_dataset.py` le traite **en flux**
  (sparse-checkout git d'une configuration matérielle à la fois → extraction → suppression).
  Les CSV bruts restent hors OneDrive (`C:\Users\nejim\datasets\RanSMAP`) ; seules les
  features (~100 Mo) sont dans `data/processed/`.
* Traçabilité des données : chaque exécution conserve le **SHA git de ses 6 CSV sources**,
  le SHA1 du fichier de features, et le commit RanSMAP (`data/processed/index_parts/*.csv`).
  Chaque run MLflow porte un `data_digest` (empreinte de la liste exacte des exécutions).

### Prétraitement et features (BF02, BF03)

Chaque exécution → matrice `[T secondes × 42 statistiques suffisantes]` (comptes, sommes,
sommes des carrés), d'où l'on dérive exactement moyennes/variances sur n'importe quelle fenêtre.
**32 features par seconde** :

| Groupe | Features |
|---|---|
| Stockage (12) | ops et débit lecture/écriture, variance des LBA, part d'accès séquentiels, entropie moyenne/écart-type des écritures, **part d'écritures à forte entropie (> 0,9)**, **part de réécritures en place** (écriture sur un secteur lu plus tôt — signature du chiffrement) |
| Mémoire (20) | pages 4 Ko / 2 Mo / MMIO lues, écrites, lues-écrites, exécutées ; variance des adresses physiques ; entropie et part forte entropie des pages écrites |

Le jeu « hirano23 » reproduit les 23 features du papier RanSMAP (5 stockage + 18 mémoire).
Nettoyage : captures incomplètes, modalités absentes et doublons exclus ; entropies bornées ;
log1p ; standardisation **ajustée sur l'entraînement uniquement** ; découpage **par exécution**.

## 2. Modèles comparés

Démarche progressive du cahier des charges : baseline → Deep Learning → modèle amélioré.

| Modèle | Type | Origine |
|---|---|---|
| RF / SVM / kNN sur fenêtres de 10 s, 23 features | classique | **Hirano & Kobayashi** (CSR 2022, RanSMAP 2024) — RF 10 arbres, profondeur 10 |
| Régression logistique, Random Forest, LightGBM, XGBoost sur résumé du préfixe | classique | baselines scikit-learn / boosting |
| LSTM, GRU | DL | Davidian et al. 2024 ; RansoGuard (Cen et al. 2024) |
| CNN 1D, CNN-LSTM | DL | Davidian et al. 2024 |
| BiLSTM | DL | variante bidirectionnelle courante |
| MHSA-LSTM | DL | Alqahtani et al. 2026 (LSTM + multi-head self-attention) |
| Transformer encoder | DL | Khan et al. 2025 (séquences comportementales) |
| **TCN causal** | DL | proposition : convolutions dilatées causales, streaming, très rapide |
| **Fusion multimodale** | DL | proposition : un encodeur par modalité (stockage / mémoire) + fusion par attention, poids interprétables |
| **GRU détecteur d'anomalies** | non supervisé | proposition : entraîné sur le **bénin seul** → `anomaly_probability`, candidat zero-day |
| **Stats-GRU hybride** | DL amélioré | proposition : GRU + statistiques robustes du préfixe calculées dans le réseau |
| **Ensemble multimodal + LightGBM** | amélioré | proposition : moyenne des probabilités d'un modèle séquentiel et d'un boosting |
| **… + mix synthétique** | amélioré | proposition : augmentation par **exécutions simultanées synthétiques** (somme des statistiques additives d'un ransomware et d'un bénin du même matériel) → robustesse quand un ransomware tourne en même temps qu'un logiciel légitime |
| GRU + dropout de features, GRU + Group DRO | DL | tentatives de régularisation pour le zero-day (résultats conservés) |
| MLP sur fenêtres | DL | référence « sans ordre temporel » |
| Ablations GRU : stockage seul, mémoire seule, 23 features Hirano | DL | apport de chaque modalité / des features ajoutées |

Entraînement DL : augmentation par **préfixes aléatoires** (chaque séquence est tronquée
à une fraction U(0,02 ; 1) à chaque époque) pour apprendre à décider tôt ; early stopping
sur la perte de validation moyennée sur 10/25/50/100 %.

## 3. Protocoles d'évaluation

| Protocole | Question | Découpage |
|---|---|---|
| **E1** détection précoce | RQ1, RQ2 | 60/20/20 par exécution, 5 graines ; tests externes : variantes Conti, mix, matériel inédit |
| **E2** famille inconnue | RQ3 | leave-one-family-out (Conti exclu avec ses 7 variantes) |
| **E3** matériel inconnu | robustesse | entraînement sur un CPU, test sur l'autre + i5/i7/DDR5 |
| **E4** bénin inconnu | fiabilité | une application légitime retirée → faux positifs sur elle |

Métriques (section 9.3) : précision, rappel, F1, accuracy, **taux de faux positifs**, AUC,
PR-AUC, MCC, à chaque fraction ; **précocité** (plus petite fraction où F1 ≥ 0,95) ;
**fonctionnement en ligne** (score recalculé toutes les 5 s : délai avant première alerte,
taux de fausses alertes) au seuil 0,5 et à un **seuil opérationnel** calibré sur la validation
(≤ 5 % de fausses alertes en ligne) ; **calibration** (Brier, ECE) ; **efficacité**
(paramètres, taille, temps d'entraînement, latence CPU mono-thread).

## 4. Traçabilité MLflow

```bash
C:\Users\nejim\venvs\ransomware-dl\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
puis ouvrir http://127.0.0.1:5000 (ou utiliser le serveur déjà lancé par `run_all.ps1`).

* **Expériences** : `RanSMAP_Data` (version du dataset), `RanSMAP_E1_early_detection`,
  `RanSMAP_E2_unknown_family`, `RanSMAP_E3_unknown_hardware`, `RanSMAP_E4_unknown_benign_app`,
  `RanSMAP_Summary` (rapport + figures + tableaux).
* **Runs parents** (un par modèle, tag `role=parent`) : métriques agrégées `<métrique>_mean` /
  `<métrique>_std` sur les graines / plis → comparer les modèles : filtrer
  `tags.role = "parent"`, sélectionner, **Compare**.
* **Runs enfants** (un par graine/pli) : paramètres, tags `data_digest` et `code_digest`,
  datasets d'entraînement/test (onglet *Datasets*), courbes `test_f1_curve`, `test_fpr_curve`…
  (axe = % de séquence observée), courbes d'apprentissage `train_loss`/`val_loss`, et artefacts :
  `predictions.csv` (score de **chaque exécution à chaque fraction**), `early_detection.png`,
  `confusion_matrices.png`, `detection_rate_by_app.csv`, `preprocessing.json`, `training_history.csv`, modèle.
* Noms de métriques : `test_f1_p25` = F1 à 25 % ; `test_op_*` = au seuil opérationnel ;
  `variants_*`, `mix_*`, `extra_*` = tests externes ; `bench_latency_ms` = benchmark.
* **Model Registry** : `ransomware-detector`, alias `champion` (meilleur modèle E1).

## 5. Reproduire

```powershell
.\scripts\run_all.ps1
```
Étapes : `build_dataset.py` → `data_report.py` → `run_experiments.py --protocol e1|e2|e3|e4`
→ `benchmark_latency.py` → `make_report.py`. Environnement : `requirements.txt`
(PyTorch CUDA 12.6, entraînement DL sur GPU).

Utiliser le détecteur (interface Hermes / JEV) :
```python
from ransomdl.inference import RansomwareDetector, analyze_run_dir
det = RansomwareDetector.from_registry()                # modèle champion + seuil opérationnel
print(analyze_run_dir(r"...\<exécution>", det, observed_seconds=30, log_file="decisions.jsonl"))
```

## 6. Structure

```
src/ransomdl/   config, features (CSV → stats/s), data (features, nettoyage, protocoles),
                models (zoo PyTorch), train_dl, baselines (sklearn/boosting), anomaly,
                metrics, experiment (run + MLflow), inference (sortie JSON)
scripts/        build_dataset, data_report, run_experiments, benchmark_latency, make_report, run_all.ps1
data/processed/ features par exécution (.npz) + index avec SHA des sources
reports/        REPORT.md, DATA_QUALITY.md, results/*.csv, tables/, figures/
mlflow.db, mlruns/   suivi MLflow
```
