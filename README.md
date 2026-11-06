# Détection précoce de ransomware par Deep Learning

Projet semestriel TEK-UP (cahier des charges : « Détection précoce de ransomware par Deep
Learning comportemental et investigation adaptative avec Hermes et JEV »). Le but : détecter un ransomware **pendant qu'il s'exécute**, avant qu'il
ait fini de chiffrer, en observant uniquement son activité **disque** et **mémoire**.

Résultat principal : le meilleur modèle détecte un ransomware après **environ 20 secondes**
d'exécution, avec un F1 de **0,97**.

## Données

On utilise le dataset public [RanSMAP](https://github.com/manabu-hirano/RanSMAP) :
1 970 exécutions enregistrées sur des machines virtuelles (environ 3 minutes chacune).

- **1 070 ransomwares** de 6 familles : Conti, Darkside, LockBit, REvil, Ryuk, WannaCry.
- **900 logiciels légitimes** : AESCrypt, Firefox, Idle, Office, SDelete, Zip.
- Des tests supplémentaires : variantes de Conti, ransomware lancé en même temps qu'un logiciel
  légitime (« mix »), et matériel jamais vu (CPU i5/i7, RAM DDR5).

Le dataset fait environ 133 Go. Le script `scripts/build_dataset.py` le télécharge **une
configuration à la fois**, en extrait les features, puis supprime les fichiers bruts.
Il reste environ 57 Mo de features.

## Démarche

1. **Prétraitement** : chaque exécution devient une séquence d'une mesure par seconde, avec
   32 features (12 disque, 20 mémoire). On y a ajouté des indicateurs propres au chiffrement,
   comme la part de réécritures en place et la part d'écritures à forte entropie.
2. **Modèles** : 26 modèles comparés, des baselines du papier RanSMAP (RF, SVM, kNN) au
   machine learning classique (LightGBM, XGBoost) puis au deep learning (LSTM, GRU, CNN,
   Transformer, TCN…).
3. **Améliorations** :
   - **fusion multimodale** : un GRU pour le disque, un GRU pour la mémoire, combinés par une
     couche d'attention ;
   - **augmentation « mix » synthétique** : on additionne les statistiques d'un ransomware et
     d'un logiciel légitime pour simuler leur exécution simultanée ;
   - **ensemble** : moyenne des probabilités de la fusion multimodale et de LightGBM.
4. **Évaluation** : 4 protocoles, tous sur des données jamais vues à l'entraînement.
   - E1 : détection précoce, à 10, 25, 50, 75 et 100 % de l'exécution.
   - E2 : famille de ransomware inconnue.
   - E3 : matériel inconnu.
   - E4 : logiciel légitime inconnu (fausses alertes).

## Résultats

| Modèle | F1 à 10 % (≈ 20 s) | Mix à 25 % | Faux positifs sur logiciel inconnu | Score global |
|---|---|---|---|---|
| **Ensemble + mix** (meilleur global) | 0,972 | 0,96 | 4 % | 0,936 |
| **Fusion multimodale + mix** (meilleur deep learning) | 0,954 | 0,97 | 10 % | 0,925 |
| Méthode du papier RanSMAP (RF) | 0,887 | 0,87 | 37 % | 0,775 |

- Au-delà de 25 % de l'exécution, tous les modèles dépassent 0,99 de F1.
- 4 familles sur 6 sont détectées sans les avoir vues à l'entraînement. Ryuk reste difficile.
- L'augmentation « mix » fait passer la détection d'un ransomware masqué par un logiciel
  légitime de 0,39 à 0,97.

Tous les tableaux et figures sont dans [reports/REPORT.md](reports/REPORT.md).

## Structure du projet

```
src/ransomdl/       code : features, données, modèles, entraînement, métriques, inférence
scripts/            scripts à lancer (dataset, expériences, rapport)
data/processed/     index des exécutions (les fichiers .npz ne sont pas versionnés)
reports/            rapport, tableaux, figures, qualité des données
```

## Lancer le projet

```bash
pip install -r requirements.txt
```

```powershell
.\scripts\run_all.ps1
```

Ce script enchaîne : téléchargement et extraction des données, rapport qualité, expériences
E1 à E4, mesure de latence, puis génération du rapport. Les expériences sont suivies dans
**MLflow**. Les chemins locaux (dataset, environnement Python) sont définis dans
`src/ransomdl/config.py` et `scripts/run_all.ps1`.

## Avancement par rapport au cahier des charges

**En résumé : le cœur obligatoire (objectifs 1 à 6) est terminé. L'extension Hermes + JEV
(objectifs 7 à 10), la démonstration en VM et le rapport final restent à faire.**

### Objectifs (section 2.2)

| # | Objectif | Statut |
|---|---|---|
| 1 | Recevoir des traces comportementales temporelles | ✅ Fait |
| 2 | Les prétraiter | ✅ Fait |
| 3 | Détecter le ransomware avec un modèle de deep learning | ✅ Fait |
| 4 | Produire un score de risque | ✅ Fait (probabilité calibrée) |
| 5 | Tester à 10, 25, 50, 75 et 100 % de la séquence | ✅ Fait (E1) |
| 6 | Tester la généralisation à une famille jamais vue | ✅ Fait (E2) |
| 7 | Ajouter l'extension Hermes + JEV | ❌ À faire |
| 8 | Laisser JEV choisir le prochain outil d'investigation | ❌ À faire |
| 9 | Récupérer de nouvelles informations | ❌ À faire |
| 10 | Construire progressivement un diagnostic final | ❌ À faire |

### Besoins fonctionnels (section 10)

| ID | Besoin | Statut |
|---|---|---|
| BF01 à BF03 | Importer, prétraiter, construire les séquences | ✅ Fait |
| BF04 à BF05 | Entraîner le modèle, produire un score de risque | ✅ Fait |
| BF06 à BF07 | Tester les fractions et une famille jamais vue | ✅ Fait |
| BF08 | Construire l'état du cas | 🟡 Base prête : `inference.py` produit le JSON initial |
| BF09 à BF16 | Hermes, JEV, tools, boucle, arrêt, analyse finale, journal | ❌ À faire |

### Besoins non fonctionnels (section 11)

| ID | Critère | Statut |
|---|---|---|
| BNF01 | Sécurité : aucun ransomware réel exécuté | ✅ Respecté (traces RanSMAP uniquement) |
| BNF02 | Reproductibilité | ✅ `run_all.ps1`, versions figées, graines fixes |
| BNF03 | Modularité (DL, Hermes, JEV, tools séparés) | 🟡 Module DL séparé ; le reste à construire |
| BNF04 | Traçabilité | 🟡 MLflow pour le DL ; journal d'investigation à faire |
| BNF05 | Robustesse aux informations manquantes | 🟡 Exécutions incomplètes exclues ; à gérer dans la boucle |
| BNF06 | Extensibilité | ✅ Nouveau modèle = une ligne dans `experiment.py` |

### Planning (section 15)

| Phase | Statut |
|---|---|
| P1 Bibliographie + dataset | ✅ Fait |
| P2 Prétraitement + baseline | ✅ Fait |
| P3 Deep learning | ✅ Fait |
| P4 Early detection | ✅ Fait |
| P5 Zero-day | ✅ Fait |
| P6 Hermes | ❌ À faire |
| P7 Tools + JEV | ❌ À faire |
| P8 Évaluation (architectures A / B / C) | ❌ À faire |
| P9 Démonstration + rapport | ❌ À faire |

## Prochaines étapes

### 1. Extension Hermes + JEV (P6, P7 : BF08 à BF16)

1. **État du cas (BF08)** : définir le format JSON de l'état transmis à Hermes et JEV, à partir
   de la sortie de `inference.py` (probabilité, anomalie, indicateurs). Prévoir les champs
   « inconnu » (exemple : contexte mémoire non encore analysé).
2. **Tools d'investigation (BF11)** : coder les 4 outils de la section 6.2, chacun lisant les
   données réellement disponibles du cas (fichiers `.npz`) :
   - `StorageAnalysis` : débit, variance des adresses, réécritures en place ;
   - `MemoryAnalysis` : activité des pages, entropie des pages écrites ;
   - `TemporalAnalysis` : évolution du score et de l'activité dans le temps ;
   - `EntropyAnalysis` : part des écritures à forte entropie.
3. **Hermes (BF09, BF12, BF15)** : orchestrateur qui construit le contexte, exécute le tool
   choisi, ajoute le résultat au contexte et rédige l'analyse finale.
4. **JEV (BF10, BF14)** : module de décision qui reçoit l'état et renvoie le prochain tool
   ou `STOP`. Il n'exécute rien lui-même.
5. **Boucle et journal (BF13, BF16)** : répéter jusqu'à `STOP`, et enregistrer chaque étape
   (état, tool choisi, résultat) en JSONL.

### 2. Évaluation de l'extension (P8, section 9)

- Comparer les architectures **A** (DL seul), **B** (DL + Hermes) et **C** (DL + Hermes + JEV)
  sur les mêmes cas.
- Comparer l'investigation **fixe** (ordre de tools prédéfini) et **adaptative** (choix par JEV).
- Mesurer : pertinence du tool choisi, nombre de tools, appels inutiles, temps d'investigation,
  complétude des informations, qualité de l'analyse finale.

### 3. Démonstration (P9, section 8)

- VM isolée + **simulateur de comportement** (aucun ransomware réel) + monitoring.
- Convertir les événements collectés au format attendu par le pipeline, puis enchaîner
  détection → Hermes → JEV → diagnostic.

### 4. Compléments sur le cœur (si le temps le permet)

- Plusieurs graines pour E2, E3 et E4 (seul E1 en a 5).
- Recherche d'hyperparamètres sur les modèles finaux.
- Améliorer la détection précoce de **Ryuk** inconnu (0,2 à 0,4 à 25 %).
- Script de démonstration qui lit une exécution depuis son `.npz` (les CSV bruts sont supprimés).

### 5. Livrables finaux (section 13)

- Rapport final et présentation.
- Mettre à jour ce README à chaque étape.

## Limites

- Les conclusions portent sur les comportements présents dans RanSMAP.
- La variante Conti_06 ne chiffre rien dans le dataset : aucun modèle ne peut la détecter.
- Pas de recherche d'hyperparamètres ; une seule graine pour E2, E3 et E4.

Documentation détaillée : [reports/DETAILS.md](reports/DETAILS.md).
