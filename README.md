# Détection précoce de ransomware par Deep Learning

Projet semestriel TEK-UP. Le but : détecter un ransomware **pendant qu'il s'exécute**, avant qu'il
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

## Limites

- Seul le cœur deep learning (objectifs 1 à 6 du cahier des charges) est réalisé ;
  l'extension Hermes + JEV ne l'est pas encore. Le module `inference.py` produit déjà la sortie
  JSON qu'elle utilisera.
- Pas de recherche d'hyperparamètres ; une seule graine pour E2, E3 et E4.

Documentation détaillée : [reports/DETAILS.md](reports/DETAILS.md).
