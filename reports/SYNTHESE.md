## Synthèse

**Données.** 1 970 exécutions RanSMAP (3,78 milliards d'événements disque/mémoire), 1 968 conservées
après nettoyage, transformées en séquences d'une mesure par seconde (≈ 194 s par exécution, 32 features).
26 modèles × 4 protocoles = **506 runs MLflow**, chacun avec ses prédictions par exécution.

### Réponses aux questions de recherche

* **RQ1 — détection sur séquence partielle : oui, dès 10 % (≈ 20 s).** Le meilleur modèle atteint
  F1 = 0,972 ± 0,007 à 10 % (rappel 0,976, FPR 3,3 %) et 0,996 à 25 %. Les modèles Deep Learning
  séquentiels atteignent F1 ≈ 0,95 à 10 % et ≈ 0,995 à 25 %. En fonctionnement continu (score recalculé
  toutes les 5 s, seuil calibré sur la validation), le délai médian avant alerte est de **10 à 17 s**,
  avec ≥ 99,7 % des ransomwares détectés pendant leur exécution.
* **RQ2 — compromis précocité / qualité.** Le gain est concentré entre 10 % et 25 % : au-delà de 25 %, tous
  les modèles sont ≥ 0,99 de F1. Le coût de la précocité est surtout en **faux positifs** : 3 à 6 % à 10 %,
  < 1 % dès 25 %. La référence du papier RanSMAP (RF, 23 features, fenêtres 10 s) monte à 19 % de faux
  positifs à 10 % et n'atteint F1 ≥ 0,95 qu'à 25 %. En ligne (une alerte à n'importe quel instant compte),
  le seuil opérationnel calibré sur la validation limite les fausses alertes à 4–9 % des exécutions bénignes
  sans perte de rappel.
* **RQ3 — famille inconnue : oui pour 4 familles sur 6, partiellement pour Ryuk et WannaCry.** En
  leave-one-family-out, Conti (et ses variantes), Darkside, LockBit et REvil inconnus restent détectés à
  ≥ 95 % sur la séquence complète par les meilleurs modèles. Pour REvil inconnu, les modèles séquentiels
  sont meilleurs tôt (100 % à 25 % contre ≈ 60 % pour le boosting). **Ryuk** (activité disque faible et lente)
  est la famille la plus difficile : à 25 %, au mieux 0,73 (GRU stockage seul), 0,2–0,4 pour la plupart des
  modèles ; sur la séquence complète 0,57–0,96 selon le modèle. **WannaCry** inconnu est détecté tôt par le
  boosting (100 % à 25 %) et en partie par la fusion multimodale (53 %), pas par les RNN classiques (≈ 0 %
  à 25 %, mais ≈ 100 % à la fin). Meilleur rappel moyen à 25 % : ensemble multimodal + LightGBM (0,865) ;
  les modèles de la référence RanSMAP plafonnent à 0,54–0,60.

### Comparaison des modèles (voir classement global ci-dessous)

1. **Les features comptent plus que l'architecture.** À architecture identique (GRU), nos 32 features
   (dont la part de *réécritures en place* et d'*écritures à forte entropie*) battent les 23 features du papier
   RanSMAP sur tous les axes : F1 @10 % 0,941 vs 0,918, famille inconnue 0,66 vs 0,60, matériel inconnu
   0,95 vs 0,80, faux positifs sur logiciel inconnu 2 % vs 25 %. Les features **mémoire seules généralisent
   mal** (matériel inconnu F1 0,60) : les motifs d'accès mémoire dépendent du CPU/RAM ; les features stockage
   sont les plus robustes.
2. **Deep Learning vs classique.** En distribution (E1), tous les modèles à features étendues sont à moins
   de 0,02 de F1 les uns des autres. Les écarts apparaissent hors distribution : le boosting (LightGBM,
   XGBoost) généralise mieux aux familles inconnues que les RNN/Transformer, tandis que la **fusion
   multimodale** (un encodeur par modalité + attention) est le meilleur modèle Deep Learning.
3. **Exécution simultanée (mix) : faiblesse corrigée par l'augmentation synthétique.** Quand un ransomware
   tourne en même temps qu'un logiciel légitime « ressemblant » (AESCrypt, SDelete, Zip), les modèles ne
   détectaient que 39–59 % des cas à 25 %. L'**augmentation par exécutions mixtes synthétiques** (somme des
   statistiques additives de deux exécutions du même matériel) porte ce rappel à **0,97** (fusion
   multimodale) et **0,96** (ensemble), et à 1,00 sur la séquence complète, sans dégrader E1.
4. **Matériel inconnu (E3).** Entraînés sur un seul CPU, les meilleurs modèles gardent F1 ≈ 0,99 sur l'autre
   CPU et sur le matériel `extra` (i5/i7, DDR5) ; la référence RanSMAP descend à 0,80–0,85 (à 25 %).
5. **Logiciel légitime inconnu (E4).** Les modèles de référence signalent à tort 31–37 % des exécutions d'une
   application jamais vue ; l'ensemble multimodal + LightGBM et le CNN-LSTM 0 %, la moitié de nos modèles
   ≤ 5 %. Points de vigilance : LightGBM + mix synthétique (14 % en moyenne, dû à SDelete inconnu signalé à
   82 %) et la fusion multimodale (≈ 10 %, dû à **Idle** inconnu signalé à 45–58 % : une machine au repos
   jamais vue lui paraît anormale). L'ensemble les compense en grande partie (Idle inconnu 1 %, SDelete inconnu 22 %).
6. **Efficacité.** Tous les modèles infèrent une séquence complète en < 8 ms sur un cœur CPU (LightGBM
   0,5 ms, fusion multimodale 7 ms, ensemble 7,7 ms), hors SVM/kNN de la référence (18–33 ms) :
   largement compatible avec une décision par seconde. Entraînement DL sur RTX 3050 : 13 s (MLP) à 106 s
   (BiLSTM) par modèle, 200 s avec l'augmentation mix ; boosting < 10 s sur CPU.
7. **Non supervisé.** Le détecteur d'anomalies entraîné sur le bénin seul (AUC 0,79 sur la séquence complète)
   ne suffit pas seul : le chiffrement intensif ressemble à AESCrypt/Zip. Il est conservé comme signal
   complémentaire `anomaly_probability` pour Hermes/JEV.
8. **Tentatives sans gain, conservées pour traçabilité** : Stats-GRU hybride, dropout de features et Group DRO
   n'améliorent pas la détection des familles inconnues.

### Recommandation

* **Modèle Deep Learning recommandé (inscrit au Model Registry, alias `champion`) : fusion multimodale +
  mix synthétique** — 2ᵉ score global (0,925), meilleur modèle 100 % DL, causal (utilisable en streaming),
  poids d'attention stockage/mémoire interprétables pour l'état du cas transmis à Hermes.
* **Meilleur système global : ensemble (fusion multimodale + LightGBM) + mix synthétique** — score global 0,936,
  F1 0,972 à 10 %, rappel mix 0,96, F1 ≈ 0,99 sur matériel inconnu, 4 % de FPR sur logiciel inconnu,
  pour 7,7 ms d'inférence.
* Variante sans augmentation à privilégier si la priorité est le zero-day et l'absence de faux positifs
  (ensemble multimodal + LightGBM : meilleur rappel famille inconnue, 0 % FPR sur logiciel inconnu).

### Limites

* 6 familles seulement : RQ3 repose sur 6 plis ; Ryuk montre qu'une famille « discrète » peut échapper.
* **Conti_06** n'a aucune activité de chiffrement dans RanSMAP (profil identique à Idle : débit d'écriture
  0,12 vs 0,15, 1 % d'écritures à forte entropie contre 67–75 % pour les autres variantes) : échantillon
  manifestement inopérant, indétectable comportementalement ; il plafonne le rappel « variantes » à 0,857
  pour tous les modèles. Conservé tel quel (pas d'exclusion a posteriori).
* Les mélanges synthétiques supposent l'additivité des statistiques ; ils sont validés sur les 100 vraies
  exécutions mixtes de RanSMAP, jamais utilisées à l'entraînement.
* Résultats valables pour les comportements présents dans RanSMAP (cf. section 13 du cahier des charges).
