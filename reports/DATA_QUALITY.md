# Qualité des données et nettoyage (RanSMAP)

Version des données (digest) : `984716504cc8cb42` — commit RanSMAP `eb00bd644b`

## Volumétrie

- Exécutions extraites : **1970** ; conservées : **1968** ; exclues : **2**
- Événements bruts traités : **3,781,920,580**
- Durée médiane d'une exécution : **194 s** (séquence de 195 pas d'1 s)

|                                            |   bénin |   ransomware |
|:-------------------------------------------|--------:|-------------:|
| ('extra', 'i3-gen12/ddr5-4800-16g')        |      60 |           60 |
| ('extra', 'i5-gen12/ddr4-2133-16g')        |      60 |           60 |
| ('extra', 'i7-gen12/ddr4-2133-16g')        |      60 |           60 |
| ('mix', 'i3-gen12/ddr4-2133-16g')          |       0 |          100 |
| ('original', 'celeron-gen6/ddr4-1600-16g') |      60 |           59 |
| ('original', 'celeron-gen6/ddr4-1600-8g')  |      60 |           60 |
| ('original', 'celeron-gen6/ddr4-1866-16g') |      60 |           59 |
| ('original', 'celeron-gen6/ddr4-1866-8g')  |      60 |           60 |
| ('original', 'celeron-gen6/ddr4-2133-16g') |      60 |           60 |
| ('original', 'celeron-gen6/ddr4-2133-8g')  |      60 |           60 |
| ('original', 'i3-gen12/ddr4-2133-16g')     |      60 |           60 |
| ('original', 'i3-gen12/ddr4-2133-8g')      |      60 |           60 |
| ('original', 'i3-gen12/ddr4-2666-16g')     |      60 |           60 |
| ('original', 'i3-gen12/ddr4-2666-8g')      |      60 |           60 |
| ('original', 'i3-gen12/ddr4-3200-16g')     |      60 |           60 |
| ('original', 'i3-gen12/ddr4-3200-8g')      |      60 |           60 |
| ('variants', 'i3-gen12/ddr4-2133-16g')     |       0 |           70 |

## Exécutions par application (conservées)

| app                  |   extra |   mix |   original |   variants |
|:---------------------|--------:|------:|-----------:|-----------:|
| AESCrypt             |      30 |     0 |        120 |          0 |
| AESCrypt_Conti       |       0 |    10 |          0 |          0 |
| AESCrypt_REvil       |       0 |    10 |          0 |          0 |
| Conti                |      30 |     0 |        120 |          0 |
| Conti_01             |       0 |     0 |          0 |         10 |
| Conti_02             |       0 |     0 |          0 |         10 |
| Conti_03             |       0 |     0 |          0 |         10 |
| Conti_04             |       0 |     0 |          0 |         10 |
| Conti_05             |       0 |     0 |          0 |         10 |
| Conti_06             |       0 |     0 |          0 |         10 |
| Conti_07             |       0 |     0 |          0 |         10 |
| Darkside             |      30 |     0 |        120 |          0 |
| Firefox              |      30 |     0 |        120 |          0 |
| Firefox_Conti        |       0 |    10 |          0 |          0 |
| Firefox_Office_REvil |       0 |    10 |          0 |          0 |
| Firefox_REvil        |       0 |    10 |          0 |          0 |
| Idle                 |      30 |     0 |        120 |          0 |
| LockBit              |      30 |     0 |        120 |          0 |
| Office               |      30 |     0 |        120 |          0 |
| Office_REvil         |       0 |    10 |          0 |          0 |
| REvil                |      30 |     0 |        120 |          0 |
| Ryuk                 |      30 |     0 |        120 |          0 |
| SDelete              |      30 |     0 |        120 |          0 |
| SDelete_Conti        |       0 |    10 |          0 |          0 |
| SDelete_REvil        |       0 |    10 |          0 |          0 |
| WannaCry             |      30 |     0 |        118 |          0 |
| Zip                  |      30 |     0 |        120 |          0 |
| Zip_Conti            |       0 |    10 |          0 |          0 |
| Zip_REvil            |       0 |    10 |          0 |          0 |

## Nettoyage appliqué

1. **Capture incomplète** : durée < 50 % de la médiane du sous-ensemble -> exclue.
2. **Modalité absente** : aucune trace disque ou aucune trace mémoire -> exclue.
3. **Doublons** : même empreinte de features -> exclue.
4. **Valeurs** : entropies bornées à [0, 1] (valeurs sentinelles -1 ignorées), entropie mémoire calculée uniquement sur les pages 4 Ko non-MMIO, NaN/inf remplacés par 0.
5. **Échelle** : comptes, débits et variances passés en log1p, puis standardisation ajustée **uniquement sur l'entraînement**, écrêtage à ±10 écarts-types.
6. **Découpage par exécution** (jamais par fenêtre) : une exécution n'apparaît que dans un seul des ensembles entraînement / validation / test.

### Exécutions exclues

| run_uid                                                           | app      |   duration_s |   n_bins | exclude_reason     |
|:------------------------------------------------------------------|:---------|-------------:|---------:|:-------------------|
| original__celeron-gen6__ddr4-1600-16g__WannaCry-20230516_11-39-36 | WannaCry |      7.90796 |        8 | capture_incomplete |
| original__celeron-gen6__ddr4-1866-16g__WannaCry-20221214_14-28-57 | WannaCry |     41.7619  |       42 | capture_incomplete |

## Profil comportemental moyen (sous-ensemble original)

`aw_ovw_frac` = part des écritures qui réécrivent un secteur lu plus tôt (chiffrement en place) ; `aw_hi_frac` = part d'écritures à entropie > 0,9 ; débits en log1p(Mo/s).

| app      | classe     |   aw_tput |   ar_tput |   aw_ent_mean |   aw_hi_frac |   aw_ovw_frac |   aw_seq_frac |   mw_t2 |   mw_ent_mean |   mr_t4 |   me_t2 |
|:---------|:-----------|----------:|----------:|--------------:|-------------:|--------------:|--------------:|--------:|--------------:|--------:|--------:|
| AESCrypt | bénin      |     2.174 |     2.423 |         0.783 |        0.677 |         0.019 |         0.874 |   2.866 |         0.73  |   6.956 |   0.124 |
| Firefox  | bénin      |     0.687 |     0.323 |         0.339 |        0.164 |         0.064 |         0.706 |   0.945 |         0.141 |   5.006 |   0.083 |
| Idle     | bénin      |     0.152 |     0.323 |         0.223 |        0.042 |         0.049 |         0.307 |   0.278 |         0.105 |   3.751 |   0.095 |
| Office   | bénin      |     0.134 |     0.217 |         0.352 |        0.091 |         0.047 |         0.484 |   0.274 |         0.075 |   3.868 |   0.079 |
| SDelete  | bénin      |     3.947 |     1.017 |         0.343 |        0.269 |         0.049 |         0.943 |   0.974 |         0.141 |   8.047 |   0.063 |
| Zip      | bénin      |     1.712 |     2.472 |         0.751 |        0.685 |         0.027 |         0.813 |   2.894 |         0.748 |   6.809 |   0.092 |
| Conti    | ransomware |     2.214 |     2.266 |         0.642 |        0.45  |         0.51  |         0.82  |   2.124 |         0.551 |   7.477 |   0.087 |
| Darkside | ransomware |     1.885 |     1.254 |         0.546 |        0.233 |         0.378 |         0.715 |   1.414 |         0.482 |   7.652 |   0.081 |
| LockBit  | ransomware |     1.977 |     2.004 |         0.633 |        0.441 |         0.54  |         0.716 |   2.176 |         0.58  |   7.529 |   0.085 |
| REvil    | ransomware |     0.909 |     2.019 |         0.43  |        0.206 |         0.249 |         0.619 |   1.57  |         0.42  |   7.507 |   0.105 |
| Ryuk     | ransomware |     0.567 |     0.862 |         0.351 |        0.145 |         0.195 |         0.53  |   1.726 |         0.246 |   5     |   0.067 |
| WannaCry | ransomware |     2.665 |     2.294 |         0.818 |        0.714 |         0.538 |         0.904 |   2.609 |         0.722 |   7.192 |   0.107 |

## Contrôle comportemental (signalé, non exclu)

Exécutions étiquetées ransomware dont moins de 5 % des écritures sont à forte entropie (aucun chiffrement visible). Elles sont **conservées** (pas d'exclusion a posteriori) mais expliquent un plafond de rappel : aucun détecteur comportemental ne peut les repérer.

|                          |   n |   aw_hi_frac_moyen |
|:-------------------------|----:|-------------------:|
| ('original', 'Ryuk')     |   1 |              0.049 |
| ('variants', 'Conti_06') |  10 |              0.008 |
