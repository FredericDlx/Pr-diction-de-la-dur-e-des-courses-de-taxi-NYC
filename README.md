
---

# Prédiction de la durée des courses de taxi à NYC
### *De la régression linéaire aux modèles complexes*
**— Rapport de synthèse —**
*Frédéric DELCROIX · Xuan PENG*
---

Le rapport complet est dispobnible https://fredericdlx.github.io/Prediction-de-la-dur-e-des-courses-de-taxi-NYC/PredictionTaxisNY.html

## Table des matières
1. [Introduction et jeu de données](#1-introduction-et-jeu-de-données)
2. [Analyse exploratoire des données](#2-analyse-exploratoire-des-données)
3. [Feature Engineering](#3-feature-engineering)
4. [Préparation pour la modélisation](#4-préparation-pour-la-modélisation)
5. [Modélisation (Machine Learning)](#5-modélisation-machine-learning)
6. [Conclusion](#6-conclusion)

---

## 1. Introduction et jeu de données

### 1.1 Objectif et méthodologie
Ce projet vise à prédire la durée des trajets de taxi à New York en exploitant des données temporelles (heure, jour, mois), géographiques (coordonnées GPS) et opérationnelles (nombre de passagers, fournisseur). Il s'agit d'un problème de régression : on cherche à prédire une valeur numérique continue (la durée) plutôt qu'une catégorie. La performance est mesurée par la **RMSE** (Root Mean Square Error), qui représente l'erreur moyenne de prédiction en minutes.

Notre approche suit quatre étapes classiques en data science :
1. Nettoyage et préparation des données.
2. Analyse exploratoire pour comprendre les patterns.
3. Feature Engineering (création de nouvelles variables pertinentes).
4. Test de plusieurs modèles : des régressions linéaires simples aux algorithmes avancés comme XGBoost.

### 1.2 Description des variables
L'étude porte sur les données officielles des taxis jaunes de NYC en 2016 : **1 013 587 trajets** pour l'entraînement et **434 330** pour le test final.

| Variable | Description |
| :--- | :--- |
| **vendor_id** | Identifiant du fournisseur (1 ou 2), répartition équilibrée 47%/53% |
| **pickup_datetime** | Date et heure de début de course |
| **passenger_count** | Nombre de passagers (1 à 6, la plupart des courses ont 1 passager) |
| **pickup/dropoff_longitude/latitude** | Coordonnées GPS du point de départ et d'arrivée |
| **store_and_fwd_flag** | Indicateur d’envoi différé de la course |
| **trip_duration** | **Variable à prédire** : durée du trajet en secondes |

### 1.3 Nettoyage des données
Le nettoyage inclut la conversion des coordonnées GPS, l'extraction de composantes temporelles et la suppression de 13 lignes avec des dates invalides. Après nettoyage : **0 doublon, 0 valeur manquante, 1 013 574 lignes exploitables**.

**Statistiques descriptives (en minutes) :**
`Médiane : 11,1 min` | `Moyenne : 14,0 min` | `Q1 : 6,7 min` | `Q3 : 17,9 min` | `Max : 235 min`

---

## 2. Analyse exploratoire des données

### 2.1 Distribution de la durée des trajets
La durée des trajets présente une distribution asymétrique : la plupart des courses durent entre 5 et 20 minutes, mais quelques valeurs extrêmes (jusqu'à 4 heures) tirent la moyenne vers le haut.

### 2.2 Effets temporels (heure, jour, mois)
* **Heure de la journée :** Les trajets sont plus longs pendant les heures de pointe (8h et 17h-18h) en raison du trafic, et plus courts la nuit.
* **Jour de la semaine :** Les durées moyennes augmentent progressivement du lundi au vendredi, puis chutent le week-end malgré un volume de courses élevé le samedi soir.

### 2.3 Variables opérationnelles
* **Nombre de passagers :** N'influence pas significativement la durée du trajet.
* **Vendor ID :** Le fournisseur 2 semble avoir des trajets légèrement plus longs en moyenne, probablement dû à une répartition géographique différente.

### 2.4 Effets spatiaux et géographiques
L'analyse montre une forte concentration à Manhattan. Les trajets traversant les ponts ou se dirigeant vers les aéroports (JFK, LaGuardia) présentent des profils de durée distincts.

---

## 3. Feature Engineering

### 3.1 Variables construites
Pour améliorer les modèles, nous avons créé :
* **Distance de Manhattan (L1) et Haversine :** Calcul de la distance "à vol d'oiseau" et selon une grille urbaine.
* **Vitesse moyenne estimée :** Pour identifier les zones de congestion.
* **Direction du trajet :** Angle entre le départ et l'arrivée.
* **Clustering (K-Means) :** Regroupement des coordonnées GPS en 100 zones pour capturer les spécificités locales des quartiers de NYC.

### 3.2 Analyse des corrélations
La corrélation la plus forte avec la durée du trajet est, sans surprise, la **distance parcourue**. Les variables temporelles (heure) montrent également un impact significatif une fois transformées.

---

## 4. Préparation pour la modélisation
Les données ont été divisées en :
* **Train (80%)** : Pour l'entraînement.
* **Validation (20%)** : Pour l'ajustement des hyperparamètres.
* **Test** : Jeu de données final séparé.

---

## 5. Modélisation (Machine Learning)

### 5.1 Baselines et régression linéaire
Le modèle de base (LM stepwise) sert de point de référence. Il capture les tendances globales mais peine sur les relations non-linéaires.

### 5.3 Approche hybride : Clustering + Elastic Net
En intégrant les clusters géographiques (zones de départ/arrivée), les performances s'améliorent nettement, montrant que la localisation est un prédicteur crucial.

### 5.5 Comparaison finale des modèles

| Métrique | LM (stepwise) | Lasso | Elastic Net | EN + Kmeans | **XGBoost** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RMSE_val_sec** | 387.806 | 387.343 | 387.253 | 341.909 | **305.003** |
| **R2_val** | 0.659 | 0.659 | 0.660 | 0.731 | **0.789** |
| **RMSE_test_sec** | 385.906 | 385.461 | 385.389 | 346.836 | **305.128** |
| **R2_test** | 0.658 | 0.659 | 0.659 | 0.724 | **0.786** |
| **RMSE_val_min** | 6.463 | 6.456 | 6.454 | 5.698 | **5.083** |
| **RMSE_test_min** | 6.432 | 6.424 | 6.423 | 5.781 | **5.085** |

---

## 6. Conclusion

### 6.1 Choix du modèle selon le contexte
* **XGBoost** est le plus précis (erreur de ~5 min), idéal pour une application de réservation.
* **Elastic Net** est préférable pour l'interprétabilité (comprendre quels facteurs influencent le prix/temps).

### 6.2 Limites et perspectives
L'absence de données météo et l'état précis du trafic en temps réel limitent la précision. L'ajout de données sur le métro ou les événements spéciaux pourrait affiner les prédictions.

### 6.3 Conclusion générale
Le passage d'une régression simple à un modèle de gradient boosting (**XGBoost**) a permis de réduire l'erreur de prédiction de près de **21%**. Le *feature engineering* (notamment la distance et les clusters) s'est avéré plus impactant que le choix de l'algorithme lui-même.
