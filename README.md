🚕 Prédiction de la durée des courses de taxi à NYC

    De la régression linéaire aux modèles complexes > Rapport de synthèse réalisé par Frédéric DELCROIX & Xuan PENG

1. Introduction et jeu de données
1.1 Objectif et méthodologie

Ce projet vise à prédire la durée des trajets de taxi à New York en exploitant des données temporelles, géographiques et opérationnelles.

    Type de problème : Régression numérique continue.

    Métrique de performance : RMSE (Root Mean Square Error), exprimée en minutes.

Notre approche en 4 étapes :

    Nettoyage et préparation des données.

    Analyse exploratoire (EDA) pour identifier les patterns.

    Feature Engineering pour enrichir le signal.

    Benchmark de modèles : de la régression linéaire à XGBoost.

1.2 Description des variables

L'étude porte sur les données officielles des taxis jaunes de NYC (2016) : 1 013 587 trajets pour l'entraînement.
Variable	Description
vendor_id	Identifiant fournisseur (1 ou 2)
pickup_datetime	Date et heure de début de course
passenger_count	Nombre de passagers (1 à 6)
pickup/dropoff	Coordonnées GPS (Longitude/Latitude)
trip_duration	Cible à prédire (en secondes)
1.3 Nettoyage et Statistiques

    Qualité des données : 0 doublon, 0 valeur manquante après suppression de 13 lignes invalides.

    Distribution de la durée (en min) :

        Médiane : 11,1 | Moyenne : 14,0 | Max : 235,0 (filtré ensuite à 60 min).

2. Analyse exploratoire des données (EDA)
2.1 Distribution et Transformation

La durée des trajets est asymétrique. Une transformation logarithmique a été appliquée pour normaliser la distribution et optimiser les performances des algorithmes statistiques.
2.2 Facteurs déterminants

    Saisonalité : Juin est le mois le plus chargé (+27% vs Janvier).

    Rythme hebdomadaire : Les jeudis à 18h sont les plus congestionnés (+60% de durée vs nuit).

    Géographie : Manhattan présente une vitesse moyenne de 13-15 km/h contre 20 km/h en périphérie.

3. Feature Engineering

Le gain de précision repose sur la création de variables métier :
Famille	Variables créées	Justification
Distances	Haversine, Manhattan	Capture la géométrie "en grille" de NYC.
Temporel	Sin/Cos de l'heure	Préserve la continuité (23h est proche de 0h).
Indicateurs	Rush hour, Weekend	Capture les régimes de trafic binaires.
Interactions	Distance × Rush_hour	La distance pèse plus lourd en plein bouchon.
4. Modélisation et Résultats
4.1 Benchmarks Linéaires

Les modèles linéaires servent de base de comparaison.
Modèle	RMSE (sec)	R²	Note
Moyenne constante	660	—	Baseline de référence
Régression Linéaire	388	0,69	Modèle complet avec interactions
Elastic Net	387	0,66	Meilleur compromis linéaire
4.2 Approche hybride : Clustering + Elastic Net

Pour capturer l'hétérogénéité spatiale, nous avons segmenté NYC en clusters (K-Means) avec un modèle par zone.

    Résultat : Le passage à k=100 clusters permet de réduire l'erreur de 10% par rapport au modèle global.

4.3 Le champion : XGBoost

XGBoost surpasse tous les modèles grâce à sa gestion des relations non-linéaires.

    RMSE : 305s (≈ 5,1 min)

    R² : 0,79

    Prédicteur n°1 : Distance log-transformée (63% du poids).

5. Synthèse comparative
Modèle	RMSE Test (min)	R² Test	Performance
Elastic Net	6,42	0,65	Standard
Clustering + EN (k=100)	5,78	0,72	Robuste & Local
XGBoost	5,09	0,78	Précision maximale
6. Conclusion et Perspectives
🏆 Choix du modèle

    Pour la précision (App VTC) : XGBoost est le choix indiscutable.

    Pour l'aide à la décision urbaine : L'approche Clustering + Elastic Net offre le meilleur rapport performance/interprétabilité.

🚀 Pistes d'amélioration

    Données Exogènes : Intégrer la météo et les incidents de trafic en temps réel.

    Spatialisation : Utiliser le système de grille H3 (Uber) pour une segmentation géographique plus fine que le K-Means.

    Deep Learning : Tester des réseaux de neurones récurrents (LSTM) pour les séries temporelles.
