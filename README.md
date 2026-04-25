 Prédiction de la durée des courses de taxi à NYC
De la régression linéaire aux modèles complexes
— Rapport de synthèse —
Frédéric DELCROIX · Xuan PENG · 

1. Introduction et jeu de données
1.1 Objectif et méthodologie

Ce projet vise à prédire la durée des trajets de taxi à New York en exploitant des données temporelles (heure, jour, mois), géographiques (coordonnées GPS) et opérationnelles (nombre de passagers, fournisseur). Il s'agit d'un problème de régression : on cherche à prédire une valeur numérique continue (la durée) plutôt qu'une catégorie. La performance est mesurée par la RMSE (Root Mean Square Error), qui représente l'erreur moyenne de prédiction en minutes.

Notre approche suit quatre étapes classiques en data science : (1) nettoyage et préparation des données, (2) analyse exploratoire pour comprendre les patterns, (3) feature engineering (création de nouvelles variables pertinentes), (4) test de plusieurs modèles de complexité croissante — des régressions linéaires simples aux algorithmes avancés comme XGBoost.
1.2 Description des variables

L'étude porte sur les données officielles des taxis jaunes de NYC en 2016 : 1 013 587 trajets pour l'entraînement et 434 330 pour le test final.
Variable	Description
vendor_id	Identifiant du fournisseur (1 ou 2), répartition équilibrée 47%/53%
pickup_datetime	Date et heure de début de course
passenger_count	Nombre de passagers (1 à 6, la plupart des courses ont 1 passager)
pickup/dropoff_longitude/latitude	Coordonnées GPS du point de départ et d'arrivée
store_and_fwd_flag	Indicateur d’envoi différé de la course
trip_duration	Variable à prédire : durée du trajet en secondes
1.3 Nettoyage des données

Le nettoyage inclut la conversion des coordonnées GPS (format européen vers format américain), l'extraction de composantes temporelles (heure, jour de la semaine, mois) et la suppression de 13 lignes avec des dates invalides. Après nettoyage : 0 doublon, 0 valeur manquante, 1 013 574 lignes exploitables.

Statistiques descriptives de la durée des trajets :

Médiane : 11,1 min Moyenne : 14,0 min Q1 : 6,7 min Q3 : 17,9 min Max : 235 min
2. Analyse exploratoire des données
2.1 Distribution de la durée des trajets

La durée des trajets présente une distribution asymétrique : la plupart des courses durent entre 5 et 20 minutes, mais quelques valeurs extrêmes (jusqu'à 4 heures) tirent la moyenne vers le haut. C'est typique des données de transport urbain.
Distribution
Figure 1 — Distribution de la durée des trajets : (gauche) filtrée < 60 min, (droite) transformation logarithmique

Après suppression des 2,3% de trajets aberrants (> 60 min), la distribution des durées suit une courbe asymétrique. Une transformation logarithmique permet de la "normaliser" — condition souvent requise pour que les algorithmes statistiques fonctionnent de manière optimale.
2.2 Effets temporels (heure, jour, mois)

Le temps est un facteur majeur : la durée varie fortement selon l'heure de la journée, le jour de la semaine et le mois.
Heure et jour Heure et mois
Figure 2 — Variation temporelle de la durée : (gauche) selon l'heure et le mois, (droite) selon l'heure et le jour de semaine

Effet saisonnier (gauche) : Juin domine avec 18,5 min aux heures de pointe contre 14,5 min en janvier (+27%), sous l'effet du tourisme estival, des travaux et d'une météo favorable aux déplacements.

Effet heure/jour (droite) : Les heures de pointe en semaine (17h-18h) atteignent 18,5 min le jeudi, soit +60% vs la nuit — là où le week-end reste stable à 12-14 min, la circulation y étant essentiellement récréative.
2.3 Variables opérationnelles

71% des trajets transportent un seul passager, la transmission est quasi-systématiquement en temps réel (99,5%), et les deux prestataires sont répartis à parts quasi-égales (47/53%). Ces trois variables apparaissent peu discriminantes : les graphiques de régression confirment que leurs modalités n'influencent pas la relation distance/durée.

Ce sont donc les dimensions spatiales et temporelles qui constituent les vrais déterminants de la durée, et qui guideront en priorité la construction des variables pour la modélisation.
2.4 Effets spatiaux et géographiques

L'activité des taxis se concentre massivement à Manhattan, avec des flux secondaires vers Brooklyn, Queens et les aéroports JFK/LaGuardia.
Carte de vitesse
Figure 3 — Vitesse moyenne (km/h) par zone de prise en charge à New York

Lecture de la carte : Les couleurs indiquent la vitesse moyenne selon le point de départ. Le rouge/orange du centre de Manhattan reflète la congestion chronique (13-15 km/h), contre 17-20 km/h en périphérie (Brooklyn, Queens) où la circulation est plus fluide. Ce contraste justifie l'importance des variables géographiques : un même trajet peut durer très différemment selon sa localisation.
3. Feature Engineering
3.1 Variables construites

Le feature engineering consiste à créer de nouvelles variables à partir des données sources et à enrichir la représentation du problème. Cette étape s'avère tout aussi décisive pour la précision finale que le simple choix de l'algorithme.
Famille	Variables créées	Justification
Distances	Haversine, Manhattan, log-transformées, direction	Haversine mesure la distance à vol d'oiseau, Manhattan capture la géométrie en grille de NYC
Temporel cyclique	sin/cos de l'heure, jour, année	Encodage qui préserve la continuité : 23h est proche de 0h
Indicateurs	Heures de pointe, week-end, nuit	Variables binaires capturant les régimes de trafic
Géographiques	Trajets aéroport (JFK, LaGuardia), midtown	Certaines zones ont des comportements de trafic spécifiques
Interactions	log_haversine × is_rush_hour
log_haversine × pickup_hour_sin/cos
log_haversine × is_weekend	L'impact de la distance varie selon le contexte temporel. Parcourir 5 km en taxi ne prend pas le même temps à 8h du matin qu'à 3h du matin.
Explication : Distance Manhattan vs Haversine

La distance Haversine mesure la ligne droite entre deux points. La distance Manhattan suit les rues perpendiculaires — bien plus réaliste pour un taxi qui ne traverse pas les immeubles.
Encodage cyclique du temps

Sans encodage, un algorithme perçoit 23h et 0h comme éloignés (différence de 23). L'encodage sin/cos place les heures sur un cercle, rendant 23h et minuit naturellement voisins.
3.2 Analyse des corrélations

La distance est le principal déterminant de la durée (corrélations > 0,70), mais les différentes façons de la mesurer sont très redondantes entre elles (r > 0,90) — il suffira d'en retenir une ou deux.

Certaines variables géographiques affichent une corrélation négative (-0,10 à -0,20) : elles reflètent un effet de zone (trajets plus courts au centre, plus longs en périphérie) plutôt qu'une causalité directe.

Les variables temporelles semblent faibles isolément (0,05 à 0,15), mais elles peuvent interagir avec la distance — une même distance prend bien plus de temps en heure de pointe qu'en pleine nuit.

Un bon modèle devra donc combiner distance, variables temporelles et leurs interactions pour capturer pleinement la durée des trajets.
4. Préparation pour la modélisation

Avant d'entraîner les modèles, trois étapes techniques sont nécessaires :

1. Nettoyage : Suppression des 0,065% de trajets à distance quasi nulle (< 0,05 km) mais durée élevée (> 20 min) — probablement des erreurs GPS violant la relation physique distance/durée.

2. Standardisation : Les variables numériques sont ramenées à la même échelle (moyenne = 0, écart-type = 1) pour éviter qu'un algorithme surpondère les grandes valeurs.

3. Division : Les données sont réparties aléatoirement en 80% entraînement / 20% validation, afin d'évaluer les modèles sur des données qu'ils n'ont pas vues.
5. Modélisation
5.1 Baselines et régression linéaire

On commence par des modèles simples pour établir une référence de performance.
Modèle	RMSE (sec)	R²	Explication
Moyenne constante	≈ 660	—	Prédit toujours la moyenne des durées d'entraînement
Distance seule (log_haversine)	≈ 440	—	Utilise uniquement la distance logarithmique comme prédicteur
Régression linéaire complète	≈ 388	≈ 0,69	Modèle complet avec distances, temporel, contextuel et interactions

Le modèle linéaire complet atteint un RMSE d'environ 388 secondes (≈ 6,5 minutes) sur la validation, ce qui représente une amélioration significative par rapport à la distance seule. Le R² d'environ 0,69 signifie que le modèle explique près de 69% de la variation de la durée des trajets.
5.2 Régressions régularisées

Trois approches de régression régularisée ont été testées — Ridge, Lasso et Elastic Net — afin de stabiliser les coefficients en présence de redondance entre variables et d'améliorer la généralisation du modèle linéaire. Les résultats montrent des performances contrastées selon le niveau de pénalisation appliqué.
Modèle 	Lambda optimal 	RMSE (sec) 	RMSE (min) 	R² 	Variables retenues 	Interprétation
Ridge 	0.0569 	439.7 	7.33 	0.561 	18 (aucune annulée) 	Pénalisation trop forte → sous-apprentissage, perte d'information explicative
Lasso 	0.00044 	387.3 	6.46 	0.6595 	18 (aucune annulée) 	Faible pénalisation, stabilisation des coefficients sans dégradation de la précision
Elastic Net 	≈ 0.0007 (α = 0.50) 	387.25 	6.45 	0.6596 	18 (aucune annulée) 	Compromis Ridge/Lasso, performances quasi identiques au Lasso

Le Ridge souffre d'une sur-régularisation (RMSE = 439s, soit 7,33 min), tandis que le Lasso et l'Elastic Net atteignent ≈ 387s (6,45 min) avec un R² ≈ 0,66 — comparable au meilleur modèle linéaire, mais avec de meilleures estimations. Le fait qu'aucune variable ne soit annulée confirme que les 18 variables retenues sont toutes informatives.

Ces résultats fixent le plafond des modèles linéaires à RMSE ≈ 387s. Pour progresser, il faudra se tourner vers des modèles non linéaires ou des approches combinant segmentation et modélisation locale.
5.3 Approche hybride : Clustering + Elastic Net

Pour mieux capturer l'hétérogénéité spatiale du trafic new-yorkais, nous testons une approche hybride combinant segmentation non supervisée (K-means) et régression régularisée (Elastic Net).

Principe : Au lieu d'un modèle global unique, les trajets sont partitionnés en k zones homogènes selon leurs caractéristiques spatio-temporelles (coordonnées, heure, jour). Un modèle Elastic Net spécialisé est entraîné dans chaque cluster. Validation croisée imbriquée (5 folds externes × 3 folds internes) pour une évaluation robuste sans fuite de données.
Configuration 	RMSE Test (min) 	Gain vs k=1 	Temps calcul 	Commentaire
k = 1 (Elastic Net global)	6,42	baseline	1,0 min	Modèle unique pour toute la ville
k = 10	6,05	−6%	3,5 min	Segmentation grossière
k = 100	5,78	−10%	4,9 min	Bon compromis : gain significatif, coût raisonnable
k = 500	5,59	−13%	13,5 min	Modélisation fine mais plus coûteuse
k = 1000	5,52	−14%	28,7 min	Meilleure performance ML (avant XGBoost)
Tableau — Performance de l'approche Clustering + Elastic Net

Résultats clés : Le clustering améliore significativement les performances (−14% d'erreur avec k=1000), confirmant que les régimes de trafic diffèrent fortement selon les zones (Manhattan dense vs périphérie). Le compromis optimal est k=100 : −10% d'erreur avec un temps de calcul raisonnable. L'écart validation-test reste faible (≈5 sec) pour toutes les configurations, garantissant une bonne généralisation. Chaque cluster conserve l'interprétabilité d'un modèle Elastic Net local.
5.4 XGBoost (comparaison avec un modèle non-linéaire)

XGBoost construit des centaines d'arbres de décision se spécialisant sur différents types de trajets, puis combine leurs prédictions en pondérant les plus fiables.

Après optimisation des paramètres, XGBoost atteint RMSE ≈ 305s (≈ 5,1 min) et R² ≈ 0,79, soit un gain d'environ 80 secondes par rapport à l'Elastic Net.

La distance log-transformée (log_haversine) domine avec 63% du pouvoir explicatif. Les variables géométriques jouent un rôle d'ajustement secondaire, les variables temporelles apportent le contexte, et les variables contextuelles (passagers, prestataire, aéroports) restent marginales. C'est sa capacité à capter les relations non linéaires qui explique le gain de performance observé.
5.5 Comparaison finale des modèles ML

Cinq modèles ont été comparés sur les ensembles de validation et de test, en s'appuyant sur deux métriques complémentaires : le RMSE (erreur quadratique moyenne, exprimée en secondes et en minutes) et le R² (part de variance expliquée). Les résultats mettent en évidence une progression claire des performances à mesure que la complexité des modèles augmente.
Modèle 	RMSE Validation (sec) 	RMSE Validation (min) 	R² Validation 	RMSE Test (sec) 	RMSE Test (min) 	R² Test
LM (stepwise final) 	387.8 	6.46 	0.659 	385.9 	6.43 	0.658
Lasso 	387.3 	6.46 	0.659 	385.5 	6.42 	0.659
Elastic Net 	387.3 	6.45 	0.660 	385.4 	6.42 	0.659
Elastic Net + Kmeans (k=100) 	341.9 	5.70 	0.731 	346.8 	5.78 	0.724
XGBoost 	305.0 	5.08 	0.789 	305.1 	5.09 	0.786

Les modèles linéaires (LM, Lasso, Elastic Net) affichent des performances quasi identiques (RMSE ≈ 387s, R² ≈ 0,66), avec des difficultés sur les trajets longs et les comportements non linéaires du trafic.

L'approche Elastic Net + K-means (k=100) réduit le RMSE à 342s (-45s) en segmentant l'espace urbain en micro-zones — confirmant que les dynamiques de trafic sont fortement locales.

XGBoost s'impose comme le plus performant (RMSE = 305s, R² = 0,789), stable en test (305s, R² = 0,786), grâce à sa capacité à capturer les interactions complexes entre variables spatiales, temporelles et de distance.

La progression de 387s à 305s illustre l'apport décisif de la modélisation non linéaire : Elastic Net + K-means offre le meilleur compromis interprétabilité/performance, XGBoost la meilleure précision.
6. Conclusion
6.1 Choix du modèle selon le contexte

Régression linéaire simple (RMSE ≈ 388s) : Interprétabilité maximale, performances proches des modèles régularisés. Point de départ idéal.

Lasso et Elastic Net (RMSE ≈ 387s, R² ≈ 0,66) : Meilleur choix linéaire, 18 variables conservées, idéal pour les contextes nécessitant de l'explicabilité.

Clustering + Elastic Net k=100 (RMSE ≈ 342s) : Meilleur compromis performance/interprétabilité : −10% d'erreur, 5 min de calcul, modèles locaux interprétables par zone.

XGBoost (RMSE ≈ 305s, R² ≈ 0,79) : Champion en précision (+82s vs linéaires). Optimal pour les applications où la précision prime (VTC, ETA).
6.2 Limites et perspectives

Principales limites : Les modèles ignorent la météo, les événements spéciaux et les accidents. La granularité horaire masque aussi les variations rapides au sein d'une même heure.

Améliorations possibles : L'intégration de variables exogènes (météo, trafic temps réel, OpenStreetMap) permettrait probablement de dépasser le plafond actuel. Le système de partitionnement spatial H3 développé et utilisé par Uber couplé à des données en temps réel sur la ciculation serait aussi un "game changer".
6.3 Conclusion générale

    Ce projet compare quatre familles de modèles sur plus d'un million de trajets NYC. Le feature engineering représente une part essentielle des gains : de RMSE ≈ 387s (Elastic Net) à 323s (Clustering + Elastic Net, −17%) jusqu'à 305s avec XGBoost (R² ≈ 0,79), grâce à sa capacité à capturer des interactions non linéaires complexes.

    L'approche Clustering + Elastic Net offre le meilleur compromis performance/interprétabilité — précieuse pour l'aide à la décision urbaine — tandis que XGBoost reste optimal pour les applications grand public. Des variables exogènes, une segmentation H3 ou des approches séquentielles (LSTM, GNN) constitueraient des pistes naturelles d'amélioration.

    Ce cadre méthodologique est généralisable à d'autres grandes métropoles, ouvrant la voie à des outils avancés d'optimisation de la mobilité urbaine.
