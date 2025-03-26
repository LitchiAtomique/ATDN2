ATDN2 TP1 - Analyse et Modélisation des Facteurs Impactant le Rendement Agricole

Auteur

Remy XU - M1 OIVM

Date

26 mars 2025

Objectif du TP

L'objectif principal de ce TP est d'analyser les facteurs influençant le rendement agricole en utilisant des méthodes statistiques et des modèles de machine learning. L'étude vise à optimiser les pratiques agricoles pour maximiser la production.

Contenu du projet

Le projet est structuré en plusieurs étapes :

1. Compréhension du problème

Variables disponibles :

Rendement (variable cible)

Précipitations

Température

Type de sol

Engrais utilisé

Problématique : Comment ajuster les pratiques agricoles pour maximiser le rendement tout en minimisant les coûts et en s’adaptant aux conditions climatiques ?

2. Analyse statistique descriptive

Mesures de tendance centrale : moyenne, médiane, mode.

Mesures de dispersion : variance, écart-type, étendue.

Visualisation des données : histogrammes, boîtes à moustaches, heatmap de corrélation.

Interprétation des corrélations : identification des facteurs les plus impactants.

3. Analyse de la variance (ANOVA)

Test pour vérifier l’influence du type de sol sur le rendement.

Hypothèses :

H0 : Le type de sol n’influence pas le rendement.

H1 : Le type de sol influence le rendement.

Interprétation de la p-value : seuil de signification à 0.05.

4. Modélisation et évaluation

Comparaison de plusieurs modèles :

Régression linéaire (relation simple entre les variables).

Random Forest (capturer des interactions complexes).

XGBoost (amélioration des prédictions pour données non linéaires).

Métriques d'évaluation :

MAE (Erreur absolue moyenne)

RMSE (Erreur quadratique moyenne)

R² (variance expliquée par le modèle)

5. Recommandations et pistes d’amélioration

Optimisation du sol : favoriser les types de sol les plus productifs.

Utilisation optimale des engrais : ajuster les dosages et types.

Gestion de l’irrigation : adapter en fonction des précipitations.

Amélioration du modèle : prise en compte de nouvelles variables comme la qualité des semences ou la rotation des cultures.

Structure du Repository

code/ : Code source du projet.

README.md : Description du projet.

Instructions d’utilisation

Installation des dépendances :

pip install -r requirements.txt

Exécution des notebooks :

Ouvrir et exécuter les notebooks dans notebooks/ avec Jupyter Notebook.

Technologies utilisées

Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)

LateX pour le compte rendu

Licence

Ce projet est sous licence MIT. Vous êtes libre de le modifier et de l’utiliser à des fins académiques ou personnelles.
