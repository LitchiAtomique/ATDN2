# ATDN2 - TD1 : Analyse des Facteurs Impactant le Rendement Agricole

## 📌 Objectif du TP
L'objectif de ce projet est d'analyser les facteurs influençant le rendement agricole en utilisant des outils statistiques et de modélisation. L'étude permet d'identifier les variables les plus pertinentes et de formuler des recommandations pour optimiser la production.

## 🗂️ Structure du Projet
Ce projet est divisé en plusieurs étapes :

### 1️⃣ Compréhension du problème
- Identification des variables disponibles : rendement, précipitations, température, type de sol, engrais utilisé.
- Définition du problème métier : maximiser la production en ajustant les pratiques agricoles.
- Formulation de la problématique : comment optimiser le rendement tout en minimisant les coûts ?

### 2️⃣ Analyse statistique descriptive
- **Mesures de tendance centrale** : moyenne, médiane, mode du rendement.
- **Mesures de dispersion** : variance, écart-type, étendue.
- **Visualisation des données** : histogrammes, boîtes à moustaches pour détecter les outliers.
- **Analyse des corrélations** : heatmap de la matrice de corrélation pour identifier les variables les plus influentes.

### 3️⃣ Analyse de la variance (ANOVA)
- **Hypothèses** :
  - H₀ : Le type de sol n'a pas d'impact sur le rendement.
  - H₁ : Le type de sol a un impact significatif sur le rendement.
- **Interprétation de la p-value** :
  - p-value < 0.05 → Influence significative du type de sol.
  - p-value ≥ 0.05 → Pas de preuve suffisante d'influence.

### 4️⃣ Modélisation et évaluation
- **Méthodes d’évaluation** :
  - MAE (Erreur Absolue Moyenne)
  - RMSE (Erreur Quadratique Moyenne)
  - R² (Coefficient de détermination)
- **Modèles utilisés** :
  - Régression linéaire pour les relations simples.
  - Random Forest ou XGBoost pour capturer des interactions plus complexes.

### 5️⃣ Recommandations et améliorations
- Optimisation du sol et utilisation ciblée des engrais.
- Meilleure gestion de l’irrigation en fonction des précipitations.
- Tests de modèles plus avancés si les performances sont insuffisantes.
- Validation croisée et augmentation de la qualité des données.

## 🛠️ Technologies utilisées
- **Python** : NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn.
- **Outils statistiques** : ANOVA, analyse de corrélation.
- **Modélisation** : Régression linéaire, Random Forest, XGBoost.

## 📊 Résultats et Conclusions
- Les variables les plus influentes sont le **type de sol** et **l'utilisation des engrais**.
- Le modèle optimal dépend du niveau de complexité des relations entre variables.
- Des ajustements ciblés des pratiques agricoles peuvent améliorer significativement le rendement.

## 📎 Fichiers du projet
- `data/` : Contient les jeux de données utilisés pour l'analyse.
- `code/` : Contient les notebooks Jupyter avec les analyses et visualisations.

## ✍️ Auteur
**Remy XU** - M1 OIVM  
*Date : 26 mars 2025*
