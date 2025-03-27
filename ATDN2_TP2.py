import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

def main():
    df = pd.read_csv("donnees_elevage_poulet.csv")
    print(df)

    #QUESTION 1.1:

    print("Voici les moyennes, les valeurs medianes, les ecarts types, les variances, et les quartiles du poids, de la nourriture et de la temperature : ")

    print(f"\nMoyenne du poids : {df['Poids_poulet_g'].mean()}")
    print(f"Moyenne de la nourriture consommee : {df['Nourriture_consommee_g_jour'].mean()}")
    print(f"Moyenne de la temperature de l'enclos : {df['Temperature_enclos_C'].mean()}")

    print(f"\nMediane du poids : {df['Poids_poulet_g'].median()}")
    print(f"Mediane de la nourriture consommee : {df['Nourriture_consommee_g_jour'].median()}")
    print(f"Mediane de la temperature de l'enclos : {df['Temperature_enclos_C'].median()}")

    print(f"\nEcart-type du poids : {df['Poids_poulet_g'].std()}")
    print(f"Ecart-type de la nourriture consommee : {df['Nourriture_consommee_g_jour'].std()}")
    print(f"Ecart-type de la temperature de l'enclos : {df['Temperature_enclos_C'].std()}")

    print(f"\nVariance du poids : {df['Poids_poulet_g'].var()}")
    print(f"Variance de la nourriture consommee : {df['Nourriture_consommee_g_jour'].var()}")
    print(f"Variance de la temperature de l'enclos : {df['Temperature_enclos_C'].var()}")

    print(f"\nQuantiles du poids :\n{df['Poids_poulet_g'].quantile([0.25, 0.50, 0.75])}")
    print(f"Quantiles de la nourriture consommee :\n{df['Nourriture_consommee_g_jour'].quantile([0.25, 0.50, 0.75])}")
    print(f"Quantiles de la temperature de l'enclos :\n{df['Temperature_enclos_C'].quantile([0.25, 0.50, 0.75])}")


    #QUESTION 1.2:


    plt.figure(figsize=(12, 6))
    sns.histplot(df['Poids_poulet_g'], bins=20, kde=True)
    plt.title("Distribution du poids")
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.histplot(df['Nourriture_consommee_g_jour'], bins=20, kde=True)
    plt.title("Distribution de la nourriture")
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.histplot(df['Temperature_enclos_C'], bins=20, kde=True)
    plt.title("Distribution de la temperature")
    plt.show()


    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    sns.boxplot(df['Poids_poulet_g'], ax=axes[0])
    sns.boxplot(df['Nourriture_consommee_g_jour'], ax=axes[1])
    sns.boxplot(df['Temperature_enclos_C'], ax=axes[2])
    axes[0].set_title("Boxplot poids")
    axes[1].set_title("Boxplot nourriture")
    axes[2].set_title("Boxplot Temperature")
    plt.show()



    #Question 2.1

    print(f"\nEcart Interquartile :\n{df['Poids_poulet_g'].quantile(0.75)-df['Poids_poulet_g'].quantile(0.25)}")
    print(f"\nEcart Interquartile :\n{df['Nourriture_consommee_g_jour'].quantile(0.75)-df['Nourriture_consommee_g_jour'].quantile(0.25)}")
    print(f"\nEcart Interquartile :\n{df['Temperature_enclos_C'].quantile(0.75)-df['Temperature_enclos_C'].quantile(0.25)}")

    z_scores = stats.zscore(df)
    print("Z-Scores:", z_scores)

    #Question 2.2

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    sns.boxplot(z_scores['Poids_poulet_g'], ax=axes[0])
    sns.boxplot(z_scores['Nourriture_consommee_g_jour'], ax=axes[1])
    sns.boxplot(z_scores['Temperature_enclos_C'], ax=axes[2])
    axes[0].set_title("Boxplot poids")
    axes[1].set_title("Boxplot nourriture")
    axes[2].set_title("Boxplot Temperature")
    plt.show()



    print(shapiro(df['Poids_poulet_g']))
    print(shapiro(df['Nourriture_consommee_g_jour']))
    print(shapiro(df['Temperature_enclos_C']))

    df_subset = df.iloc[:, :3]
    pop1 = df_subset.sample(n=20, random_state=42)
    pop2 = df_subset.drop(pop1.index).sample(n=20, random_state=42)

    y = stats.ttest_ind(pop1,pop2)

    print(stats.f_oneway(pop1,pop2))

    #Questions 4.1

    # Etape 1: Standardiser les données
    z_scores = stats.zscore(df)

    # Etape 2: Calcul de la matrice de corrélation

    z_scores_df = pd.DataFrame(z_scores, columns=df.columns)
    corr = z_scores_df.corr()

    sns.heatmap(corr, cmap="coolwarm", annot=True, linewidths=0.9)
    plt.show()

    # Etape 3: Calcul des valeurs propres et vecteurs propres
    eigenvalues, eigenvectors = np.linalg.eigh(corr)

    sorted_indices = np.argsort(eigenvalues)[::-1]  # indices triés dans l'ordre décroissant
    eigenvalues_sorted = eigenvalues[sorted_indices]
    eigenvectors_sorted = eigenvectors[:, sorted_indices]

    # Etape 4: Projection des données sur les k premières composantes principales
    k = 2 
    principal_components = eigenvectors_sorted[:, :k]
    X_pca = np.dot(z_scores, principal_components)

    plt.scatter(X_pca[:, 0], X_pca[:, 1])
    plt.xlabel("Composante principale 1")
    plt.ylabel("Composante principale 2")
    plt.show()


    #Matrice de covariance, valeurs propres et vecteurs propres.

    print(np.cov(z_scores))
    print(eigenvalues)
    print(eigenvectors)

    #Question 4.2
    #Calcul de la proportion de variance expliquée par chaque composante
    explained_variance = eigenvalues_sorted / np.sum(eigenvalues_sorted)

    # Cumul de la variance expliquée
    cumulative_variance = np.cumsum(explained_variance)

    # Affichage de la variance expliquée et du cumul
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
    plt.xlabel("Nombre de composantes principales")
    plt.ylabel("Variance expliquée cumulée")
    plt.title("Variance expliquée cumulée par les composantes principales")
    plt.grid(True)
    plt.show()

    # Code a ne fonctionne pas et bloque la suite du code pour une raison que j'ignore
    """
    # Noyau linéaire
    kpca_linear = KernelPCA(kernel="linear", n_components=2)
    X_kpca_linear = kpca_linear.fit_transform(X)

    # Noyau RBF
    kpca_rbf = KernelPCA(kernel="rbf", gamma=0.1, n_components=2)
    X_kpca_rbf = kpca_rbf.fit_transform(X)

    # Noyau polynomial
    kpca_poly = KernelPCA(kernel="poly", degree=3, n_components=2)
    X_kpca_poly = kpca_poly.fit_transform(X)

    # Visualisation des résultats
    plt.figure(figsize=(18, 6))

    plt.subplot(131)
    plt.scatter(X_kpca_linear[:, 0], X_kpca_linear[:, 1], c=y)
    plt.title("KernelPCA - Linéaire")

    plt.subplot(132)
    plt.scatter(X_kpca_rbf[:, 0], X_kpca_rbf[:, 1], c=y)
    plt.title("KernelPCA - RBF")

    plt.subplot(133)
    plt.scatter(X_kpca_poly[:, 0], X_kpca_poly[:, 1], c=y)
    plt.title("KernelPCA - Polynomial")

    plt.show()"""



    #QUESTION 6.1
    # Discrétisation en 2 catégories : faible (80-90) et élevée (90-100)
    bins = [0,80, 90, 100]  # Définir les intervalles
    labels = [0, 1,2]  # Noms des catégories
    df['Taux_survie_cat'] = pd.cut(df['Taux_survie_%'], bins=bins, labels=labels)

    X = df.drop(['Taux_survie_%', 'Taux_survie_cat'], axis=1)  # Ne pas inclure la colonne cible 'Taux_survie_%' ni la colonne catégorielle
    y = df['Taux_survie_cat']  # Utiliser la colonne 'Taux_survie_cat' comme cible
    print(y)
    # Diviser les données en ensembles d'apprentissage et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Entraîner le modèle RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Prédictions
    y_pred = rf.predict(X_test)

    # Analyse des performances
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1-score: {f1:.2f}")    

    
    #Question 6.2
    # Affichage des variables importantes
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Affichage des résultats
    for i in range(X.shape[1]):
        print(f"{X.columns[indices[i]]}: {importances[indices[i]]:.4f}")


    #Question 7.1
    X = df.drop(columns=["Gain_poids_jour_g"])
    y = df["Gain_poids_jour_g"]

    # Division en jeu d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    ada_model = AdaBoostRegressor(n_estimators=100, random_state=42)
    ada_model.fit(X_train, y_train)
    y_pred_ada = ada_model.predict(X_test)

    # Modèle Gradient Boosting
    gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)

    # Évaluation des performances
    mse_ada = mean_squared_error(y_test, y_pred_ada)
    r2_ada = r2_score(y_test, y_pred_ada)

    mse_gb = mean_squared_error(y_test, y_pred_gb)
    r2_gb = r2_score(y_test, y_pred_gb)

    print(f"AdaBoost - MSE: {mse_ada:.4f}, R²: {r2_ada:.4f}")
    print(f"Gradient Boosting - MSE: {mse_gb:.4f}, R²: {r2_gb:.4f}")

    return(0)




if __name__=="__main__":
    main()