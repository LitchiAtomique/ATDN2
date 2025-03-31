from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor
import time


from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

##############################PARTIE 1##############################################

# On charge les donnees avec Pandas
df = pd.read_csv("tp2_atdn_donnees.csv")

# Il y a une variable Qualitative, donc il faut la changer en etiquette (en entier)
le = LabelEncoder()
df['Type de sol'] = le.fit_transform(df['Type de sol'])

# On separe la base en deux sous-bases : La base entiere sans les variables 
# cibles et une base avec seulement les variables cible
X = df.drop(columns=["Température (°C)", "Humidité (%)"])
y = df[["Température (°C)", "Humidité (%)"]]


def rf_cv(n_estimators, max_depth, min_samples_leaf):
    """
    Fonction utilisée pour la validation croisée d'un modèle RandomForestRegressor dans le cadre de l'optimisation bayésienne.

    Paramètres :
    n_estimators (int) : Le nombre d'estimateurs dans le modèle.
    max_depth (int) : La profondeur maximale des arbres.
    min_samples_leaf (int) : Le nombre minimum d'échantillons par feuille.

    Retour :
    float : La moyenne de l'erreur quadratique moyenne (MSE) pour l'évaluation du modèle.
    """
    model = MultiOutputRegressor(RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_leaf=int(min_samples_leaf),
        random_state=42
    ))
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    return scores.mean()

def plot_convergence(results, title="Convergence de l'optimisation bayésienne"):
    """
    Fonction pour afficher un graphique de convergence des résultats d'une optimisation bayésienne.

    Paramètres :
    results (list) : Liste des résultats obtenus lors de l'optimisation.
    title (str) : Titre du graphique.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(results)), [res['target'] for res in results], marker='o', linestyle='--', color='b')
    plt.title(title)
    plt.xlabel("Nombre d'itérations")
    plt.ylabel("Score")
    plt.show()

def bayesian_optimization(plot=True):
    """
    Fonction qui effectue une optimisation bayésienne pour rechercher les meilleurs hyperparamètres du modèle RandomForestRegressor.

    Paramètres :
    plot (bool) : Si True, affiche le graphique de convergence de l'optimisation.
    
    Retour :
    tuple : Les meilleurs hyperparamètres trouvés et le meilleur score.
    """
    bo = BayesianOptimization(
        f=rf_cv,
        pbounds={'n_estimators': (10, 200), 'max_depth': (2, 50), 'min_samples_leaf': (1, 10)},
        verbose=2
    )
    bo.maximize(init_points=10, n_iter=50)

    best_params = bo.max['params']
    best_score = bo.max['target']
    
    print("Meilleurs hyperparamètres (Bayesian Optimization) :", best_params)
    print("Meilleur score (Bayesian Optimization) :", best_score)

    if plot:
        plot_convergence(bo.res)

    return best_params, best_score

def randomized_search(plot=True):
    """
    Fonction qui effectue une recherche aléatoire sur les hyperparamètres du modèle RandomForestRegressor.

    Paramètres :
    plot (bool) : Si True, affiche le graphique de convergence de la recherche.

    Retour :
    tuple : Les meilleurs hyperparamètres trouvés et le meilleur score.
    """
    model = RandomForestRegressor(random_state=42)
    param_dist = {
        'n_estimators': np.arange(10, 201),
        'max_depth': np.arange(2, 51),
        'min_samples_leaf': np.arange(1, 11)
    }
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=50,
        cv=5,
        scoring='neg_mean_squared_error',
        random_state=42,
        verbose=2
    )
    random_search.fit(X, y)

    best_params = random_search.best_params_
    best_score = random_search.best_score_
    
    print("Meilleurs hyperparamètres (RandomizedSearchCV) :", best_params)
    print("Meilleur score (RandomizedSearchCV) :", best_score)

    if plot:
        results = -random_search.cv_results_['mean_test_score']
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(results)), results, marker='o', linestyle='--', color='g')
        plt.title("Convergence de la recherche RandomizedSearchCV")
        plt.xlabel("Nombre d'itérations")
        plt.ylabel("Score (Erreur quadratique moyenne)")
        plt.show()

    return best_params, best_score

def grid_search(plot=True):
    """
    Fonction qui effectue une recherche exhaustive par grille sur les hyperparamètres du modèle RandomForestRegressor.

    Paramètres :
    plot (bool) : Si True, affiche le graphique de convergence de la recherche.

    Retour :
    tuple : Les meilleurs hyperparamètres trouvés et le meilleur score.
    """
    model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [10, 50, 100, 150, 200],
        'max_depth': [10, 20, 30, 40, 50],
        'min_samples_leaf': [1, 2, 5, 10]
    }
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        verbose=2
    )
    grid_search.fit(X, y)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print("Meilleurs hyperparamètres (GridSearchCV) :", best_params)
    print("Meilleur score (GridSearchCV) :", best_score)

    if plot:
        results = -grid_search.cv_results_['mean_test_score']
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(results)), results, marker='o', linestyle='--', color='r')
        plt.title("Convergence de la recherche GridSearchCV")
        plt.xlabel("Nombre d'itérations")
        plt.ylabel("Score (Erreur quadratique moyenne)")
        plt.show()

    return best_params, best_score

def training_bayesian(n, plot=True):
    """
    Fonction pour entraîner le modèle à l'aide de l'optimisation bayésienne.

    Paramètres :
    n (int) : Le nombre d'itérations d'entraînement.
    plot (bool) : Si True, affiche le graphique de convergence.
    """
    for i in range(n):
        print(f"Entrainement numéro {i+1} :")
        bayesian_optimization(False)
    return()

##############################PARTIE 2##############################################

def charger_donnees(df):
    """
    Fonction pour charger et séparer les données en variables explicatives et cibles.

    Paramètres :
    df (DataFrame) : Le DataFrame contenant les données.

    Retour :
    tuple : Les variables explicatives, les variables cibles pour la régression et la classification.
    """
    X = df.drop(columns=["Température (°C)", "Humidité (%)", "Type de sol"])
    y_reg = df[["Température (°C)", "Humidité (%)"]]  # Pour la régression
    y_clf = df["Type de sol"]  # Pour la classification
    return X, y_reg, y_clf

def normaliser_donnees(X):
    """
    Fonction pour normaliser les données.

    Paramètres :
    X (array-like) : Les données à normaliser.

    Retour :
    array : Les données normalisées.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def separation_train_test(X, y_reg, y_clf):
    """
    Fonction pour séparer les données en ensembles d'entraînement et de test.

    Paramètres :
    X (array-like) : Les variables explicatives.
    y_reg (array-like) : Les cibles pour la régression.
    y_clf (array-like) : Les cibles pour la classification.

    Retour :
    tuple : Les ensembles d'entraînement et de test pour la régression et la classification.
    """
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)
    return X_train, X_test, y_train_reg, y_test_reg, X_train_clf, X_test_clf, y_train_clf, y_test_clf

def regression_bayesienne(X_train, y_train_reg, X_test, kernel):
    """
    Fonction pour entraîner et évaluer un modèle de régression bayésienne.

    Paramètres :
    X_train (array-like) : Les données d'entraînement.
    X_test (array-like) : Les données de test.
    y_train (array-like) : Les cibles d'entraînement.
    y_test (array-like) : Les cibles de test.

    Retour :
    tuple : Les prédictions du modèle et l'erreur quadratique moyenne.
    """
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=42)
    gpr.fit(X_train, y_train_reg)
    y_pred, std_dev = gpr.predict(X_test, return_std=True)
    return y_pred, std_dev

def classification_bayesienne(X_train_clf, y_train_clf, X_test_clf, y_test_clf, kernel):
    """
    Fonction pour entraîner et évaluer un modèle de classification bayésienne avec un noyau spécifique.

    Paramètres :
    X_train_clf (array-like) : Les données d'entraînement.
    y_train_clf (array-like) : Les cibles d'entraînement.
    X_test_clf (array-like) : Les données de test.
    y_test_clf (array-like) : Les cibles de test.
    kernel (object) : Le noyau utilisé pour le modèle GaussianProcessClassifier.

    Retour :
    float : La précision du modèle sur l'ensemble de test.
    """
    gpc = GaussianProcessClassifier(kernel=kernel, random_state=42)
    gpc.fit(X_train_clf, y_train_clf)
    return gpc.score(X_test_clf, y_test_clf)

def comparaison_svm(X_train_clf, y_train_clf, X_test_clf, y_test_clf):
    """
    Fonction pour entraîner et évaluer un modèle de machine à vecteurs de support (SVM) avec noyau RBF.

    Paramètres :
    X_train_clf (array-like) : Les données d'entraînement.
    y_train_clf (array-like) : Les cibles d'entraînement.
    X_test_clf (array-like) : Les données de test.
    y_test_clf (array-like) : Les cibles de test.

    Retour :
    float : La précision du modèle SVM sur l'ensemble de test.
    """
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X_train_clf, y_train_clf)
    return svm.score(X_test_clf, y_test_clf)

def afficher_predictions(y_pred, std_dev, subset_size=50):
    """
    Fonction pour afficher les prédictions avec les intervalles de confiance pour la température et l'humidité.

    Paramètres :
    y_pred (array-like) : Les prédictions du modèle.
    std_dev (array-like) : L'écart type des prédictions (utilisé pour les intervalles de confiance).
    subset_size (int) : Nombre d'exemples à afficher dans les graphiques (par défaut 50).
    """
    # Affichage des prédictions pour la température
    plt.figure(figsize=(12, 6))
    indices = np.linspace(0, len(y_pred) - 1, min(subset_size, len(y_pred))).astype(int)
    plt.errorbar(indices, y_pred[indices, 0], yerr=std_dev[indices, 0], fmt='o', 
                 label='Température', alpha=0.7, markersize=5, capsize=3)
    plt.xlabel("Index")
    plt.ylabel("Température prédite")
    plt.title("Prédictions de Température avec intervalles de confiance")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()
    
    # Affichage des prédictions pour l'humidité
    plt.figure(figsize=(12, 6))
    plt.errorbar(indices, y_pred[indices, 1], yerr=std_dev[indices, 1], fmt='o', 
                 label='Humidité', alpha=0.7, markersize=5, capsize=3, color='green')
    plt.xlabel("Index")
    plt.ylabel("Humidité prédite")
    plt.title("Prédictions d'Humidité avec intervalles de confiance")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()

def tester_noyaux(X_train_clf, y_train_clf, X_test_clf, y_test_clf):
    """
    Fonction pour tester différents noyaux pour la classification bayésienne et afficher leurs précisions.

    Paramètres :
    X_train_clf (array-like) : Les données d'entraînement.
    y_train_clf (array-like) : Les cibles d'entraînement.
    X_test_clf (array-like) : Les données de test.
    y_test_clf (array-like) : Les cibles de test.
    """
    kernels = {"Linéaire": DotProduct(), "RBF": RBF(length_scale=1.0), "Matern": Matern()}
    for name, kernel in kernels.items():
        score = classification_bayesienne(X_train_clf, y_train_clf, X_test_clf, y_test_clf, kernel)
        print(f"Précision avec noyau {name}: {score:.3f}")


def main():

 # partie 1
    #training_bayesian(10,False)  # Pour entrainer le modele, a commenter ou decommenter.
    debut_bayesien=time.time()
    print("Début de la recherche Bayesian Optimization...")
    bayesian_optimization()
    fin_bayesien=time.time()

    debut_randoms=time.time()
    print("\nDébut de la recherche RandomizedSearchCV...")
    randomized_search()
    fin_randoms=time.time()

    debut_grids=time.time()
    print("\nDébut de la recherche GridSearchCV...")
    grid_search()
    fin_grids=time.time()

    print("\n Temps d'execution des 3 tests d'hyperparamètres:")
    print(f"Bayesian Optimization : {"{:.3f}".format(fin_bayesien-debut_bayesien)} secondes")
    print(f"RandomizedSearchCV : {"{:.3f}".format(fin_randoms-debut_randoms)} secondes")
    print(f"GridSearchCV : {"{:.3f}".format(fin_grids-debut_grids)}")


 # partie 2

    X, y_reg, y_clf = charger_donnees(df)
    X_scaled = normaliser_donnees(X)
    X_train, X_test, y_train_reg, y_test_reg, X_train_clf, X_test_clf, y_train_clf, y_test_clf = separation_train_test(X_scaled, y_reg, y_clf)
    
    # Régression bayésienne
    y_pred, std_dev = regression_bayesienne(X_train, y_train_reg, X_test, RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e5)))
    afficher_predictions(y_pred, std_dev)
    
    # Classification bayésienne
    score_bayesien = classification_bayesienne(X_train_clf, y_train_clf, X_test_clf, y_test_clf, RBF())
    print("Précision de la classification bayésienne:", score_bayesien)
    
    # Comparaison avec un SVM
    score_svm = comparaison_svm(X_train_clf, y_train_clf, X_test_clf, y_test_clf)
    print("Précision du SVM:", score_svm)
    
    # Test de différents noyaux
    tester_noyaux(X_train_clf, y_train_clf, X_test_clf, y_test_clf)

    return(0)
if __name__ == "__main__":
    main()
