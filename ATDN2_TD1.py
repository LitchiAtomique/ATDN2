import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    df = pd.read_csv("rendement_mais.csv")

    # Encodage de la variable "TYPE_SOL" avec LabelEncoder
    encoder = LabelEncoder()
    df["TYPE_SOL_ENCODE"] = encoder.fit_transform(df["TYPE_SOL"])
    print("DataFrame avec encodage de TYPE_SOL:")
    print(df)

    # QUESTION 2.1
    rendement = df["RENDEMENT_T_HA"]

    # Calcul de la moyenne
    mean = rendement.mean()
    print(f"Moyenne du rendement : {mean:.2f}")

    # Calcul de la médiane
    median = rendement.median()
    print(f"Médiane du rendement : {median:.2f}")

    # Calcul du mode
    mode = rendement.mode().iloc[0]  # On prend uniquement la première valeur si plusieurs modes
    print(f"Mode du rendement (valeur la plus fréquente) : {mode:.2f}")
    
    # QUESTION 2.2
    # Variance
    variance = rendement.var()
    print(f"Variance du rendement : {variance:.2f}")

    # Écart type
    standard_deviation = rendement.std()
    print(f"Écart type du rendement : {standard_deviation:.2f}")

    # Étendue
    range_rendement = rendement.max() - rendement.min()
    print(f"Étendue du rendement (max - min) : {range_rendement:.2f}")

    # QUESTION 2.3
    # Histogrammes
    plt.figure(figsize=(10, 6))  # Taille de la figure pour plus de lisibilité

    plt.subplot(1, 3, 1)  # Subplot 1
    plt.hist(df['RENDEMENT_T_HA'], bins=4, color='blue', edgecolor='black')
    plt.title('Histogramme de RENDEMENT_T_HA')
    plt.xlabel('Rendement (t/ha)')
    plt.ylabel('Fréquence')

    plt.subplot(1, 3, 2)  # Subplot 2
    plt.hist(df['PRECIPITATIONS_MM'], bins=4, color='green', edgecolor='black')
    plt.title('Histogramme de PRECIPITATIONS_MM')
    plt.xlabel('Précipitations (mm)')
    plt.ylabel('Fréquence')

    plt.subplot(1, 3, 3)  # Subplot 3
    plt.hist(df['TEMPERATURE_C'], bins=4, color='red', edgecolor='black')
    plt.title('Histogramme de TEMPERATURE_C')
    plt.xlabel('Température (°C)')
    plt.ylabel('Fréquence')

    plt.tight_layout()  # Ajuste l'espace entre les subplots
    plt.show()

    # Boxplot
    plt.figure(figsize=(8, 6))  # Taille de la figure pour le boxplot
    boxplot = df.boxplot(column=["RENDEMENT_T_HA", "PRECIPITATIONS_MM", "TEMPERATURE_C"], grid=True)
    plt.title('Boxplot des variables de rendement, précipitations et température')
    plt.ylabel('Valeurs')
    plt.show()

    # QUESTION 2.4
    # Vous pouvez ajouter ici d'autres analyses ou visualisations, si nécessaire.


    # Étape 3 : Test ANOVA sur le type de sol
    anova_result = stats.f_oneway(
        *[df[df['Type_Sol'] == soil]['Rendement'] for soil in df['Type_Sol'].unique()]
    )
    print("\nRésultat ANOVA : F =", anova_result.statistic, ", p-value =", anova_result.pvalue)
    if anova_result.pvalue < 0.05:
        print("Le type de sol influence significativement le rendement.")
    else:
        print("Le type de sol n'a pas d'effet significatif sur le rendement.")

    # Étape 4 : Modélisation
    # Séparation des données
    X = df.drop(columns=['Rendement'])
    y= df['Rendement']
    X = pd.get_dummies(X, drop_first=True)  # Encodage des variables catégorielles
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modèle de régression linéaire
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Évaluation du modèle
    print("\nÉvaluation du modèle :")
    print("MAE :", mean_absolute_error(y_test, y_pred))
    print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R² :", r2_score(y_test, y_pred))

    # Étape 5 : Interprétation et recommandations
    importances = pd.Series(model.coef_, index=X.columns)
    print("\nImportance des variables :\n", importances.sort_values(ascending=False))

    print("\nRecommandations :")
    print("- Si les précipitations sont fortement corrélées au rendement, ajuster l'irrigation.")
    print("- Si la température impacte négativement, envisager des variétés résistantes à la chaleur.")
    print("- Si le type de sol est significatif, favoriser les meilleurs types identifiés.")
    return 0

if __name__ == "__main__":
    main()
