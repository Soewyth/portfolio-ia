# ==============================
#    Import librairies
# ==============================
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
import os
import warnings

warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.ensemble import RandomForestRegressor  # type: ignore
from sklearn.metrics import mean_squared_error, r2_score  # type: ignore
from sklearn.preprocessing import LabelEncoder  # type: ignore


# ==============================
#      Chargement du dataset
# ==============================
df = pd.read_csv("train.csv")


# ==============================
#     Exploration du dataset
# ==============================

print("5 premières lignes du dataset : ")
print(df.head())

print("\n Informations générales du dataset : ")
print(df.info())

print("\n Statistiques description du dataset : ")
print(df.describe())

# ==============================
#       Corrélation numérique
# ==============================


output_dir_correlation = "1_correlation"
os.makedirs(output_dir_correlation, exist_ok=True)

# selection de colonnes numériques
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns
print("\n Colonnes numériques du dataset : ")
print(numerical_cols)
# Corrélation des colonnes numériques en rapport avec SalePrice et on trie par ordre décroissant Saleprice = 1 colonne, donc retourne Série pas dataframe
correlations = df[numerical_cols].corr()["SalePrice"].sort_values(ascending=False)
print("\n Correlation entre les colonnes numériques et SalePrice du dataset : ")
print(correlations.head(10))

# récupère les noms des 10 colonnes les plus corrélées à SalePrice
top_corr_cols = correlations.head(10).index


# Génération d'un heatmap entre les 10 colonnes les plus corrélées entre elles  :
plt.figure(figsize=(15, 10))
sns.heatmap(df[top_corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.ylabel("Colonnes Numériques")
plt.title("Matrice de corrélation")
plt.savefig(f"{output_dir_correlation}/Matrice_de_correlation.png")
plt.close()

# Génération d'un barplot en lien qu'avec SalePrice
plt.figure(figsize=(15, 10))
# Récupère values et index(noms des colonnes pour l'axe y )
sns.barplot(x=correlations[top_corr_cols].values, y=top_corr_cols, palette="coolwarm")
plt.xlabel("Valeurs Numériques")
plt.ylabel("Colonnes")
plt.title("Top 10 des colonnes les plus corrélées à SalePrice")
plt.savefig(f"{output_dir_correlation}/Barplot_de_correlation_SalePrice_Top10.png")
plt.close()

# ==============================
#     Corrélation catégorique
# ==============================

# Sélection des colonnes de types objets
categorical_cols = df.select_dtypes(include=("object")).columns
print("Colonne catégoriques du dataset : ")
print(categorical_cols)
# Nombre de catégories différentes pour chaques colonnes
print("Nombres de catégories différentes pour chaque colonnes du dataset : ")
numberCategoriesDifferentsCols = df[categorical_cols].nunique()
print(numberCategoriesDifferentsCols)

# Boucle pour créer des boxplots pour chaque colonnes < 15 catégories
max_categories = 15
output_dir = "0_boxplots"
os.makedirs(output_dir, exist_ok=True)

for col in numberCategoriesDifferentsCols.index:
    if numberCategoriesDifferentsCols[col] <= max_categories:
        plt.figure(figsize=(15, 10))
        sns.boxplot(x=col, y="SalePrice", data=df, palette="coolwarm")
        plt.title(f"Corrélation de {col} sur SalePrice")
        plt.xticks(rotation=45)
        plt.savefig(f"{output_dir}/{col}_boxplot.png")
        plt.close()

# Liste colonnes à encoder
cols_to_encode = [
    "Alley",
    "BsmtQual",
    "Condition2",
    "ExterQual",
    "FireplaceQu",
    "GarageType",
    "KitchenQual",
    "MiscFeature",
    "PoolQC",
    "RoofMatl",
    "SaleCondition",
    "SaleType",
]
# Label encoder
le = LabelEncoder()

# Map et transforme les var catégoriques en numériques
for col in cols_to_encode:

    df[col + "_encoded"] = le.fit_transform(df[col].astype(str))


# Calcul des corrélations
corr_dict = {}
for col in cols_to_encode:
    corr_value = df[col + "_encoded"].corr(df["SalePrice"])
    corr_dict[col + "_encoded"] = corr_value

# transformation en DataFrame, item() --> transforme le dictionnaire en key : valeur, list --> pour transformer en tuple et pouvoir filtrer ensuite (sort_values)
corr_df_categoriques = pd.DataFrame(
    list(corr_dict.items()), columns=["Feature", "Correlation"]
)


top_cat_features = (
    corr_df_categoriques.sort_values(
        by="Correlation", key=lambda x: x.abs(), ascending=False
    )
    .head(5)["Feature"]
    .tolist()
)

plt.figure(figsize=(15, 10))
sns.barplot(x="Correlation", y="Feature", data=corr_df_categoriques, palette="coolwarm")
plt.title("Corrélation approximatives des colonnes catégoriques avec SalePrice")
plt.xlabel("Corrélation avec SalePrice")
plt.ylabel("Colonnes catégoriques encodées")
plt.xticks(rotation=45)
plt.savefig(
    f"{output_dir_correlation}/Barplot_Correlation_Approx_Categoriques_SalePrices.png"
)
plt.close()

# ==============================
#       Modélisation
# ==============================

numerical_features = [
    col for col in top_corr_cols if col != "SalePrice"
]  # on parcourt et on récupère les cols sans SalePrice

features = numerical_features + top_cat_features
y = df["SalePrice"]
x = df[features]

# ========================================
#  Comparaison modèle LR & Random Forest
# ========================================

# Split train test
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=28
)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

rf = RandomForestRegressor(n_estimators=150, random_state=28)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# ==============================
#           Evaluation
# ==============================
# Linear Regression
print("Linear regression with R2 : ", r2_score(y_test, y_pred_lr))
print("Linear Regression RMSE :", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
# Random Forest
print("Random forest with R2 : ", r2_score(y_test, y_pred_rf))
print("Random Forest RMSE :", np.sqrt(mean_squared_error(y_test, y_pred_rf)))


# ==============================
#       Graphics Evaluation
# ==============================
output_dir_scatter = "2_evaluation_plots/scatter"
output_dir_line = "2_evaluation_plots/Line"
output_dir_histogram = "2_evaluation_plots/histogram"

os.makedirs(output_dir_scatter, exist_ok=True)
os.makedirs(output_dir_line, exist_ok=True)
os.makedirs(output_dir_histogram, exist_ok=True)

# ==============================
#           Scatter LR
# ==============================
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.6, color="blue")  # alpha = opacité
plt.plot(
    [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--"
)  # en utilisant la même ligne x et y --> ligne diagonale pour repérer l'endroit ou la pred serait parfaite
plt.xlabel("Valeurs réelles")
plt.ylabel("Prédictions Linear Regression")
plt.title("Scatter plot : Linear Regression")
plt.savefig(f"{output_dir_scatter}/scatter_lr.png")
plt.close()

# ==============================
#           Scatter RF
# ==============================
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.6, color="blue")  # alpha = opacité
plt.plot(
    [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--"
)  # en utilisant la même ligne x et y --> ligne diagonale pour repérer l'endroit ou la pred serait parfaite
plt.xlabel("Valeurs réelles")
plt.ylabel("Prédictions Random Forest")
plt.title("Scatter plot : Random Forest")
plt.savefig(f"{output_dir_scatter}/scatter_rf.png")
plt.close()

# ==============================
#          Line plot LR
# ==============================
# Ordre des indices pour tracer la série
indices = range(len(y_test))

plt.figure(figsize=(15, 6))
# Valeurs réelles
plt.plot(indices, y_test.values, label="Valeurs réelles", color="blue")
# Prédictions Linear Regression
plt.plot(indices, y_pred_lr, label="Predictions LR", color="red")
plt.title("Linear Regression : valeurs réelles vs prédictions")
plt.xlabel("Index")
plt.ylabel("SalePrice")
plt.legend()
plt.savefig(f"{output_dir_line}/lineplot_lr.png")
plt.close()

# ==============================
#          Line plot RF
# ==============================
plt.figure(figsize=(15, 6))
# Valeurs réelles
plt.plot(indices, y_test.values, label="Valeurs réelles", color="blue")
# Prédictions Random Forest
plt.plot(indices, y_pred_rf, label="Predictions RF", color="green")
plt.title("Random Forest : valeurs réelles vs prédictions")
plt.xlabel("Index")
plt.ylabel("SalePrice")
plt.legend()
plt.savefig(f"{output_dir_line}/lineplot_rf.png")
plt.close()

# ==============================
#           Histogram
# ==============================
# Erreurs
errors_lr = y_test - y_pred_lr
errors_rf = y_test - y_pred_rf

# Histogramme LR
plt.figure(figsize=(10, 6))
plt.hist(errors_lr, bins=20, color="blue", alpha=0.7)
plt.title("Histogramme des erreurs - Linear Regression")
plt.xlabel("Erreur (y_test - y_pred)")
plt.ylabel("Nombre d'échantillons")
plt.savefig(f"{output_dir_histogram}/histogram_lr.png")
plt.close()

# Histogramme RF
plt.figure(figsize=(10, 6))
plt.hist(errors_rf, bins=20, color="green", alpha=0.7)
plt.title("Histogramme des erreurs - Random Forest")
plt.xlabel("Erreur (y_test - y_pred)")
plt.ylabel("Nombre d'échantillons")
plt.savefig(f"{output_dir_histogram}/histogram_rf.png")
plt.close()

# ==============================
#     Importance des features
# ==============================

# Extraction des features importantes

importances = rf.feature_importances_
# Creation d'un dataframe avec les features , et l'importance
importances_features_df = pd.DataFrame({"Feature": features, "Importance": importances})

# Tri par ordre décroissant
importances_features_df = importances_features_df.sort_values(
    by="Importance", ascending=False
)

print("Top 10 des features les plus importantes :  \n")
print(importances_features_df.head(10))


output_dir_top_features = "3_topFeatures"
os.makedirs(output_dir_top_features, exist_ok=True)


plt.figure(figsize=(15, 10))
sns.barplot(
    x="Importance", y="Feature", data=importances_features_df, palette="viridis"
)
plt.title("Importance des variables dans le modèle Random Forest", fontsize=14)
plt.xlabel("Importances")
plt.ylabel("Features")
for i, (imp, feat) in enumerate(
    zip(importances_features_df["Importance"], importances_features_df["Feature"])
):
    plt.text(imp + 0.005, i, f"{imp:.3f}", va="center")

plt.tight_layout()
plt.savefig(f"{output_dir_top_features}/RandomForest_Feature_Importance.png")
plt.close()
