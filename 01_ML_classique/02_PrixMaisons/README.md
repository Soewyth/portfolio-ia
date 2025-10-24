# Projet Prix des Maisons 🏡

## 1) Objectif

Prédire le **prix de vente des maisons** à partir de leurs caractéristiques telles que :

- Surface habitable et superficie terrain
- Nombre de chambres et salles de bains
- Qualité des matériaux et finitions
- Présence de garage, piscine et autres commodités

Ce projet pédagogique permet de pratiquer **Python**, **pandas**, **seaborn**, **matplotlib**, et le **Machine Learning supervisé** avec **Linear Regression** et **Random Forest Regressor**.

---

## 2) Dataset

- **Fichier utilisé** : `train.csv`
- **Colonnes principales** : variables numériques et catégorielles, et la colonne cible `SalePrice`
- **Nombre d’échantillons** : **1460**

---

## 3) Exploration et prétraitement

- **Exploration initiale** : `df.head()`, `df.info()`, `df.describe()`
- **Corrélations numériques** : identification des 10 colonnes les plus corrélées avec `SalePrice`
- **Corrélations catégorielles** : encodage avec `LabelEncoder` pour certaines colonnes
- **Visualisations générées** :
  - Heatmap des corrélations : `1_correlation/Matrice_de_correlation.png`
  - Barplot des top 10 corrélations numériques : `1_correlation/Barplot_de_correlation_SalePrice_Top10.png`
  - Boxplots des colonnes catégorielles avec < 15 catégories : `0_boxplots/*.png`
  - Barplot des corrélations approximatives des colonnes catégorielles : `1_correlation/Barplot_Correlation_Approx_Categoriques_SalePrices.png`

---

## 4) Modélisation

- **Features sélectionnées** : combinaison des top 10 colonnes numériques + top 5 colonnes catégorielles encodées
- **Séparation train/test** : 80% / 20% (`train_test_split`)
- **Modèles entraînés** :
  - **Linear Regression**
  - **Random Forest Regressor** (150 arbres)

---

## 5) Évaluation

- **Metrics utilisées** :
  - R² score
  - RMSE (Root Mean Squared Error)

| Modèle            | R²  | RMSE |
| ----------------- | --- | ---- |
| Linear Regression | ... | ...  |
| Random Forest     | ... | ...  |

- **Visualisations d’évaluation** :
  - Scatter plot prédictions vs valeurs réelles : `2_evaluation_plots/scatter/scatter_lr.png` et `scatter_rf.png`
  - Line plot valeurs réelles vs prédictions : `2_evaluation_plots/Line/lineplot_lr.png` et `lineplot_rf.png`
  - Histogrammes des erreurs : `2_evaluation_plots/histogram/histogram_lr.png` et `histogram_rf.png`
  - Feature importance Random Forest : `3_topFeatures/RandomForest_Feature_Importance.png`

---

## 6) Observations

- Les features les plus importantes pour prédire le prix sont généralement : `OverallQual`, `GrLivArea`, `GarageCars`…
- Random Forest capture mieux les non-linéarités que la régression linéaire
- Les visualisations permettent d’identifier rapidement les prédictions proches ou éloignées des valeurs réelles

---

## 7) Conclusion

Ce projet permet de :

- Explorer et nettoyer un dataset complexe
- Comprendre l’importance de la corrélation et de l’encodage des colonnes catégorielles
- Comparer les performances de la **Linear Regression** et du **Random Forest Regressor**
- Visualiser et interpréter les résultats pour tirer des conclusions sur les facteurs influençant le prix des maisons

---

## 8) Instructions pour reproduire le projet

1. Cloner le dépôt ou copier les fichiers dans un dossier local
2. Créer un environnement virtuel :
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
