# Projet Titanic

## 1) Objectif

L’objectif de ce projet est de prédire la survie des passagers du Titanic à partir de caractéristiques telles que :

- Âge
- Sexe
- Classe de cabine
- Nombre de frères/sœurs et parents à bord
- Tarif du billet

Le but principal de ce projet est pédagogique : mettre en pratique Python et ses bibliothèques (pandas, NumPy, seaborn, matplotlib) et le Machine Learning (régression logistique). Beaucoup de commentaires sont présents dans le code pour faciliter l’auto-apprentissage.

---

## 2) Dataset

Le dataset utilisé est le **dataset Titanic** disponible via **Seaborn**.

- Nombre de passagers : 891
- Colonnes principales : celles qui ont été **converties, créées ou utilisées** comme features dans le modèle.

---

## 3) Nettoyage et préparation

- Suppression de colonnes avec trop de valeurs manquantes : `deck`
- Remplissage des valeurs manquantes :

  - `age` → moyenne des âges
  - `embarked` et `embark_town` → valeur la plus fréquente

- Encodage des variables catégorielles (`sex`, `embarked`, `class`) avec `get_dummies`
- Création de nouvelles colonnes (**feature engineering**) :
  - `family_size` = `sibsp` + `parch`
  - `is_alone` = 1 si `family_size == 0`, sinon 0

---

## 4) Modélisation

- Séparation des données en **train (80%)** et **test (20%)** avec `train_test_split` et `random_state=42`
- Modèle : **Logistic Regression**
- Entraînement sur `X_train` et `y_train`
- Prédiction sur `X_test`

---

## 5) Évaluation

- **Accuracy** : ~0.79
- **Matrice de confusion** :

- **Classification report** : précision, rappel, F1-score pour chaque classe

- Visualisation :
  - Matrice de confusion sauvegardée dans `matrice_confusion.png`
  - Probabilités de survie par sexe (`matrice_sex.png`)
  - Probabilités de survie par tranche d’âge (`matrice_age.png`)
  - Probabilités combinées sexe + âge (`matrice_combined.png`)

---

## 6) Conclusion

- Le modèle prédit correctement environ **79% des passagers**
- La précision est meilleure pour la classe majoritaire (non-survivants)
- Ce projet permet de comprendre **l’ensemble du pipeline ML** : exploration des données, nettoyage, feature engineering, modélisation et évaluation.

## 7) Instructions pour reproduire le projet

1. Cloner le dépôt ou copier les fichiers dans un dossier local
2. Créer un environnement virtuel :
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
