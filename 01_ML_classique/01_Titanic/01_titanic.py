# ========================
# Import des librairies
# ========================
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # type: ignore

# ========================
# Chargement du dataset
# ========================

df = sns.load_dataset("titanic")

# ========================w
# Exploration du dataset
# ========================

print("===== 5 premières lignes =====")
print(df.head())

print("\n===== Résumé info() =====")
print(df.info())

print("\n===== Résumé describe() =====")
print(df.describe())

# ========================
# Nettoyage et préparation
# ========================

# Afficher le nombres de valeurs non nulles pour chaque colonne
print("\n===== affichage des valeurs non nulles pour chaque colonne =====")
valeurs_manquantes = df.isnull().sum()
print(valeurs_manquantes)


df = df.drop(columns=["deck"])  # suppression de la colonne deck, trop de valeurs nulles
moyenne_age = df["age"].mean()  # Moyenne
df["age"] = df["age"].fillna(moyenne_age)  # Combler les NaN par moyenne_age


valeur_freq_embarked = df["embarked"].mode()[
    0
]  # Récuperer la valeur la plus fréquente d'embarked et stocker dans les valeurs nulles.
df["embarked"] = df["embarked"].fillna(valeur_freq_embarked)

valeur_freq_town = df["embark_town"].mode()[0]
df["embark_town"] = df["embark_town"].fillna(valeur_freq_town)

# Vérifier qu'il n'y a plus de valeurs manquantes
print("===== Valeurs manquantes après traitement =====")
print(df.isnull().sum())

# Encodage des variables catégorielles
# Quand une colonne a N catégories, get_dummies crée N colonnes.pip install jupyter

df = pd.get_dummies(df, columns=["sex", "embarked", "class"], drop_first=True)

# Création de nouvelles features
df["family_size"] = (
    df["sibsp"] + df["parch"]
)  # Création d'une colonne qui joint lien fraternel paternel

df["is_alone"] = df["family_size"].apply(
    lambda x: 1 if x == 0 else 0
)  # apply(fonction anonyme lambda, si family size = 1 alors is_alone = 0 et inverse)

print(df[["sibsp", "parch", "is_alone", "family_size"]].head())


# ========================
# Sélection des features et target
# ========================
features = [
    "pclass",
    "sex_male",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked_S",
    "embarked_Q",
    "class_Second",
    "class_Third",
    "is_alone",
    "family_size",
]
x = df[features]
y = df["survived"]

# ========================
# Split train/test
# ========================
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
# Vérifier les dimensions','
print("===== Nombres de lignes de datas utilisés pour le train et le test =====")
print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)


# ========================
# Modélisation
# ========================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
Y_pred = model.predict(X_test)

# ========================
# Évaluation du modèle
# ========================


print("===== Vérification de la cohérence des prédictions et valeurs réelles =====")
print("Prédictions :", Y_pred[:50])
print("Valeurs réelles :", y_test.values[:50])

accuracy = accuracy_score(y_test, Y_pred)
print("Accuracy :", accuracy)

# matrice de confusion : TN, FN, FP, TP
print("===== Matrice de confusion des prédictions =====")
cm = confusion_matrix(y_test, Y_pred)
print("Matrice de confusion \n:", cm)

# Classification report : Recall : parmi toutes les vraies valeurs 1, combien le modèle a détectées correctement ?
# Classification report : Support : Nombre réel d'exemples pour chaque classe

# Interprétation du rapport :
# Classe 0(non survivant) :Precision = 0.80 → parmi toutes les prédictions “non-survivant”, 80% étaient correctes.
# Recall = 0.86 → parmi tous les vrais non-survivants, le modèle en a identifié 86%.
# F1-score = 0.83 → un score global combinant précision et rappel.
# Support = 105 → il y avait 105 vrais non-survivants dans le test.

# Classe 1(survivant) :
# Precision = 0.78 → parmi toutes les prédictions “survécu”, 78% étaient correctes.
# Recall = 0.70 → parmi tous les vrais survivants, le modèle en a trouvé 70%.
# F1-score = 0.74 → score global pour cette classe.
# Support = 74 → 74 survivants dans le test.


print("===== Classification report =====")
report = classification_report(y_test, Y_pred)
print("Classification Report \n:", report)


print(
    '===== Création d\'un graphique "matrice_confusion.png" pour la visualisation ====='
)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Purples"
)  # annot=True(affiche les valeurs dans les cases), fmt="d" : format (valeur entières)
plt.xlabel("Prédiction")
plt.ylabel("Valeurs Réel")
plt.title("Matrice de confusion")


plt.savefig("matrice_confusion.png")
plt.close()


# ========================
# Probabilité de survie
# ========================

y_prob = model.predict_proba(X_test)[:, 1]

# Copier X_test pour ne pas modifier l'original
X_test_copy = X_test.copy()
X_test_copy["y_prob"] = y_prob
X_test_copy["y_true"] = y_test.values  # ajouter la vraie valeur pour référence


X_test_copy["age_group"] = pd.cut(X_test_copy["age"], bins=[0, 12, 18, 35, 60, 100])

# Moyenne par sexe
prob_by_sex = X_test_copy.groupby("sex_male")["y_prob"].mean()
print("=== Probabilité moyenne de survie par sexe ===")
print(prob_by_sex)
print()


# Moyenne par âge
prob_by_age = X_test_copy.groupby("age_group")["y_prob"].mean()
print("=== Probabilité moyenne de survie par tranche d'âge ===")
print(prob_by_age)
print()

# Moyenne par âge + sexe
prob_combined = X_test_copy.groupby(["sex_male", "age_group"])["y_prob"].mean()
print("=== Probabilité moyenne de survie combinée (sexe + âge) ===")
print(prob_combined)


# Création graphique par sexe
plt.figure(figsize=(6, 4))
sns.barplot(x=prob_by_sex.index, y=prob_by_sex.values, palette="pastel")
plt.xticks([0, 1], ["Femme", "Homme"])
plt.ylabel("Probabilité moyenne de survie")
plt.title("Survie prédite par sexe")
plt.ylim(0, 1)
plt.savefig("matrice_sex.png")
plt.close()


# Création graphique par tranche d'âge
plt.figure(figsize=(8, 5))
sns.barplot(x=prob_by_age.index.astype(str), y=prob_by_age.values, palette="coolwarm")
plt.xticks(rotation=45)
plt.ylabel("Probabilité moyenne de survie")
plt.xlabel("Tranche d'âge")
plt.title("Survie prédite par tranche d'âge")
plt.ylim(0, 1)
plt.savefig("matrice_age.png")
plt.close()


# Recode sexe pour affichage
prob_age_sex = (
    X_test_copy.groupby(["age_group", "sex_male"])["y_prob"].mean().reset_index()
)

prob_age_sex["sex"] = prob_age_sex["sex_male"].replace({0: "Femme", 1: "Homme"})


# Création graphique par tranche d'âge + sexe
plt.figure(figsize=(10, 5))
sns.barplot(x="age_group", y="y_prob", hue="sex", data=prob_age_sex, palette="pastel")
plt.ylabel("Probabilité moyenne de survie")
plt.xlabel("Tranche d'âge")
plt.title("Survie prédite par tranche d'âge et sexe")
plt.ylim(0, 1)
plt.legend(title="Sexe")
plt.savefig("matrice_combined.png")
plt.close()
