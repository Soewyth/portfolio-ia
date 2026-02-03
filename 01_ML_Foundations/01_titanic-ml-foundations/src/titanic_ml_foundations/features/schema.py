# Feature schema for the Titanic dataset
TARGET_COL = "Survived"  # Target variable
FEATURES_COLS = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
]  # All feature columns


NUMERICAL_FEATURES = [
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Pclass",
]  # Numerical feature columns
CATEGORICAL_FEATURES = ["Sex", "Embarked"]  # Categorical feature columns
