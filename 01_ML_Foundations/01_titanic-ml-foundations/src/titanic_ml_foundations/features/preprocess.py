from __future__ import annotations
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from titanic_ml_foundations.features.schema import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)


def build_preprocessor() -> ColumnTransformer:
    """Build a preprocessing pipeline for numerical and categorical features.
    Returns:
        ColumnTransformer: A preprocessing pipeline.
    """

    # Numerical features : Impute missing values with median
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    # Categorical features : Impute missing values with most frequent and one-hot encode (for ignoring unknown categories)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Combine numerical and categorical transformers into a single matrix
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor
