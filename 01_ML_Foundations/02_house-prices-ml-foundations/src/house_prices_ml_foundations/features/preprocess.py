from __future__ import annotations  # for future compatibility

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from house_prices_ml_foundations.features.schema import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    ORDINAL_FEATURES,
)


def build_preprocessor() -> ColumnTransformer:
    """Build a preprocessing pipeline for numerical and categorical features.
    Returns:
        ColumnTransformer: A preprocessing pipeline.
    """

    # Numerical pipeline: impute missing values with median
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "scaler",
                StandardScaler(),
            ),  # add feature scaling because some features have very different scales
        ]
    )

    # Categorical pipeline: impute missing values with most frequent and one-hot encode
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    # Ordinal pipeline: impute missing values with most frequent and ordinal encode

    ordinal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    # Combine pipelines into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
            ("ord", ordinal_pipeline, ORDINAL_FEATURES),
        ]
    )

    return preprocessor
