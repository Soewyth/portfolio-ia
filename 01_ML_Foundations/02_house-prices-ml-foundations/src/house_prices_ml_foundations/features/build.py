from __future__ import annotations  # for future compatibility

import pandas as pd

from house_prices_ml_foundations.features.schema import FEATURES_COLS, TARGET_COL


def get_feature_columns() -> list[str]:
    """Return the list of feature column names."""
    return FEATURES_COLS


def get_target_name() -> str:
    """Return the name of the target column."""
    return TARGET_COL


def make_features(df: pd.DataFrame, return_target: bool = True) -> tuple[pd.DataFrame, pd.Series] | pd.DataFrame:
    """Build features from raw dataframe and optionally return target."""
    df = df.copy()  # avoid modifying
    df["house_age"] = df["YrSold"] - df["YearBuilt"]
    df["garage_age"] = df["YrSold"] - df["GarageYrBlt"]
    df["remod_age"] = df["YrSold"] - df["YearRemodAdd"]

    X = df[FEATURES_COLS]

    if not return_target:
        print(f"Features shape : {X.shape}")
        return X

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' is missing. Use return_target=False for inference/test data.")

    y = df[TARGET_COL]
    print(f"Features shape : {X.shape} and target shape : {y.shape}")
    return X, y
