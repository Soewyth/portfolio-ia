from __future__ import annotations  # for future compatibility
import pandas as pd

from house_prices_ml_foundations.features.schema import FEATURES_COLS, TARGET_COL


def get_feature_columns() -> list[str]:
    """Return the list of feature column names."""
    return FEATURES_COLS


def get_target_name() -> str:
    """Return the name of the target column."""
    return TARGET_COL


def make_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Given a DataFrame, return the features DataFrame X and target Series y."""
    df = df.copy()  # avoid modifying 
    df["house_age"] = df["YrSold"] - df["YearBuilt"]  # new feature: age of the house at the time of sale
    df["garage_age"] = df["YrSold"] - df["GarageYrBlt"]  # new feature: age of the garage at the time of sale
    df["remod_age"] = df["YrSold"] - df["YearRemodAdd"]  # new feature: age since last remodel at the time of sale
    X = df[FEATURES_COLS]
    y = df[TARGET_COL]
    print(f"Features shape : {X.shape} and target shape : {y.shape}")
    return X, y
