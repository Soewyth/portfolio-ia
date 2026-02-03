from __future__ import annotations
import pandas as pd

from titanic_ml_foundations.features.schema import TARGET_COL, FEATURES_COLS


def get_feature_columns() -> list[str]:
    """Return the list of feature column names."""
    return FEATURES_COLS


def make_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Given a DataFrame, return the features DataFrame X and target Series y."""
    X = df[FEATURES_COLS]
    y = df[TARGET_COL]
    return X, y
