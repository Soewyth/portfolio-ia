from __future__ import annotations  # for future compatibility

import pandas as pd

from house_prices_ml_foundations.config.paths import get_paths, get_project_root

# ============== IMPORTS FROM house_prices_ml_foundations ==============
from house_prices_ml_foundations.data.load import load_train_test
from house_prices_ml_foundations.features.build import make_features


def explore_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Main function to check data loading."""

    print("================= DATA CHECK =================")
    print("Train DataFrame shape:", train_df.shape)
    print("Test DataFrame shape:", test_df.shape)

    print("================= COLUMNS =================")
    print("\n--- Train DataFrame columns ---")
    print(train_df.columns.tolist())
    print("================= HEAD=================")
    print("\n--- Train DataFrame head ---")
    print(train_df.head())
    print("\n--- Test DataFrame head ---")
    print(test_df.head())
    print("================= MISSING VALUES =================")
    print("\n--- Missing values in Train DataFrame (top 10) ---")
    print(train_df.isnull().sum().sort_values(ascending=False).head(10))


if __name__ == "__main__":
    root_dir = get_project_root()
    paths = get_paths(root_dir)
    train_df, test_df = load_train_test(root_dir)
    X, y = make_features(train_df)
    print("\n--- Features (X) shape ---")
    print(X.shape)
    print("\n--- Target (y) shape ---")
    print(y.shape)
    print("Sale price false in X columns:", "SalePrice" in X.columns)
    print("\n--- Target (y) description ---")
    print(y.describe())

    print("\n--- List of all features ---")
    print(X.dtypes)
    print("\n--- List of object features ---")
    object_features = X.select_dtypes(include=["object"]).columns.tolist()
    print(object_features)

    print("\n--- List of numbers features ---")
    numbers_features = X.select_dtypes(include=["number", "float64", "int64"]).columns.tolist()
    print(numbers_features)
