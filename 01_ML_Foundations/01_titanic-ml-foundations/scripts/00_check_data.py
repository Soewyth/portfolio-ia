from __future__ import annotations  # for future compatibility
from pathlib import Path
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent  #  scripts directory
ROOT_DIR = SCRIPTS_DIR.parent  # root directory


from titanic_ml_foundations.data.load import load_train_test
from titanic_ml_foundations.features.build import make_X_y
from titanic_ml_foundations.data.split import make_train_valid_split
from titanic_ml_foundations.config import TEST_SIZE, RANDOM_STATE



def explore_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    print("\n================= DATA CHECK =================")
    print(f"Train shape: {train_df.shape}")
    print(f"Test  shape: {test_df.shape}")

    print("\n--- Columns (train) ---")
    print(train_df.columns.tolist())

    print("\n--- Target (Survived) ---")
    print("Unique values:", train_df["Survived"].unique().tolist())
    print(f"Survival rate (mean): {train_df['Survived'].mean():.3f}")

    print("\n--- Head (train) ---")
    print(train_df.head(5))

    print("\n--- Missing values (top 10) ---")
    print(train_df.isnull().sum().sort_values(ascending=False).head(10))


def check_split(train_df: pd.DataFrame) -> None:
    X, y = make_X_y(train_df)
    X_train, X_valid, y_train, y_valid = make_train_valid_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print("\n================= SPLIT CHECK =================")
    print(f"X shape: {X.shape} | y shape: {y.shape}")
    print(f"X_train: {X_train.shape} | X_valid: {X_valid.shape}")
    print(f"y_train mean: {y_train.mean():.3f} | y_valid mean: {y_valid.mean():.3f}")

    print("\n--- Feature dtypes ---")
    print(X.dtypes)

    print("\n--- Missing values in X (sorted) ---")
    print(X.isnull().sum().sort_values(ascending=False))


if __name__ == "__main__":
    train_df, test_df = load_train_test(ROOT_DIR)
    explore_data(train_df, test_df)
    check_split(train_df)
