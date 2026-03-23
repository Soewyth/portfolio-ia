from __future__ import annotations  # for future compatibility
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from pathlib import Path
import pandas as pd

# ============== IMPORTS FROM house_prices_ml_foundations ==============
# data imports
from house_prices_ml_foundations.data.load import load_train_test
from house_prices_ml_foundations.features.build import make_features
from house_prices_ml_foundations.data.split import make_train_valid_split
from house_prices_ml_foundations.features.preprocess import build_preprocessor
from house_prices_ml_foundations.models.baseline import build_ridge_pipeline

# config imports
from house_prices_ml_foundations.config import TEST_SIZE, RANDOM_STATE
from house_prices_ml_foundations.features.schema import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ORDINAL_FEATURES,
)


SCRIPTS_DIR = Path(__file__).resolve().parent  #  scripts directory
ROOT_DIR = SCRIPTS_DIR.parent  # root directory


def main():
    """Main function to run the baseline model."""
    # Model parameters
    test_size = TEST_SIZE
    random_state = RANDOM_STATE

    # Load data
    print(f"=== Loading data from : {ROOT_DIR}... ===")
    train_df, test_df = load_train_test(ROOT_DIR)
    # Prepare features and target
    print("=== Preparing features and target ===")
    X, y = make_features(train_df)
    # Split data into training and validation sets
    print("=== Splitting data into training and validation sets ===")
    X_train, X_valid, y_train, y_valid = make_train_valid_split(
        X, y, test_size, random_state
    )

    print(
        f"Data shape after split -> X_train : {X_train.shape}, X_valid : {X_valid.shape}, y_train : {y_train.shape}, y_valid : {y_valid.shape}"
    )
    print(
        f" y mean in train set : {y_train.mean():.3f} and in validation set : {y_valid.mean():.3f}"
    )

    pipe_ridge = build_ridge_pipeline(alpha=1.0)  # preprocess and model app
    pipe_ridge.fit(X_train, y_train)  # fit datas

    y_pred = pipe_ridge.predict(X_valid)  # predict to x_valid

    # metrics
    mae = mean_absolute_error(y_valid, y_pred)
    rmse = root_mean_squared_error(y_valid, y_pred)
    r2 = r2_score(y_valid, y_pred)

    print("=== Metrics accuracy of Ridge model ===")
    print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}\n")


if __name__ == "__main__":
    main()
