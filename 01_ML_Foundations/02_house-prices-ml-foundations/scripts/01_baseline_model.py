from __future__ import annotations  # for future compatibility
from pathlib import Path
import pandas as pd

# ============== IMPORTS FROM house_prices_ml_foundations ==============
# data imports
from house_prices_ml_foundations.data.load import load_train_test
from house_prices_ml_foundations.features.build import make_X_y
from house_prices_ml_foundations.data.split import split_data

# config imports
from house_prices_ml_foundations.config import TEST_SIZE, RANDOM_STATE


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
    X, y = make_X_y(train_df)
    # Split data into training and testing sets
    print("=== Splitting data into training and testing sets ===")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)


if __name__ == "__main__":
    main()
