from __future__ import annotations
from sklearn.model_selection import train_test_split

from house_prices_ml_foundations.config import TEST_SIZE, RANDOM_STATE


def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Splits the data into training and testing sets.

    Args:
        X (pd.DataFrame): The features.
        y (pd.Series): The target variable.
        test_size (float): The proportion of the dataset to include in the test split. Defaults to TEST_SIZE.
        random_state (int): Controls the shuffling applied to the data before applying the split. Defaults to RANDOM_STATE.

    Returns:
        tuple: A tuple containing the training and testing sets (X_train, X_test, y_train, y_test).
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(
        f"Data shape after split -> X_train : {X_train.shape}, X_test : {X_test.shape}, y_train : {y_train.shape}, y_test : {y_test.shape}"
    )
    print(f" y mean in train set : {y_train.mean():.3f} and in test set : {y_test.mean():.3f}")
    return X_train, X_test, y_train, y_test
