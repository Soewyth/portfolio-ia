from __future__ import annotations
from sklearn.model_selection import train_test_split

from titanic_ml_foundations.config import RANDOM_STATE, TEST_SIZE

def make_train_valid_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Split features and target into training and validation sets.

    Args:
        X (pd.DataFrame): Features DataFrame.
        y (pd.Series): Target Series.
        test_size: Proportion of the dataset to include in the validation split.
        random_state : Random seed for reproducibility. Defaults to RANDOM_STATE.
    Returns:
        tuple: Tuple containing training features, validation features, training target, and validation target.
    """ 
    # Stratify by y to maintain the same proportion of classes in train and valid sets 
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    return X_train, X_valid, y_train, y_valid