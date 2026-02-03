from __future__ import annotations
from pathlib import Path
import pandas as pd


def load_train_test(root_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test datasets from CSV files in data/raw.

    Args:
        root_dir (Path): The root directory of the project.

    Raises:
        FileNotFoundError: If the root directory or data files are not found.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    train_path = root_dir / "datasets" / "raw" / "train.csv"
    test_path = root_dir / "datasets" / "raw" / "test.csv"

    if not root_dir.exists():
        raise FileNotFoundError(f"The specified root directory does not exist: {root_dir}")
    if not train_path.exists():
        raise FileNotFoundError(f"Training data file not found at: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Testing data file not found at: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df
