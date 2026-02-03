from __future__ import annotations
from sklearn.model_selection import cross_val_score


def cross_validate_model(
    model: object, X: object, y: object, cv: int, scoring: str
) -> tuple[float, float]:
    """
    Perform cross-validation on the given model.
    Args:
        model (object): The machine learning model to evaluate.
        X (object): Features dataset.
        y (object): Target dataset.
        cv (int): Number of cross-validation folds.
        scoring (str): Scoring metric to use.
    Returns:
        tuple[float, float]: Mean and standard deviation of the cross-validation scores.
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return scores.mean(), scores.std()
