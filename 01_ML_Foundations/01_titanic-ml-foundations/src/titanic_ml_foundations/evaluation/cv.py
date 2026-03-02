from __future__ import annotations
import numpy as np
from sklearn.model_selection import cross_validate


def cross_validate_model(
    model: object, X: object, y: object, cv: object, scoring: list[str] | str
) -> dict[str, dict[str, float]]:
    """
    Perform cross-validation on the given model.
    Args:
        model (object): The machine learning model to evaluate.
        X (object): Features dataset.
        y (object): Target dataset.
        cv (object): Cross-validation strategy.
        scoring (str | list[str]): Scoring metric(s) to use.
    Returns:
        Dict [str, dict[str, float]]: A dictionary containing the mean and standard deviation of the scores.
    """

    cv_out = cross_validate(model, X, y, cv=cv, scoring=scoring)
    results: dict[str, dict[str, float]] = {}

    scorings = (
        [scoring] if isinstance(scoring, str) else scoring
    )  # if single string, convert to list for uniform processing
    for s in scorings:
        score_key = f"test_{s}" if f"test_{s}" in cv_out else "test_score"
        vals = cv_out[score_key]
        results[s] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    return results
