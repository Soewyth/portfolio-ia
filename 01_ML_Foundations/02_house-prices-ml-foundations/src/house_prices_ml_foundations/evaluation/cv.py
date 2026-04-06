from __future__ import annotations

import numpy as np
from sklearn.model_selection import cross_validate


def cross_validate_model(model: object, X: object, y: object, cv: object, scoring: list[str] | str) -> dict[str, dict[str, float]]:
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
    scoring = [scoring] if isinstance(scoring, str) else scoring  # for uniform process ,if str -> list

    # from sklearn
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)

    results = {}

    # Dict build of results
    for metric in scoring:
        key = f"test_{metric}"
        scores = cv_results[key]
        # if negative metrics -> * -1 to be in positive
        if metric.startswith("neg"):
            scores = -scores
            metric_name = metric.replace("neg_", "")  # remove prefix
        else:
            metric_name = metric

        results[metric_name] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
        }

    return results
