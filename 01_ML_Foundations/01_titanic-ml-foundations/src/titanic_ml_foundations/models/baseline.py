from __future__ import annotations  # for future compatibility
from titanic_ml_foundations.features.preprocess import build_preprocessor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


from titanic_ml_foundations.config import RANDOM_STATE, MAX_ITER_LOGREG


def build_baseline_model() -> Pipeline:
    """
    Build a baseline logistic regression model with preprocessing from build_preprocessor.
    Returns:
        Pipeline: Sklearn Pipeline with preprocessing and logistic regression model.
    """

    preprocess = build_preprocessor()

    model = LogisticRegression(random_state=RANDOM_STATE, max_iter=MAX_ITER_LOGREG)

    baseline_model = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
    return baseline_model
