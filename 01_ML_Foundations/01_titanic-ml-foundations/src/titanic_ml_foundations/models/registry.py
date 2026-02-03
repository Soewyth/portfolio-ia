from __future__ import annotations  # for future compatibility

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from titanic_ml_foundations.features.preprocess import build_preprocessor
from titanic_ml_foundations.config import RANDOM_STATE, MAX_ITER_LOGREG, N_ESTIMATORS_RF


def make_model_registry() -> dict[str, str]:
    """Create a registry of available models.
    Args:
        None

    Returns:
        dict[str, str]: A dictionary mapping model names to their module paths.
    """
    preprocess = build_preprocessor()
    logreg = LogisticRegression(random_state=RANDOM_STATE, max_iter=MAX_ITER_LOGREG)
    rf = RandomForestClassifier(
        random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS_RF, n_jobs=-1
    )

    return {
        "logreg": Pipeline(
            steps=[("preprocessor", preprocess), ("classifier", logreg)]
        ),
        "rf": Pipeline(steps=[("preprocessor", preprocess), ("classifier", rf)]),
    }
