from __future__ import annotations  # for future compatibility
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from house_prices_ml_foundations.features.preprocess import build_preprocessor
from house_prices_ml_foundations.config import (
    RANDOM_STATE,
    N_ESTIMATORS_RF,
    MAX_ITER_LASSO,
)


def build_ridge_pipeline(alpha: float = 1.0) -> Pipeline:
    """Build a Ridge regression pipeline.
    Args:
        alpha: Regularization strength
    Returns:
        Pipeline: A Ridge regression pipeline.
    """
    # encoding
    pre = build_preprocessor()

    # model ridge
    ridge = Ridge(alpha=alpha)
    return Pipeline(steps=[("preprocess", pre), ("model", ridge)])


# Lasso Pipe and model
def build_lasso_pipeline(
    alpha: float = 1.0, max_iter: int = MAX_ITER_LASSO
) -> Pipeline:
    """Build a Lasso regression pipeline."""
    # encoding
    pre = build_preprocessor()

    # model lasso
    lasso = Lasso(alpha=alpha, max_iter=max_iter)
    return Pipeline(steps=[("preprocess", pre), ("model", lasso)])


# randomforest Pipe and model
def build_rf_pipeline(
    n_estimators: int = N_ESTIMATORS_RF,
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
) -> Pipeline:
    """Build a RandomForest regression pipeline."""
    # encoding
    pre = build_preprocessor()

    # model rf
    rf = RandomForestRegressor(
        n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs
    )
    return Pipeline(steps=[("preprocess", pre), ("model", rf)])
