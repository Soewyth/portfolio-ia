from __future__ import annotations  # for future compatibility

from sklearn.pipeline import Pipeline

from house_prices_ml_foundations.models.baseline import build_ridge_pipeline, build_lasso_pipeline, build_rf_pipeline


def make_model_registry() -> dict[str, Pipeline]:
    """Return a dict of model pipelines for comparison."""
    return { "bridge" : build_ridge_pipeline(),
            "lasso" : build_lasso_pipeline(),
            "rf": build_rf_pipeline()
    }