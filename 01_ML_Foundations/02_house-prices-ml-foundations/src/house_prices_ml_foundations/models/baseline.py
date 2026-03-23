from __future__ import annotations  # for future compatibility
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from house_prices_ml_foundations.features.schema import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ORDINAL_FEATURES,
)
from house_prices_ml_foundations.features.preprocess import build_preprocessor

def build_ridge_pipeline(alpha=1.0) -> Pipeline:
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
    ridge_pipeline = Pipeline(
        steps= [("preprocess", pre), ("model", ridge)
        ]
    )

    return ridge_pipeline
        
