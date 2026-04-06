import json

import numpy as np
import pytest

from house_prices_ml_foundations.config.paths import get_project_root
from house_prices_ml_foundations.data.load import load_train_test
from house_prices_ml_foundations.features.build import make_features
from house_prices_ml_foundations.models.champion import build_champion_pipeline


@pytest.fixture
def mini_train_xy():
    root_dir = get_project_root()
    train_df, _ = load_train_test(root_dir)
    sample = train_df.head(20).copy()
    X, y = make_features(sample, return_target=True)
    return X, y


def test_build_champion_pipeline_fit_predict(mini_train_xy):
    X, y = mini_train_xy
    pipe, _ = build_champion_pipeline(None)

    pipe.fit(X, y)
    preds = pipe.predict(X)

    assert len(preds) == len(X)
    assert np.isfinite(preds).all()


def test_build_champion_pipeline_with_tuning(tmp_path, mini_train_xy):
    reports_path = tmp_path / "reports"
    reports_path.mkdir(parents=True, exist_ok=True)
    # Simulate a tuning report with best parameters
    tuning_report = reports_path / "tuning_rf_tuning_rf_20260406_000000.json"
    best_params = {"model__n_estimators": 123, "model__max_depth": 4}
    # Write the tuning report
    with tuning_report.open("w") as f:
        json.dump({"best_params": best_params}, f)
    # Build champion pipeline which read the tuning report and use best_params
    X, y = mini_train_xy
    pipe, source = build_champion_pipeline(reports_path)

    assert pipe.get_params()["model__n_estimators"] == 123
    assert pipe.get_params()["model__max_depth"] == 4

    pipe.fit(X, y)
    preds = pipe.predict(X)

    assert len(preds) == len(X)
    assert source is not None
