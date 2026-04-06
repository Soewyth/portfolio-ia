from __future__ import annotations

from sklearn.model_selection import GridSearchCV, KFold

from house_prices_ml_foundations.config.config import N_SPLITS_CV, RANDOM_STATE
from house_prices_ml_foundations.config.paths import get_paths, get_project_root
from house_prices_ml_foundations.data.load import load_train_test
from house_prices_ml_foundations.evaluation.reporting import save_report_json
from house_prices_ml_foundations.features.build import make_features
from house_prices_ml_foundations.io.run_id import make_run_id
from house_prices_ml_foundations.models.registry import make_model_registry


def main() -> None:
    """Tune RandomForest hyperparameters with GridSearchCV."""
    root_dir = get_project_root()
    paths = get_paths(root_dir)

    run_id = make_run_id("tuning_rf_")
    json_path = paths["reports"] / f"{run_id}.json"

    train_df, _ = load_train_test(root_dir)
    X, y = make_features(train_df)

    registry = make_model_registry()
    rf_pipeline = registry["rf"]

    cv = KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)

    param_grid = {
        "model__n_estimators": [100, 300, 500],
        "model__max_depth": [None, 10, 30, 50],
        "model__min_samples_split": [2, 10, 20],
        "model__min_samples_leaf": [1, 5, 10],
        "model__max_features": ["sqrt", 0.5, 0.75, 1.0],
    }

    search = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
        return_train_score=False,
    )

    search.fit(X, y)

    best_rmse_cv = -float(search.best_score_)
    n_candidates = len(search.cv_results_["params"])

    ranked = sorted(
        zip(
            search.cv_results_["params"],
            search.cv_results_["mean_test_score"],
            search.cv_results_["std_test_score"],
        ),
        key=lambda row: row[1],
        reverse=True,
    )

    top_k = []
    for rank, (params, mean_score, std_score) in enumerate(ranked[:5], start=1):
        top_k.append(
            {
                "rank": rank,
                "params": params,
                "rmse_mean": round(float(-mean_score), 4),
                "rmse_std": round(float(std_score), 4),
            }
        )

    payload = {
        "run_time": run_id,
        "random_state": RANDOM_STATE,
        "n_splits_cv": N_SPLITS_CV,
        "scoring": "neg_root_mean_squared_error",
        "n_candidates": n_candidates,
        "best_params": search.best_params_,
        "best_rmse_cv": round(best_rmse_cv, 4),
        "best_index": int(search.best_index_),
        "top_5": top_k,
    }

    save_report_json(json_path, payload)

    print("=== RF tuning done ===")
    print(f"Best params: {search.best_params_}")
    print(f"Best CV RMSE: {best_rmse_cv:.4f}")
    print(f"JSON report saved: {json_path}")


if __name__ == "__main__":
    main()
