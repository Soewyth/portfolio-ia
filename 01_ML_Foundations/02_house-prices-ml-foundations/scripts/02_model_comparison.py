from __future__ import annotations

from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold

from house_prices_ml_foundations.config.config import N_SPLITS_CV, RANDOM_STATE, TEST_SIZE
from house_prices_ml_foundations.config.paths import get_paths, get_project_root

# ============== IMPORTS FROM house_prices_ml_foundations ==============
from house_prices_ml_foundations.data.load import load_train_test
from house_prices_ml_foundations.data.split import make_train_valid_split
from house_prices_ml_foundations.evaluation.cv import cross_validate_model
from house_prices_ml_foundations.evaluation.reporting import save_report_json
from house_prices_ml_foundations.features.build import make_features
from house_prices_ml_foundations.io.run_id import make_run_id
from house_prices_ml_foundations.models.registry import make_model_registry


def main() -> None:
    """Compare Ridge, Lasso and RandomForest with holdout + CV."""
    root_dir = get_project_root()
    paths = get_paths(root_dir)

    # run metadata
    run_id = make_run_id("model_comparison")
    json_path = paths["reports"] / f"report_{run_id}.json"

    # Model parameters
    test_size = TEST_SIZE
    random_state = RANDOM_STATE
    n_splits_cv = N_SPLITS_CV

    # Load data
    print(f"=== Loading data from : {root_dir}... ===")
    train_df, _ = load_train_test(root_dir)
    # Prepare features and target
    print("=== Preparing features and target ===")
    X, y = make_features(train_df)
    # Split data into training and validation sets
    print("=== Splitting data into training and validation sets ===")
    X_train, X_valid, y_train, y_valid = make_train_valid_split(X, y, test_size, random_state)

    print(f"Data shape after split -> X_train : {X_train.shape}, X_valid : {X_valid.shape}, y_train : {y_train.shape}, y_valid : {y_valid.shape}")
    print(f" y mean in train set : {y_train.mean():.3f} and in validation set : {y_valid.mean():.3f}")

    # === Cross-Validation + models comparisons ===
    print(" === Cross Validation === ")
    cv = KFold(n_splits=n_splits_cv, random_state=random_state, shuffle=True)
    registry = make_model_registry()

    results = {}

    for name, pipe in registry.items():
        # Holdout evaluation
        holdout_pipe = clone(pipe)
        holdout_pipe.fit(X_train, y_train)
        y_pred = holdout_pipe.predict(X_valid)

        mae = mean_absolute_error(y_valid, y_pred)
        rmse = root_mean_squared_error(y_valid, y_pred)
        r2 = r2_score(y_valid, y_pred)

        # Cross-validation evaluation

        cv_results = cross_validate_model(
            model=clone(pipe),
            X=X,
            y=y,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            # clone for dont fit the model again on the whole data
        )
        rmse_mean = cv_results["root_mean_squared_error"]["mean"]
        rmse_std = cv_results["root_mean_squared_error"]["std"]

        results[name] = {
            "holdout": {"mae": round(float(mae), 4), "rmse": round(float(rmse), 4), "r2": round(float(r2), 4)},
            "cv": {"rmse_mean": round(float(rmse_mean), 4), "rmse_std": round(float(rmse_std), 4)},
        }

        print(f"{name}: MAE={mae:.2f} | RMSE={rmse:.2f} | R2={r2:.4f} | CV_RMSE={rmse_mean:.2f} +/- {rmse_std:.2f}")

    # === REPORT JSON ===
    payload = {
        "run_time": run_id,
        "random_state": random_state,
        "test_size": test_size,
        "n_splits_cv": n_splits_cv,
        "features": list(X.columns),
        "models": results,
    }

    save_report_json(json_path, payload)
    print(f" === Report JSON saved at :\n {json_path} === ")


if __name__ == "__main__":
    main()
