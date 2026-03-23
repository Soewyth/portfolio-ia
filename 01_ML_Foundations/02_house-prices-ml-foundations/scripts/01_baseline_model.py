from __future__ import annotations
from datetime import datetime  # for future compatibility
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import KFold
from pathlib import Path

# ============== IMPORTS FROM house_prices_ml_foundations ==============
# data imports
from house_prices_ml_foundations.data.load import load_train_test
from house_prices_ml_foundations.features.build import make_features
from house_prices_ml_foundations.data.split import make_train_valid_split
from house_prices_ml_foundations.models.baseline import build_ridge_pipeline
from house_prices_ml_foundations.evaluation.cv import cross_validate_model
from house_prices_ml_foundations.evaluation.reporting import save_report_json

# config imports
from house_prices_ml_foundations.config import TEST_SIZE, RANDOM_STATE, N_SPLITS_CV


def main():
    """Main function to run the baseline model."""
    scripts_dir = Path(__file__).resolve().parent  #  scripts directory
    root_dir = scripts_dir.parent  # root directory

    # ============== CONFIG PATHS ==============

    # Define output directories,
    outputs_path = root_dir / "outputs"
    figures_path = outputs_path / "figures"
    reports_path = outputs_path / "reports"

    # creating paths
    outputs_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)

    # run metadata
    run_time = datetime.now()
    run_time_str = run_time.strftime("%Y%m%d_%H%M%S")  # for file names

    # Define report paths
    json_path = reports_path / f"metrics_{run_time_str}.json"

    # Model parameters
    test_size = TEST_SIZE
    random_state = RANDOM_STATE

    # Load data
    print(f"=== Loading data from : {root_dir}... ===")
    train_df, _ = load_train_test(root_dir)
    # Prepare features and target
    print("=== Preparing features and target ===")
    X, y = make_features(train_df)
    # Split data into training and validation sets
    print("=== Splitting data into training and validation sets ===")
    X_train, X_valid, y_train, y_valid = make_train_valid_split(
        X, y, test_size, random_state
    )

    print(
        f"Data shape after split -> X_train : {X_train.shape}, X_valid : {X_valid.shape}, y_train : {y_train.shape}, y_valid : {y_valid.shape}"
    )
    print(
        f" y mean in train set : {y_train.mean():.3f} and in validation set : {y_valid.mean():.3f}"
    )

    pipe_ridge = build_ridge_pipeline(alpha=1.0)  # preprocess and model app
    pipe_ridge.fit(X_train, y_train)  # fit datas

    y_pred = pipe_ridge.predict(X_valid)  # predict to x_valid

    # holdout metrics
    mae = mean_absolute_error(y_valid, y_pred)
    rmse = root_mean_squared_error(y_valid, y_pred)
    r2 = r2_score(y_valid, y_pred)

    print("=== Metrics accuracy of Ridge model ===")
    print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}\n")

    # === Cross-VAlidation ===
    print(" === Cross Validation === ")
    cv = KFold(n_splits=N_SPLITS_CV, random_state=RANDOM_STATE, shuffle=True)
    pipe_cv = build_ridge_pipeline(alpha=1.0)

    cv_results = cross_validate_model(
        pipe_cv, X, y, cv=cv, scoring="neg_root_mean_squared_error"
    )

    rmse_mean = cv_results["root_mean_squared_error"]["mean"]
    rmse_std = cv_results["root_mean_squared_error"]["std"]

    print(f"CV_RMSE_MEAN : {rmse_mean:.2f} | CV_RMSE_STD : {rmse_std:.2f}")

    # === REPORT JSON ===
    payload = {
        "run_time": run_time_str,  # cohérence avec la consigne
        "random_state": random_state,
        "test_size": test_size,
        "n_splits_cv": N_SPLITS_CV,  # cohérence
        "features": list(X.columns),
        "holdout": {
            "mae": round(float(mae), 4),  # ← conversion + arrondi
            "rmse": round(float(rmse), 4),
            "r2": round(float(r2), 4),
        },
        "cv": {  # minuscule pour cohérence
            "rmse_mean": round(float(rmse_mean), 4),
            "rmse_std": round(float(rmse_std), 4),
        },
    }

    save_report_json(json_path, payload)
    print(f" === Report JSON saved at :\n {json_path} === ")


if __name__ == "__main__":
    main()
