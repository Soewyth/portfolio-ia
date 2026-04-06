from __future__ import annotations

import mlflow
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from house_prices_ml_foundations.config.config import (
    RANDOM_STATE,
    TEST_SIZE,
)
from house_prices_ml_foundations.config.mlflow_config import MLFLOW_TRACKING_URI
from house_prices_ml_foundations.config.paths import get_paths, get_project_root, latest_file
from house_prices_ml_foundations.data.load import load_train_test
from house_prices_ml_foundations.data.split import make_train_valid_split
from house_prices_ml_foundations.features.build import make_features
from house_prices_ml_foundations.io.run_id import make_run_id
from house_prices_ml_foundations.models.champion import build_champion_pipeline


def main() -> None:
    """Train Champion Model and log metrics to MLflow."""

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("houses-price")
    print(f"TRACKING URI MLFLOW : {mlflow.get_tracking_uri()}")

    run_id = make_run_id()
    root_dir = get_project_root()
    paths = get_paths(root_dir)
    figure_path = paths["figures"]
    report_path = paths["reports"]
    models_path = paths["models"]

    # Load and make features
    train_df, _ = load_train_test(root_dir=root_dir)
    X, y = make_features(df=train_df, return_target=True)
    X_train, X_valid, y_train, y_valid = make_train_valid_split(X=X, y=y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    # Build RF pipeline and fit
    pipe, champion_source = build_champion_pipeline(paths["reports"])
    pipe.fit(X_train, y_train)

    # predict
    y_pred = pipe.predict(X_valid)

    # metrics
    holdout_mae = mean_absolute_error(y_valid, y_pred)
    holdout_rmse = root_mean_squared_error(y_valid, y_pred)
    holdout_r2 = r2_score(y_valid, y_pred)

    # === Logs ===
    with mlflow.start_run(run_name=run_id) as run:
        # metadata
        mlflow.log_param("run_id", run_id)
        mlflow.log_param("champion_source", champion_source)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("test_size", TEST_SIZE)

        # Params
        model_params = {k: v for k, v in pipe.get_params().items() if k.startswith("model__")}
        for key, value in model_params.items():
            mlflow.log_param(key, value)

        # Metrics
        mlflow.log_metric("holdout_mae", float(holdout_mae))
        mlflow.log_metric("holdout_rmse", float(holdout_rmse))
        mlflow.log_metric("holdout_r2", float(holdout_r2))

        # Artifacts
        latest_report = latest_file(report_path, "REPORT_*.md")
        if latest_report is not None and latest_report.exists():
            mlflow.log_artifact(str(latest_report))
        else:
            print("[MLflow] No report artifact found to log.")

        latest_residuals_hist = latest_file(figure_path, "residuals_hist*.png")
        if latest_residuals_hist is not None and latest_residuals_hist.exists():
            mlflow.log_artifact(str(latest_residuals_hist))
        else:
            print("[MLflow] No residuals histogram artifact found to log.")

        latest_ytrue_vs_ypred = latest_file(figure_path, "ytrue_vs_ypred*.png")
        if latest_ytrue_vs_ypred is not None and latest_ytrue_vs_ypred.exists():
            mlflow.log_artifact(str(latest_ytrue_vs_ypred))
        else:
            print("[MLflow] No ytrue vs ypred artifact found to log.")

        latest_abs_error_vs_ytrue = latest_file(figure_path, "abs_error_vs_ytrue*.png")
        if latest_abs_error_vs_ytrue is not None and latest_abs_error_vs_ytrue.exists():
            mlflow.log_artifact(str(latest_abs_error_vs_ytrue))
        else:
            print("[MLflow] No absolute error vs ytrue artifact found to log.")

        latest_error_summary = latest_file(report_path, "error_analysis_*_summary.json")
        if latest_error_summary is not None and latest_error_summary.exists():
            mlflow.log_artifact(str(latest_error_summary))
        else:
            print("[MLflow] No error summary artifact found to log.")

        # Model
        mlflow.sklearn.log_model(pipe, artifact_path="model")

        champion_joblib = models_path / "champion.joblib"
        if champion_joblib.exists():
            mlflow.log_artifact(str(champion_joblib))
            print(f"[MLflow] model artifact logged: {champion_joblib.name}")
        else:
            print("[MLflow] champion.joblib not found locally, skipped.")

        print("\n=== TRAIN + LOG MLFLOW DONE ===")
        print(f"run_name      : {run_id}")
        print(f"mlflow_run_id : {run.info.run_id}")
        print(f"champion_src  : {champion_source}")
        print(f"holdout_mae   : {holdout_mae:.6f}")
        print(f"holdout_rmse  : {holdout_rmse:.6f}")
        print(f"holdout_r2    : {holdout_r2:.6f}")


if __name__ == "__main__":
    main()
