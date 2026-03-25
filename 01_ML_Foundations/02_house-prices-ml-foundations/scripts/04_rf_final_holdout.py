from __future__ import annotations

from datetime import datetime
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

from house_prices_ml_foundations.config import RANDOM_STATE, TEST_SIZE
from house_prices_ml_foundations.data.split import make_train_valid_split
from house_prices_ml_foundations.data.load import load_train_test
from house_prices_ml_foundations.evaluation.reporting import save_report_json
from house_prices_ml_foundations.features.build import make_features
from house_prices_ml_foundations.models.champion import build_champion_pipeline


def main() -> None:
    """Evaluate RF with best params from tuning on a fixed holdout split."""
    scripts_dir = Path(__file__).resolve().parent
    root_dir = scripts_dir.parent
    # ============== CONFIG PATHS ==============

    # Define output directories,
    outputs_path = root_dir / "outputs"
    figures_path = outputs_path / "figures"
    reports_path = outputs_path / "reports"

    # creating paths
    outputs_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)

    random_state = RANDOM_STATE
    test_size = TEST_SIZE

    # run metadata
    run_time = datetime.now()
    run_time_str = run_time.strftime("%Y%m%d_%H%M%S")  # for file names

    json_path = reports_path / f"rf_final_holdout_{run_time_str}.json"

    train_df, _ = load_train_test(root_dir)
    X, y = make_features(train_df)

    X_train, X_valid, y_train, y_valid = make_train_valid_split(
        X, y, test_size=test_size, random_state=random_state
    )

    rf_pipeline, champion_source = build_champion_pipeline(reports_path=reports_path)

    # Load best hyperparameters from the latest tuning report JSON
    print(f"Champion source: {champion_source}")
    rf_pipeline.fit(X_train, y_train)
    y_pred = rf_pipeline.predict(X_valid)

    mae = mean_absolute_error(y_valid, y_pred)
    rmse = root_mean_squared_error(y_valid, y_pred)
    r2 = r2_score(y_valid, y_pred)

    holdout_results = {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4),
    }
    params = rf_pipeline.get_params()
    payload = {
        "run_time": run_time_str,
        "random_state": random_state,
        "test_size": test_size,
        "champion_source": champion_source,
        "champion_params": {
            k: v for k, v in params.items() if k.startswith("model__")
        },  # Filter params to include only those related to the model
        "holdout": holdout_results,
    }

    save_report_json(json_path, payload)

    print("=== RF final holdout ===")
    print(
        f"MAE={holdout_results['mae']:.2f} | RMSE={holdout_results['rmse']:.2f} | R2={holdout_results['r2']:.4f}"
    )
    print(f"Tuning source: {champion_source}")
    print(f"JSON report saved: {json_path}")


if __name__ == "__main__":
    main()
