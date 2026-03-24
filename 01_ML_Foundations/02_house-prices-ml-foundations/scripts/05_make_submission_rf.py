from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from house_prices_ml_foundations.data.load import load_train_test
from house_prices_ml_foundations.evaluation.reporting import save_report_json
from house_prices_ml_foundations.features.build import make_features
from house_prices_ml_foundations.io.tuning import (
    find_latest_tuning_report,
    load_best_params_from_tuning_json,
)
from house_prices_ml_foundations.models.baseline import build_rf_pipeline


def main() -> None:
    """Train RF on full training data and generate Kaggle submission."""
    scripts_dir = Path(__file__).resolve().parent
    root_dir = scripts_dir.parent

    outputs_path = root_dir / "outputs"
    reports_path = outputs_path / "reports"
    submissions_path = outputs_path / "submissions"

    reports_path.mkdir(parents=True, exist_ok=True)
    submissions_path.mkdir(parents=True, exist_ok=True)

    run_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    submission_path = submissions_path / f"submission_rf_{run_time_str}.csv"
    json_path = reports_path / f"submission_rf_{run_time_str}.json"

    latest_tuning_report = find_latest_tuning_report(reports_path)
    best_params = load_best_params_from_tuning_json(latest_tuning_report)

    train_df, test_df = load_train_test(root_dir)
    X_train, y_train = make_features(train_df)
    X_test = make_features(test_df, return_target=False)

    if "Id" not in test_df.columns:
        raise ValueError("Column 'Id' missing in test dataset.")

    rf_pipeline = build_rf_pipeline()
    rf_pipeline.set_params(**best_params)
    rf_pipeline.fit(X_train, y_train)
    y_pred = rf_pipeline.predict(X_test)

    submission_df = pd.DataFrame(
        {
            "Id": test_df["Id"].astype(int),
            "SalePrice": y_pred,
        }
    )
    submission_df.to_csv(submission_path, index=False)

    payload = {
        "run_time": run_time_str,
        "tuning_source_file": str(latest_tuning_report),
        "best_params": best_params,
        "n_train_samples": int(len(train_df)),
        "n_test_samples": int(len(test_df)),
        "submission_file": str(submission_path),
    }
    save_report_json(json_path, payload)

    print("=== RF submission generated ===")
    print(f"Tuning source: {latest_tuning_report.name}")
    print(f"Submission saved: {submission_path}")
    print(f"Run report saved: {json_path}")


if __name__ == "__main__":
    main()
