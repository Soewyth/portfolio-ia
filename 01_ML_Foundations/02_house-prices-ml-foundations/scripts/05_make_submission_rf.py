from __future__ import annotations

import pandas as pd

from house_prices_ml_foundations.config.paths import get_paths, get_project_root
from house_prices_ml_foundations.data.load import load_train_test
from house_prices_ml_foundations.evaluation.reporting import save_report_json
from house_prices_ml_foundations.features.build import make_features
from house_prices_ml_foundations.io.run_id import make_run_id
from house_prices_ml_foundations.models.champion import build_champion_pipeline


def main() -> None:
    """Train RF on full training data and generate Kaggle submission."""
    root_dir = get_project_root()
    paths = get_paths(root_dir)

    run_id = make_run_id("submission_rf_")
    submission_path = paths["submissions"] / f"{run_id}.csv"
    json_path = paths["reports"] / f"{run_id}.json"

    train_df, test_df = load_train_test(root_dir)
    X_train, y_train = make_features(train_df)
    X_test = make_features(test_df, return_target=False)

    if "Id" not in test_df.columns:
        raise ValueError("Column 'Id' missing in test dataset.")

    rf_pipeline, champion_source = build_champion_pipeline(reports_path=paths["reports"])
    rf_pipeline.fit(X_train, y_train)
    y_pred = rf_pipeline.predict(X_test)

    submission_df = pd.DataFrame(
        {
            "Id": test_df["Id"].astype(int),
            "SalePrice": y_pred,
        }
    )
    submission_df.to_csv(submission_path, index=False)

    params = rf_pipeline.get_params()
    payload = {
        "run_time": run_id,
        "champion_source": champion_source,
        "champion_params": {k: v for k, v in params.items() if k.startswith("model__")},  # Filter params to include only those related to the model
        "n_train_samples": int(len(train_df)),
        "n_test_samples": int(len(test_df)),
        "submission_file": str(submission_path),
    }
    save_report_json(json_path, payload)

    print("=== RF submission generated ===")
    print(f"Champion source: {champion_source}")
    print(f"Submission saved: {submission_path}")
    print(f"Run report saved: {json_path}")


if __name__ == "__main__":
    main()
