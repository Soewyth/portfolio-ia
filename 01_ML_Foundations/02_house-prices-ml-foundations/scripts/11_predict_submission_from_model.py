from __future__ import annotations

import pandas as pd

from house_prices_ml_foundations.config.paths import get_paths, get_project_root
from house_prices_ml_foundations.data.load import load_train_test
from house_prices_ml_foundations.features.build import make_features
from house_prices_ml_foundations.io.model_artifacts import load_model
from house_prices_ml_foundations.io.run_id import make_run_id


def main() -> None:
    """Predict submission file from champion model and export it."""

    root_dir = get_project_root()
    paths = get_paths(root_dir)
    run_id = make_run_id(tag="submission_rf_pipe_inference")

    model_path = paths["models"] / "champion.joblib"
    submissions_path = paths["submissions"]

    submission_path = submissions_path / f"{run_id}.csv"

    model = load_model(model_path)

    _, test_df = load_train_test(root_dir)
    X_test = make_features(test_df, return_target=False)

    y_pred = model.predict(X_test)

    if not len(test_df) == len(y_pred):
        raise ValueError(f"Length of test data ({len(test_df)}) does not match length of predictions ({len(y_pred)}).")

    submission_df = pd.DataFrame(
        {
            "Id": test_df["Id"].astype(int),
            "SalePrice": y_pred.astype(float),
        }
    )

    if "Id" not in test_df.columns:
        raise ValueError("Column 'Id' missing in test dataset.")

    submission_df.to_csv(submission_path, index=False)

    print(" === First rows of submission === \n")
    print(f" {submission_df.head(3)} \n")
    print(" === Submission generated from model inference === \n ")
    print(f"Model loaded from: {model_path}\n")
    print(f"Submission file saved to: {submission_path}")


if __name__ == "__main__":
    main()
