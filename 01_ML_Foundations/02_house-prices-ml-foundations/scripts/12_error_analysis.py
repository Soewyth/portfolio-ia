from __future__ import annotations

import numpy as np
import pandas as pd

from house_prices_ml_foundations.config.config import RANDOM_STATE, TEST_SIZE
from house_prices_ml_foundations.config.paths import get_paths, get_project_root
from house_prices_ml_foundations.data.load import load_train_test
from house_prices_ml_foundations.data.split import make_train_valid_split
from house_prices_ml_foundations.evaluation.error_analysis import (
    build_error_analysis_frame,
    build_error_analysis_summary,
    save_error_analysis_csv,
)
from house_prices_ml_foundations.evaluation.reporting import save_report_json
from house_prices_ml_foundations.features.build import make_features
from house_prices_ml_foundations.io.run_id import make_run_id
from house_prices_ml_foundations.models.champion import build_champion_pipeline


def main() -> None:
    """Evaluate the champion model on the holdout set and export error analysis files."""
    root_dir = get_project_root()
    paths = get_paths(root_dir)
    run_id = make_run_id(tag="rf_pipe")

    train_df, _ = load_train_test(root_dir)
    X, y = make_features(train_df)
    X_train, X_valid, y_train, y_valid = make_train_valid_split(X, y)

    pipe, champion_source = build_champion_pipeline(paths["reports"])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_valid)

    rmse = np.sqrt(((y_valid - y_pred) ** 2).mean())
    mae = abs(y_valid - y_pred).mean()
    error_analysis_df = build_error_analysis_frame(X_valid=X_valid, y_valid=y_valid, y_pred=y_pred, raw_df=train_df)
    error_analysis_csv_path = paths["reports"] / f"error_analysis_{run_id}.csv"
    error_analysis_json_path = paths["reports"] / f"error_analysis_{run_id}.json"
    error_analysis_json_summary_path = paths["reports"] / f"error_analysis_{run_id}_summary.json"

    error_analysis_sum = build_error_analysis_summary(error_analysis_df)
    save_report_json(error_analysis_json_summary_path, error_analysis_sum)
    save_error_analysis_csv(error_analysis_df, error_analysis_csv_path)

    payload = {
        "run_time": run_id,
        "champion_source": champion_source,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "error_analysis_csv": str(error_analysis_csv_path),
    }
    save_report_json(error_analysis_json_path, payload)

    print(" === Error analysis completed === \n")
    print(f"RMSE on holdout set: {rmse:.2f}")
    print(f"MAE on holdout set: {mae:.2f}")
    print("Top 3 errors preview:")
    print(error_analysis_df.sort_values("abs_error", ascending=False).head(3))
    print(f"Champion source: {champion_source}")
    print(f"Error analysis JSON saved to: {error_analysis_json_path}")
    print(f"Error analysis CSV saved to: {error_analysis_csv_path}")

    print("\nTop 10 Neighborhoods by mean absolute error:")
    top_10_neighborhoods = error_analysis_df.groupby("Neighborhood")["abs_error"].mean().sort_values(ascending=False).head(10)
    print(top_10_neighborhoods)
    print("\nTop 10 OverallQual by mean absolute error:")
    top_10_overallqual = error_analysis_df.groupby("OverallQual")["abs_error"].mean().sort_values(ascending=False).head(10)
    print(top_10_overallqual)

    print("\nMAE by bins of GrLivArea: (10 bins)")
    error_analysis_df["GrLivArea_bin"] = pd.qcut(error_analysis_df["GrLivArea"], q=10)
    print(error_analysis_df.groupby("GrLivArea_bin")["abs_error"].mean())

    print("\n % of large errors (>100k) per Neighborhood:")
    large_errors_by_neighborhood = (
        error_analysis_df.groupby("Neighborhood")["abs_error"].apply(lambda x: (x > 100000).mean()).sort_values(ascending=False)
    )
    large_errors_by_neighborhood = large_errors_by_neighborhood[large_errors_by_neighborhood > 0]
    print(large_errors_by_neighborhood)


if __name__ == "__main__":
    main()
