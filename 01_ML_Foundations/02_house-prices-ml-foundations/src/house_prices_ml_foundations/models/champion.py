from __future__ import annotations  # for future compatibility

from pathlib import Path

from sklearn.pipeline import Pipeline

from house_prices_ml_foundations.io.tuning import find_latest_tuning_report, load_best_params_from_tuning_json
from house_prices_ml_foundations.models.baseline import build_rf_pipeline


def build_champion_pipeline(reports_path: Path | None = None) -> tuple[Pipeline, str]:
    """Build RF champion pipeline.

    Returns:
        tuple[Pipeline, str]:
            - pipeline: configured Random Forest pipeline
            - champion_source: "tuned" if tuning params were applied,
              otherwise "default"
    """
    rf_pipe = build_rf_pipeline()

    if reports_path is None:
        return rf_pipe, "default"

    try:
        latest_tuning_report = find_latest_tuning_report(reports_path)
        best_params = load_best_params_from_tuning_json(latest_tuning_report)
        rf_pipe.set_params(**best_params)

        return rf_pipe, "tuned"

    except (FileNotFoundError, ValueError, OSError):  # Fallback
        print("No valid tuning artifacts found. Using default RF parameters.")
        return rf_pipe, "default"
