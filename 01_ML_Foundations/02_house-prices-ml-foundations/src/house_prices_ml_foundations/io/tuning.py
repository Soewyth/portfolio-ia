from __future__ import annotations  # for future compatibility

import json
from pathlib import Path

from house_prices_ml_foundations.models.baseline import build_rf_pipeline


def find_latest_tuning_report(reports_path: Path) -> Path:
    """Find the latest tuning report JSON file in the reports directory."""
    tuning_reports = list(reports_path.glob("tuning_rf_*.json"))  # Look for tuning report JSON files
    if not tuning_reports:
        raise FileNotFoundError(f"No tuning report found in {reports_path}")
    latest_report = max(tuning_reports, key=lambda p: p.stem.replace("tuning_rf_", ""))  # Get the timestamp from the filename and find the latest one
    return latest_report


def load_best_params_from_tuning_json(tuning_json_path: Path) -> dict:
    """Load best hyperparameters from the latest tuning report JSON."""
    if not tuning_json_path.exists():
        raise FileNotFoundError(f"Tuning report JSON not found: {tuning_json_path}")

    with open(tuning_json_path, "r") as f:
        report = json.load(f)  # Load the JSON content into a dictionary

    best_params = report.get("best_params")  # Extract the 'best_params' field from the report
    if not best_params:
        raise ValueError(f"'best_params' missing or empty in tuning report: {tuning_json_path}")
    rf_pipeline = build_rf_pipeline()
    known = set(rf_pipeline.get_params().keys())  # Get the valid parameter keys from the RF pipeline
    unknown = set(best_params) - known  # Check if any keys in best_params are not in the RF pipeline's parameters
    if unknown:
        raise ValueError(f"best_params contains keys not found in RF pipeline: {unknown}\nKnown keys (sample): {sorted(known)[:10]}")

    return best_params
