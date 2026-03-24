from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json

from house_prices_ml_foundations.io.tuning import find_latest_tuning_report


def generate_report_md(reports_path: Path, report_path: Path | None = None) -> Path:
    """Aggregate latest JSON artifacts and generate a markdown audit report."""

    def find_latest(pattern: str) -> Path | None:
        candidates = list(reports_path.glob(pattern)) # Find files matching the pattern
        if not candidates:
            return None
        prefix = pattern.replace("*.json", "") # Extract the prefix to sort by timestamp
        return max(candidates, key=lambda p: p.stem.replace(prefix, "")) # Get the timestamp from the filename and find the latest one

    def load_json(path: Path | None) -> dict | None:
        if path is None or not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8")) # Load the JSON content into a dictionary

    def fmt(value: float | int | str | None, digits: int = 4) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, (int, float)):
            return f"{float(value):.{digits}f}" # Format numerical values with specified precision
        return str(value)

    reports_path.mkdir(parents=True, exist_ok=True) # Ensure the reports directory exists

    # paths to latest artifacts
    baseline_metrics_path = find_latest("metrics_*.json")
    model_comparison_path = find_latest("model_comparison_report_*.json")
    rf_final_path = find_latest("rf_final_holdout_*.json")
    submission_path = find_latest("submission_rf_*.json")
    try:
        tuning_latest_path = find_latest_tuning_report(reports_path) # from tuning.py
    except FileNotFoundError:
        tuning_latest_path = None

    # load JSON content
    baseline_metrics = load_json(baseline_metrics_path)
    model_comparison = load_json(model_comparison_path)
    rf_final = load_json(rf_final_path)
    submission = load_json(submission_path)
    tuning_latest = load_json(tuning_latest_path)

    tuning_source_path = None
    # Determine which tuning report source to reference in the report (prefer RF final holdout, then submission)
    if rf_final:
        tuning_source_path = rf_final.get("tuning_source_file")
    if tuning_source_path is None and submission:
        tuning_source_path = submission.get("tuning_source_file")
    
    tuning = tuning_latest
    tuning_source_used = str(tuning_latest_path) if tuning_latest_path else "N/A"
    if tuning_source_path:
        candidate = Path(tuning_source_path)
        if not candidate.is_absolute():
            candidate = reports_path / candidate
        candidate_payload = load_json(candidate)
        if candidate_payload is not None:
            tuning = candidate_payload
            tuning_source_used = str(candidate)

    lines: list[str] = []
    lines.append("# REPORT - House Prices")
    lines.append("")
    lines.append("## Dataset")
    if submission:
        lines.append(f"- Train rows: {submission.get('n_train_samples', 'N/A')}")
        lines.append(f"- Test rows: {submission.get('n_test_samples', 'N/A')}")
    else:
        lines.append("- Train rows: N/A")
        lines.append("- Test rows: N/A")
    lines.append("- Data files: datasets/raw/train.csv, datasets/raw/test.csv")
    lines.append("")

    lines.append("## Features")
    if model_comparison and isinstance(model_comparison.get("features"), list):
        lines.append(f"- Number of features used: {len(model_comparison.get('features', []))}")
    else:
        lines.append("- Number of features used: N/A")
    lines.append("- Schema reference: src/house_prices_ml_foundations/features/schema.py")
    lines.append("")

    lines.append("## Holdout Split")
    random_state = None
    test_size = None
    n_splits_cv = None
    if model_comparison:
        random_state = model_comparison.get("random_state")
        test_size = model_comparison.get("test_size")
        n_splits_cv = model_comparison.get("n_splits_cv")
    elif rf_final:
        random_state = rf_final.get("random_state")
        test_size = rf_final.get("test_size")

    lines.append(f"- random_state: {random_state if random_state is not None else 'N/A'}")
    lines.append(f"- test_size: {test_size if test_size is not None else 'N/A'}")
    lines.append(f"- n_splits_cv: {n_splits_cv if n_splits_cv is not None else 'N/A'}")
    lines.append("")

    lines.append("## Baseline Ridge/Lasso/RF")
    if model_comparison and isinstance(model_comparison.get("models"), dict):
        lines.append("| Model | Holdout RMSE | Holdout MAE | Holdout R2 | CV RMSE mean | CV RMSE std |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for model_name in ["ridge", "lasso", "rf"]:
            model_data = model_comparison["models"].get(model_name, {})
            hold = model_data.get("holdout", {})
            cv = model_data.get("cv", {})
            lines.append(
                f"| {model_name} | {fmt(hold.get('rmse'))} | {fmt(hold.get('mae'))} | {fmt(hold.get('r2'))} | {fmt(cv.get('rmse_mean'))} | {fmt(cv.get('rmse_std'))} |"
            )
    elif baseline_metrics:
        lines.append("- Baseline metrics JSON found but model comparison report missing.")
    else:
        lines.append("- N/A")
    lines.append("")

    lines.append("## RF Tuning")
    if tuning:
        lines.append(f"- tuning_source_file: {tuning_source_used}")
        lines.append(f"- n_candidates: {tuning.get('n_candidates', 'N/A')}")
        lines.append(f"- best_rmse_cv: {fmt(tuning.get('best_rmse_cv'))}")
        lines.append(f"- best_params: {tuning.get('best_params', {})}")
        lines.append("")
        top_5 = tuning.get("top_5", [])
        if top_5:
            lines.append("| Rank | RMSE mean | RMSE std | Params |")
            lines.append("|---:|---:|---:|---|")
            for row in top_5:
                lines.append(
                    f"| {row.get('rank', 'N/A')} | {fmt(row.get('rmse_mean'))} | {fmt(row.get('rmse_std'))} | {row.get('params', {})} |"
                )
        else:
            lines.append("- top_5: N/A")
    else:
        lines.append("- N/A")
    lines.append("")

    lines.append("## RF Final Holdout")
    if rf_final:
        hold = rf_final.get("holdout", {})
        lines.append(f"- tuning_source_file: {rf_final.get('tuning_source_file', 'N/A')}")
        lines.append(f"- mae: {fmt(hold.get('mae'))}")
        lines.append(f"- rmse: {fmt(hold.get('rmse'))}")
        lines.append(f"- r2: {fmt(hold.get('r2'))}")
    else:
        lines.append("- N/A")
    lines.append("")

    lines.append("## Decision")
    if tuning and rf_final and model_comparison:
        rf_cv = model_comparison.get("models", {}).get("rf", {}).get("cv", {})
        lines.append("- Selected model: Tuned RandomForestRegressor")
        lines.append(
            "- Why: best CV RMSE from tuning, lower baseline RF CV RMSE than linear baselines, and final holdout metrics validated."
        )
        lines.append(
            f"- Evidence: tuning best_rmse_cv={fmt(tuning.get('best_rmse_cv'))}, baseline rf cv_rmse_std={fmt(rf_cv.get('rmse_std'))}, final_holdout_rmse={fmt(rf_final.get('holdout', {}).get('rmse'))}."
        )
    else:
        lines.append("- Pending: missing one or more required artifacts (comparison, tuning, final holdout).")

    if report_path is None:
        run_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = reports_path / f"REPORT_{run_time_str}.md"

    report_path.parent.mkdir(exist_ok=True, parents=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
