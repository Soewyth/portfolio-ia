from __future__ import annotations  # for future compatibility
from pathlib import Path  # for handling file paths
from datetime import datetime
import json
import textwrap  # for saving report as JSON


# Function to write the report
def write_reports_baseline(
    report_path: Path,
    run_time: datetime,
    accuracy: float,
    f1: float,
    roc_auc: float,
    cm,
    random_state: int,
    test_size: float,
    features: list[str],
    mean_cv_score_f1: float,
    std_cv_score_f1: float,
    mean_cv_score_auc: float,
    std_cv_score_auc: float,
) -> None:
    """Write model evaluation metrics to a report file.
    Args:
        report_path (Path): Path to save the report file.
        run_time (datetime): Timestamp of the model run.
        accuracy (float): Accuracy score.
        f1 (float): F1 score.
        roc_auc (float): ROC AUC score.
        cm: Confusion matrix.
        random_state (int): Random state used for splitting data.
        test_size (int): Test size used for splitting data.
        features (list[str]): List of feature names used in the model.
    """
    # report.md
    features_md = "\n".join(f"- {f}" for f in features)
    report = textwrap.dedent(
        f"""\
# Titanic Baseline Logistic Regression Model Report
**Run Time:** {run_time.strftime("%Y-%m-%d %H:%M:%S")}

## Features Used :
{features_md}

## Data Split Parameters
- **Random State:** `{random_state}`
- **Test Size:** `{test_size}`

## Models : 
### Pipeline : 
- **Preprocessing:** 
    - SimpleImputer (median for numerical, most_frequent for categorical)
    - OneHotEncoder (handle_unknown='ignore')
- **Classifier:** LogisticRegression (random_state={random_state})

## Model Evaluation Metrics
- **Accuracy:** `{accuracy:.4f}`
- **F1 Score:** `{f1:.4f}`
- **ROC AUC Score:** `{roc_auc:.4f}`

## Confusion Matrix : 
Format : `[[TN,FP],[FN,TP]]` \n
Saved at : `outputs/figures/confusion_matrix.png`
```
{cm}
```

## Cross-validation
- F1 Score (5-fold Stratified CV): Mean = `{mean_cv_score_f1:.4f}`, Std = `{std_cv_score_f1:.4f}`
- ROC AUC Score (5-fold Stratified CV): Mean = `{mean_cv_score_auc:.4f}`, Std = `{std_cv_score_auc:.4f}`

"""
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def save_report_json(path: Path, payload: dict) -> None:
    """Save model evaluation metrics to a JSON report file.
    Args:
        path (Path): Path to save the report file.
        payload (dict): Dictionary containing model evaluation metrics.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=4, ensure_ascii=False), encoding="utf-8")
    

