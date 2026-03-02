from __future__ import annotations

from pathlib import Path
from datetime import datetime

from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

# ============== IMPORTS FROM titanic_ml_foundations ==============
# config imports
from titanic_ml_foundations.config import (
    RANDOM_STATE,
    TEST_SIZE,
    N_SPLITS_CV,
    OUTPUTS_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
)

# data imports
from titanic_ml_foundations.data.load import load_train_test
from titanic_ml_foundations.data.split import make_train_valid_split

# features imports
from titanic_ml_foundations.features.build import make_X_y

# model imports
from titanic_ml_foundations.models.registry import make_model_registry

# evaluations imports
from titanic_ml_foundations.evaluation.plots import save_confusion_matrix
from titanic_ml_foundations.evaluation.cv import cross_validate_model
from titanic_ml_foundations.evaluation.reporting import save_report_json
from titanic_ml_foundations.evaluation.reporting import write_report_model_comparison


def main() -> None:
    """Train and compare multiple models on Titanic data."""

    scripts_dir = Path(__file__).resolve().parent  #  scripts directory
    root_dir = scripts_dir.parent  # root directory

    # ============== CONFIG PATHS ==============

    # Define output directories,
    outputs_path = root_dir / OUTPUTS_DIR
    figures_path = outputs_path / FIGURES_DIR
    reports_path = outputs_path / REPORTS_DIR

    # creating paths
    outputs_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)

    # run metadata
    run_time = datetime.now()
    run_time_str = run_time.strftime("%Y%m%d_%H%M%S")  # for file names
    ts_pct = int(TEST_SIZE * 100)  # test size percentage
    run_tag = f"rs{RANDOM_STATE}_ts{ts_pct}_cv{N_SPLITS_CV}"

    # Define report paths
    report_md_path = reports_path / f"comparison_report_{run_tag}.md"
    json_path = reports_path / f"metrics_{run_tag}_{run_time_str}.json"

    # ========== data ==========
    train_df, _ = load_train_test(root_dir)
    X, y = make_X_y(train_df)

    # ========== split fixed ==========
    X_train, X_valid, y_train, y_valid = make_train_valid_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)

    # ========== models ==========
    registry = make_model_registry()
    results: dict[str, dict] = {}

    for model_name, model_pipeline in registry.items():
        # define path for confusion matrix
        cm_path = figures_path / f"confusion_matrix_{model_name}_{run_tag}.png"
        # fit model
        model_pipeline.fit(X_train, y_train)

        # predict
        y_predict = model_pipeline.predict(X_valid)
        y_proba = model_pipeline.predict_proba(X_valid)[:, 1]

        # ========== Metrics ==========
        holdout_acc = accuracy_score(y_valid, y_predict)
        holdout_f1 = f1_score(y_valid, y_predict)
        holdout_roc_auc = roc_auc_score(y_valid, y_proba)
        holdout_cm = confusion_matrix(y_valid, y_predict)

        # ========== cross-validation (unfitted pipe and full data) ==========
        scores = cross_validate_model(
            model=clone(model_pipeline),
            X=X,
            y=y,
            cv=cv,
            scoring=["f1", "roc_auc"],
        )

        cv_f1_mean = scores["f1"]["mean"]
        cv_f1_std = scores["f1"]["std"]
        cv_roc_auc_mean = scores["roc_auc"]["mean"]
        cv_roc_auc_std = scores["roc_auc"]["std"]
        # results
        results[model_name] = {
            "holdout": {
                "accuracy": round(float(holdout_acc), 4),
                "f1_score": round(float(holdout_f1), 4),
                "roc_auc": round(float(holdout_roc_auc), 4),
                "confusion_matrix": holdout_cm.tolist(),
            },
            "cv": {
                "f1_score": {
                    "mean": round(float(cv_f1_mean), 4),
                    "std": round(float(cv_f1_std), 4),
                },
                "roc_auc": {
                    "mean": round(float(cv_roc_auc_mean), 4),
                    "std": round(float(cv_roc_auc_std), 4),
                },
            },
        }
    # ========== artifacts saving ==========
        save_confusion_matrix(holdout_cm, cm_path)
    write_report_model_comparison(
        report_path=report_md_path,
        run_time=run_time,
        random_state=RANDOM_STATE,
        test_size=TEST_SIZE,
        features=list(X.columns),
        results=results,
    )
    # json report payload
    payload = {
        "run_time": run_time_str,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "features": list(X.columns),
        "models": results,
    }
    # save json report
    save_report_json(json_path, payload)

    # Print summary to console
    print("\n============= Metrics  ============== ")
    for model_name, metric_dict in results.items():
        print(f"\n Model: {model_name}")
        holdout_metrics = metric_dict["holdout"]
        print(f"Holdout Accuracy: {holdout_metrics['accuracy']}")
        print(f"Holdout F1 Score: {holdout_metrics['f1_score']}")
        print(f"Holdout ROC AUC: {holdout_metrics['roc_auc']}")

    print(f"\n============ Saved reports ============")
    print(f"Markdown report: {report_md_path}")
    print(f"JSON metrics: {json_path}")
    print("Confusion matrices:")
    for model_name in results:
        model_cm_path = figures_path / f"confusion_matrix_{model_name}_{run_tag}.png"
        print(f"- {model_name}: {model_cm_path}")


if __name__ == "__main__":
    main()
