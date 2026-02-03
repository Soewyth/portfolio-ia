from __future__ import annotations

from pathlib import Path
from datetime import datetime

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
from titanic_ml_foundations.models.baseline import build_baseline_model

# evaluations imports
from titanic_ml_foundations.evaluation.plots import save_confusion_matrix
from titanic_ml_foundations.evaluation.cv import cross_validate_model
from titanic_ml_foundations.evaluation.reporting import save_report_json
from titanic_ml_foundations.evaluation.reporting import write_reports_baseline


def main() -> None:
    """Main function to execute the baseline logistic regression model training"""

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
    cm_path = figures_path / f"confusion_matrix_{run_tag}.png"
    report_md_path = reports_path / f"baseline_report_{run_tag}.md"
    json_path = reports_path / f"metrics_{run_tag}.json"

    # ========== data ==========
    train_df, test_df = load_train_test(root_dir)
    X, y = make_X_y(train_df)

    # ========== split fixed ==========
    X_train, X_valid, y_train, y_valid = make_train_valid_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # ========== holdout (fixed split) ==========
    pipe = build_baseline_model()  # pipeline
    pipe.fit(X_train, y_train)  # fit the model

    y_predict = pipe.predict(X_valid)  # make predictions on validation set
    y_proba = pipe.predict_proba(X_valid)[
        :, 1
    ]  # get predicted probabilities for positive class (2nd column survived)

    # ========== Metrics ==========
    acc = accuracy_score(y_valid, y_predict)
    f1 = f1_score(y_valid, y_predict)
    roc_auc = roc_auc_score(y_valid, y_proba)
    cm = confusion_matrix(y_valid, y_predict)

    # ========== cross-validation (unfitted pipe and full data) ==========
    cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)

    # F1 Score
    cv_model_f1 = build_baseline_model()  # new pipeline for cross-validation
    mean_cv_score_f1, std_cv_score_f1 = cross_validate_model(
        cv_model_f1, X, y, cv=cv, scoring="f1"
    )
    # ROC AUC Score
    cv_model_auc = build_baseline_model()  # new pipeline for cross-validation
    mean_cv_score_auc, std_cv_score_auc = cross_validate_model(
        cv_model_auc, X, y, cv=cv, scoring="roc_auc"
    )

    # ========== artifacts saving ==========
    save_confusion_matrix(cm, cm_path)
    write_reports_baseline(
        report_md_path,
        run_time,
        acc,
        f1,
        roc_auc,
        cm,
        RANDOM_STATE,
        TEST_SIZE,
        list(X.columns),
        mean_cv_score_f1,
        std_cv_score_f1,
        mean_cv_score_auc,
        std_cv_score_auc,
    )

    # json report payload
    payload = {
        "run_time": run_time_str,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "features": list(X.columns),
        "holdout": {
            "accuracy": round(float(acc), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(roc_auc), 4),
            "confusion_matrix": cm.tolist(),
        },
        "cv": {
            "f1_score": {
                "mean": round(float(mean_cv_score_f1), 4),
                "std": round(float(std_cv_score_f1), 4),
            },
            "roc_auc": {
                "mean": round(float(mean_cv_score_auc), 4),
                "std": round(float(std_cv_score_auc), 4),
            },
        },
    }
    # save json report
    save_report_json(json_path, payload)

    # Print summary to console
    print("\n============= Metrics  ============== ")
    print(f"Holdout :  Accuracy={acc:.4f} | F1 Score={f1:.4f} | ROC AUC={roc_auc:.4f}")
    print(f"CV F1 Score: {mean_cv_score_f1:.4f}, Std: {std_cv_score_f1:.4f}")
    print(f"CV ROC AUC Score: {mean_cv_score_auc:.4f}, Std: {std_cv_score_auc:.4f}")

    print(
        f"\n============ Saved : {cm_path} | {report_md_path} | {json_path}  ============== "
    )


if __name__ == "__main__":
    main()
