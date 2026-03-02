# Titanic — ML Foundations (Classification)

classification pipeline reproducible(Scikit-Learn) with preprocessing (imputation + one-hot), holdout evaluation + cross-validation, comparison models,and versionnings artifacts (`metrics.json`, reports `.md`, figures `.png`).

## Project goals

- Build a clean, reusable ML pipeline (preprocess + model)
- Evaluate properly: holdout split + Stratified CV
- Compare at least two models (LogReg vs RandomForest)
- Produce human-readable + machine-readable outputs (MD + JSON)

## Tech stack

- Python
- Pandas
- Scikit-Learn
- Matplotlib (confusion matrix)

## Repository structure (simplified)

```text
src/titanic_ml_foundations/
  data/            # load + split
  features/        # schema + preprocess (ColumnTransformer)
  models/          # baseline + registry
  evaluation/      # cv + plots + reporting
scripts/
  01_baseline_logreg.py
  02_model_comparison.py
outputs/
  figures/
  reports/
datasets/
  raw/
    train.csv
    test.csv
```

## Setup

### 1) Create venv + install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) (Recommended) install as editable

From project root:

```bash
pip install -e .
```

## How to run

### Baseline (Logistic Regression)

Runs holdout evaluation + CV, saves confusion matrix + report + metrics JSON.

```bash
python scripts/01_baseline_logreg.py
```

Outputs (examples):

- `outputs/figures/confusion_matrix_*.png`
- `outputs/reports/baseline_report_*.md`
- `outputs/reports/metrics_*.json`

### Model comparison (LogReg vs RandomForest)

Compares multiple models using the same preprocessing and evaluation protocol.

```bash
python scripts/02_model_comparison.py
```

Outputs:

- `outputs/figures/confusion_matrix_<model>_*.png`
- `outputs/reports/comparison_report_*.md`
- `outputs/reports/metrics_*.json`

## Evaluation protocol

- Holdout: stratified train/valid split (`TEST_SIZE`, `RANDOM_STATE`)
- Cross-validation: `StratifiedKFold` (`N_SPLITS_CV`, `shuffle=True`, `RANDOM_STATE`)

Metrics:

- Holdout: Accuracy / F1 / ROC-AUC + confusion matrix
- CV: mean ± std for F1 and ROC-AUC

## Notes

- Kaggle `test.csv` has no labels (`y`), so metrics are computed only on `train.csv`.
- JSON metrics are machine-readable and facilitate run/model comparison.
- Datasets from Kaggle are not committed; place files in `datasets/raw/`.
