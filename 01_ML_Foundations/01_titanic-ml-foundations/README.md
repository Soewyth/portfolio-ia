# Titanic — ML Foundations (Kaggle)

## Objective

Build a clean and reproducible Machine Learning pipeline to predict passenger survival on the Titanic dataset (binary classification).

---

## Dataset

- Source: Kaggle — _Titanic: Machine Learning from Disaster_
- Expected files (not committed to git):
  - `datasets/raw/train.csv`
  - `datasets/raw/test.csv`
  - (optional) `datasets/raw/gender_submission.csv`

### Download (Kaggle CLI)

```bash
kaggle competitions download -c titanic -p datasets/raw
unzip -o datasets/raw/titanic.zip -d datasets/raw
```

### Metrics

Baseline evaluation focuses on:

- Accuracy
- F1-score
- ROC-AUC

### Approach

- Data loading through a dedicated module: titanic_ml_foundations.data
- Feature selection through: titanic_ml_foundations.features
- Baseline model: Logistic Regression (first benchmark)
- Next steps (planned):
  - Proper preprocessing with ColumnTransformer + Pipeline
  - Stratified cross-validation
  - Confusion matrix + ROC curve saved in outputs/figures/
  - Short evaluation report saved in outputs/reports/

### Project Structure

```01_titanic-ml-foundations/
├── datasets/
│   └── raw/                 # Kaggle CSV files (ignored by git)
├── outputs/
│   ├── figures/             # plots (can be committed)
│   └── reports/             # reports/metrics
├── scripts/                 # runnable entry points
├── src/
│   └── titanic_ml_foundations/
│       ├── data/            # load + checks
│       ├── features/        # feature selection / preprocessing
│       ├── models/          # baseline models / pipelines
│       └── evaluation/      # metrics + plots
├── pyproject.toml
├── requirements.txt
└── README.md
```

### How to Run

1. Create and activate environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Install the project (editable)

```bash
   pip install -e .
```

3. Run baseline script

```bash
   python scripts/01_baseline_logreg.py
```

# Notes

Kaggle datasets are not committed. Download them and place them under datasets/raw/.
