# Titanic Baseline Logistic Regression Model Report
**Run Time:** 2026-01-31 20:29:22

## Features Used :
- Pclass
- Sex
- Age
- SibSp
- Parch
- Fare
- Embarked

## Data Split Parameters
- **Random State:** `6`
- **Test Size:** `0.2`

## Models : 
### Pipeline : 
- **Preprocessing:** 
    - SimpleImputer (median for numerical, most_frequent for categorical)
    - OneHotEncoder (handle_unknown='ignore')
- **Classifier:** LogisticRegression (random_state=6)

## Model Evaluation Metrics
- **Accuracy:** `0.7933`
- **F1 Score:** `0.7132`
- **ROC AUC Score:** `0.8310`

## Confusion Matrix : 
Format : `[[TN,FP],[FN,TP]]` 

Saved at : `outputs/figures/confusion_matrix.png`
```
[[96 14]
 [23 46]]
```

## Cross-validation
- F1 Score (5-fold Stratified CV): Mean = `0.7168`, Std = `0.0419`
- ROC AUC Score (5-fold Stratified CV): Mean = `0.8508`, Std = `0.0318`

