# Titanic Model Comparison Report
**Run Time:** 2026-02-04 21:19:08

## Configuration
- **Random State:** `6`
- **Test Size:** `20%`
- **Cross-Validation:** 5-fold Stratified

## Features Used
- Pclass
- Sex
- Age
- SibSp
- Parch
- Fare
- Embarked

## Models Compared
- Logistic Regression (random_state=6)
- Random Forest Classifier (random_state=6)

### Pipeline Architecture
- **Preprocessing:** 
    - SimpleImputer (median for numerical, most_frequent for categorical)
    - OneHotEncoder (handle_unknown='ignore')

## Results

## LOGREG

### Holdout Metrics
- **Accuracy:** `0.7933`
- **F1 Score:** `0.7132`
- **ROC AUC Score:** `0.8310`

### Confusion Matrix
```
[[96, 14], [23, 46]]
```

### Cross-Validation (5-fold Stratified)
- **F1 Score:** Mean = `0.7168`, Std = `0.0419`
- **ROC AUC Score:** Mean = `0.8508`, Std = `0.0318`

## RF

### Holdout Metrics
- **Accuracy:** `0.8045`
- **F1 Score:** `0.7368`
- **ROC AUC Score:** `0.8437`

### Confusion Matrix
```
[[95, 15], [20, 49]]
```

### Cross-Validation (5-fold Stratified)
- **F1 Score:** Mean = `0.7419`, Std = `0.0521`
- **ROC AUC Score:** Mean = `0.8573`, Std = `0.0198`

