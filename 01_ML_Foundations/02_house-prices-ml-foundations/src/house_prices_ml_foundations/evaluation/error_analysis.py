from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_INTERPRETABILITY_COLUMNS = [
    "OverallQual",
    "GrLivArea",
    "Neighborhood",
    "YearBuilt",
]


def build_error_analysis_frame(
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    y_pred: np.ndarray,
    raw_df: pd.DataFrame | None = None,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Build a machine-readable DataFrame with residuals and useful features.
        Priority for explanatory columns:
    1. X_valid
    2. raw_df aligned on index
    """

    if feature_columns is None:
        feature_columns = DEFAULT_INTERPRETABILITY_COLUMNS

    residual = y_valid - y_pred
    abs_error = abs(residual)

    pct_error = abs_error / y_valid.replace(0, np.nan)

    error_analysis_df = pd.DataFrame(
        {
            "row_id": X_valid.index,
            "y_true": y_valid,
            "y_pred": y_pred,
            "residual": residual,
            "abs_error": abs_error,
            "pct_error": pct_error,
        }
    )
    for col in feature_columns:
        if col in X_valid.columns:
            error_analysis_df[col] = X_valid[col]
        elif raw_df is not None and col in raw_df.columns:
            error_analysis_df[col] = raw_df.loc[X_valid.index, col]
        else:
            print(f"Warning: column '{col}' not found in X_valid or raw_df.")
    return error_analysis_df


def save_error_analysis_csv(error_analysis_df: pd.DataFrame, csv_path: Path) -> None:
    """Save error analysis DataFrame as CSV."""
    error_analysis_df.to_csv(csv_path, index=False)


def build_error_analysis_summary(error_analysis_df: pd.DataFrame) -> dict:
    """Build a summary dictionary with key error metrics."""
    summary = {
        "abs_error_max": float(error_analysis_df["abs_error"].max()),
        "abs_error_mean": float(error_analysis_df["abs_error"].mean()),
        "abs_error_median": float(error_analysis_df["abs_error"].median()),
        "abs_error_p99": float(error_analysis_df["abs_error"].quantile(0.99)),
        "n_large_errors": int((error_analysis_df["abs_error"] > 100000).sum()),
        "abs_error_p95": float(error_analysis_df["abs_error"].quantile(0.95)),
        "abs_error_p90": float(error_analysis_df["abs_error"].quantile(0.90)),
        "abs_error_std": float(error_analysis_df["abs_error"].std()),
        "pct_error_mean": float(error_analysis_df["pct_error"].mean(skipna=True)),
        "top_10_errors": error_analysis_df.sort_values("abs_error", ascending=False).head(10).to_dict(orient="records"),
    }
    return summary
