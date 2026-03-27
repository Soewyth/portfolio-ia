from __future__ import annotations

from house_prices_ml_foundations.evaluation.plots import (
    plot_abs_error_vs_ytrue,
    plot_residuals_hist,
    plot_ytrue_vs_ypred,
)
from house_prices_ml_foundations.config.paths import get_project_root, get_paths
from house_prices_ml_foundations.io.run_id import make_run_id
import pandas as pd


def main() -> None:
    """Run error analysis and generate plots for the latest model run."""
    run_id = make_run_id()
    root_dir = get_project_root()
    paths = get_paths(root_dir)

    path_reports = paths["reports"]
    path_figures = paths["figures"]
    # search last csv error analysis file in reports directory
    csv_errors_files = sorted(list(path_reports.glob("error_analysis_rf_pipe_*.csv")))
    if not csv_errors_files:
        print(" Not CSV files have been founded ! ")
    # recuperate the last one
    latest_csv_file = csv_errors_files[-1]

    # Load and execute plots functions
    error_df = pd.read_csv(latest_csv_file)

    plot_abs_error_vs_ytrue(error_df=error_df, out_path=path_figures, run_id=run_id)
    plot_residuals_hist(error_df=error_df, out_path=path_figures, run_id=run_id)
    plot_ytrue_vs_ypred(error_df=error_df, out_path=path_figures, run_id=run_id)


if __name__ == "__main__":
    main()
