from __future__ import annotations

import pandas as pd

from house_prices_ml_foundations.config.paths import get_paths, get_project_root
from house_prices_ml_foundations.data.load import load_train_test
from house_prices_ml_foundations.evaluation.plots import (
    plot_correlation_heatmap,
)
from house_prices_ml_foundations.features.build import get_target_name, make_features
from house_prices_ml_foundations.features.preprocess import build_preprocessor
from house_prices_ml_foundations.io.run_id import make_run_id


def main() -> None:
    """Run correlation features in model."""

    run_id = make_run_id()
    root_dir = get_project_root()
    paths = get_paths(root_dir)

    figure_path = paths["figures"]

    # Load the training data, feature X, and target y and preprocess fit
    train_df, _ = load_train_test(root_dir)
    X, y = make_features(train_df, return_target=True)
    target_name = get_target_name()

    # X_encoded from preprocessing fit
    preprocessor = build_preprocessor()
    X_encoded = preprocessor.fit_transform(X)
    # get feature names from preprocessor and create a DataFrame with the encoded features
    features_name = preprocessor.get_feature_names_out()
    X_encoded = pd.DataFrame(X_encoded, columns=features_name)

    df_corr = X_encoded.copy()  # avoid modifying the original X_encoded
    df_corr[target_name] = y.values

    top_10_corr = df_corr.corr()[target_name].abs().drop(target_name).sort_values(ascending=False).head(10).index

    heatmap_df = df_corr[list(top_10_corr) + [target_name]]

    plot_correlation_heatmap(df=heatmap_df, out_path=figure_path, run_id=run_id, title="Correlation heatmap of features with target")


if __name__ == "__main__":
    main()
