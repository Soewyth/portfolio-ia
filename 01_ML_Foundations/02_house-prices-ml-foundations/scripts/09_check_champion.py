from __future__ import annotations

from house_prices_ml_foundations.config.paths import get_paths, get_project_root
from house_prices_ml_foundations.models.champion import build_champion_pipeline


def main() -> None:
    """Train RF on full training data and generate Kaggle submission."""
    root_dir = get_project_root()
    paths = get_paths(root_dir)

    rf_pipe, champion_source = build_champion_pipeline(paths["reports"])
    params = rf_pipe.get_params()

    print(" === Champion Pipeline ===")
    print("n_estimators : ", params["model__n_estimators"])
    print("model__max_features : ", params["model__max_features"])
    print("model__max_depth : ", params["model__max_depth"])


if __name__ == "__main__":
    main()
