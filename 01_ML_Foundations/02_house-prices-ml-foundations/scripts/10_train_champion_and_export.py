from __future__ import annotations

from house_prices_ml_foundations.config.paths import get_paths, get_project_root
from house_prices_ml_foundations.data.load import load_train_test
from house_prices_ml_foundations.features.build import make_features
from house_prices_ml_foundations.io.model_artifacts import save_model
from house_prices_ml_foundations.io.run_id import make_run_id
from house_prices_ml_foundations.models.champion import build_champion_pipeline


def main() -> None:
    """Train the champion model and export it."""
    run_id = make_run_id(tag="champion_")

    root_dir = get_project_root()

    paths = get_paths(root_dir)
    versioned_model_path = paths["models"] / f"{run_id}.joblib"
    stable_model_path = paths["models"] / "champion.joblib"

    train_df, _ = load_train_test(root_dir)

    X_train, y_train = make_features(train_df)

    pipe, champion_source = build_champion_pipeline(paths["reports"])

    pipe.fit(X_train, y_train)

    save_model(pipe, versioned_model_path)
    save_model(pipe, stable_model_path)

    params = pipe.get_params()

    model_param = {k: v for k, v in params.items() if k.startswith("model__")}

    print(" === Champion model trained and exported === ")
    print(f"Champion source: {champion_source}")
    for key, value in model_param.items():
        print(f" {key} : {value}")

    print(f"Versioned model saved to: {versioned_model_path}")
    print(f"Stable model saved to: {stable_model_path}")


if __name__ == "__main__":
    main()
