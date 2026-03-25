from __future__ import annotations
from pathlib import Path


from house_prices_ml_foundations.models.champion import build_champion_pipeline


def main() -> None:
    """Train RF on full training data and generate Kaggle submission."""
    scripts_dir = Path(__file__).resolve().parent
    root_dir = scripts_dir.parent

    outputs_path = root_dir / "outputs"
    reports_path = outputs_path / "reports"

    reports_path.mkdir(parents=True, exist_ok=True)

    rf_pipe = build_champion_pipeline(reports_path)


    params = rf_pipe.get_params()

    print(" === Champion Pipeline ===")
    print("n_estimators : ", params["model__n_estimators"])
    print("model__max_features : ", params["model__max_features"])
    print("model__max_depth : ", params["model__max_depth"])


if __name__ == "__main__":
    main()
