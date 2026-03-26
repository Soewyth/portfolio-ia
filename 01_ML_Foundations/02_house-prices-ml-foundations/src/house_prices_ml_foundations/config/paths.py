from __future__ import annotations
from pathlib import Path
from house_prices_ml_foundations.config.config import (
    OUTPUTS_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    SUBMISSIONS_DIR,
    MODELS_DIR,
)


def get_project_root() -> Path:
    """Get the root directory of the project based on the location of this file."""
    return Path(__file__).resolve().parent.parent.parent.parent  # up to project root


def get_paths(root_dir: Path) -> dict:
    """Get a dictionary of important project paths, creating directories if they don't exist."""
    paths = {
        "datasets_raw": root_dir / "datasets" / "raw",
        "outputs": root_dir / OUTPUTS_DIR,
        "reports": root_dir / OUTPUTS_DIR / REPORTS_DIR,
        "models": root_dir / OUTPUTS_DIR / MODELS_DIR,
        "submissions": root_dir / OUTPUTS_DIR / SUBMISSIONS_DIR,
        "figures": root_dir / OUTPUTS_DIR / FIGURES_DIR,
    }
    # Create directories if they don't exist but skip datasets/raw which should be manually managed
    for name, path in paths.items():
        if name == "datasets_raw":
            continue
        else:
            path.mkdir(parents=True, exist_ok=True)

    if not paths["datasets_raw"].exists():
        raise FileNotFoundError(
            f"Expected raw datasets directory not found at: {paths['datasets_raw']}. Please ensure it exists and contains the necessary data files."
        )

    return paths
