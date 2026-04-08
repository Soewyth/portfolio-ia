from __future__ import annotations

from pathlib import Path

from house_prices_ml_foundations.config.config import (
    FIGURES_DIR,
    MLRUNS_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    REPORTS_DIR,
    SUBMISSIONS_DIR,
)


def get_project_root() -> Path:
    # Walk up from this file until we find pyproject.toml (project root marker)
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not find project root: no pyproject.toml found in any parent directory.")


def get_paths(root_dir: Path) -> dict:
    """Get a dictionary of important project paths, creating directories if they don't exist."""
    paths = {
        "datasets_raw": root_dir / "datasets" / "raw",
        "outputs": root_dir / OUTPUTS_DIR,
        "reports": root_dir / OUTPUTS_DIR / REPORTS_DIR,
        "models": root_dir / OUTPUTS_DIR / MODELS_DIR,
        "submissions": root_dir / OUTPUTS_DIR / SUBMISSIONS_DIR,
        "figures": root_dir / OUTPUTS_DIR / FIGURES_DIR,
        "mlruns": root_dir / OUTPUTS_DIR / MLRUNS_DIR,
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


def latest_file(directory: Path, pattern: str = "*") -> Path:
    """Get the latest file in a directory matching a pattern."""
    files = list(directory.glob(pattern))
    if not files:
        raise FileNotFoundError("Files not found at directory {directory} matching the pattern {pattern} ")
    latest = max(files, key=lambda f: f.stat().st_mtime)  # stat for last time modified file

    return latest
